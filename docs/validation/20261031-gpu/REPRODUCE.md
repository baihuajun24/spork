# Reproduce the GPU smoke checks

Use the code commit and file hashes in the accompanying receipts. This is a
small functional check, not a performance benchmark or a quality equivalence
study. Original paper results are unchanged.

Use the current PR checkout to rerun these commands. Historical commit IDs
in receipts identify the original runs; those preparation commits are not
required for reproduction. The [code-equivalence manifest](code-equivalence.json)
checks the included runtime files against each historical execution commit.
The original receipt files are unchanged. No new GPU measurement was made
while extracting this update. The new D3 bridge is outside this PR.

## Dependencies and allocation

Obtain trusted copies of the official BFCL snapshot
`ea13468e4423454d0c213704fb87cf7cb3990433` and tau2-bench snapshot
`2174a603f6d014ef94473ffa95957f6ce27100db`. Data/tool hashes identify the
checked source inventory; they are not a file-access trace. Install `requirements-bfcl.txt` in the BFCL
client environment. The tau2 client additionally needs the tau2 dependencies
and Python 3.12. Model servers use the separate runtimes recorded in the receipts.
The available tau2-bench checkout includes newer τ³-bench additions and task
fixes. Its equivalence to the original paper's benchmark revision has not been
verified; this is a client workflow check on the pinned snapshot.

```bash
export MODEL_ROOT=/your/local/models
export BFCL_ROOT=/your/local/bfcl-snapshot
export SPORK_TAU2_ROOT=/your/local/tau2-bench
export CLIENT_PYTHON=/your/client/bin/python
export TAU_CLIENT_PYTHON=/your/tau-client/bin/python
export SERVER_PYTHON=/your/vllm024/bin/python
export GLM_SERVER_PYTHON=/your/native-glm/bin/python
export GPU_IDS=0,1,2,3
export PORT=18231
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
export HF_HUB_DISABLE_TELEMETRY=1 VLLM_NO_USAGE_STATS=1 DO_NOT_TRACK=1
export LITELLM_LOCAL_MODEL_COST_MAP=True
export NO_PROXY=127.0.0.1,localhost no_proxy=127.0.0.1,localhost
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
# Match the local compilation limits; reserve host memory independently.
export MAX_JOBS=1 FLASHINFER_NVCC_THREADS=1
export CMAKE_BUILD_PARALLEL_LEVEL=1 TORCHINDUCTOR_COMPILE_THREADS=1
```

The run host has H20-3e GPUs with 143771 MiB per GPU. Qwen3 and GLM use four
GPUs; Qwen3.5 and DeepSeek use two. Before launching, verify GPU ownership,
cooperative GPU locks and shared host memory. The local run additionally used
a host memory supervisor with a 48 GiB startup allowance for Qwen and
32 GiB for the native-model queue, plus a 20% host-memory reserve. Native
checkpoints load one shard at a time: the largest per-wave input buffers are
6.878 GiB for V4 TP2 and 19.987 GiB for GLM TP4. These are conservative
eager-loader input-size estimates, not measured RAM peaks; the V4 retry uses
lazy loading. The Qwen primary allowed one compiler job per worker;
the Qwen3.5 companion allowed three, with one NVCC thread per job. Compiler
budgets for other cases are recorded in the receipts. A companion service
budgets its jobs jointly with the primary. The supervisor stopped only its own process tree.
Those operational wrappers are not model or SPORK dependencies.

The host's existing optional pyarrow extension required an import workaround:
its server-only `sitecustomize.py` set `sys.modules['pyarrow'] = None`. The shim
hash is recorded. A working clean environment does not require this workaround.
Do not modify a shared installation to reproduce it.

## Model servers

Use the recorded checkpoint hashes. All cases use eager execution, disabled
prefix caching, four maximum sequences, 2048 batched tokens and GPU memory
utilization 0.75. These deliberately differ from performance-tuned serving
settings. To reproduce each model separately, allocate its GPU count and set its port;
the checks do not require concurrent services. The unmodified receipts describe
the original allocations, including a separate D3 service outside this PR.
Account for other shared-host jobs when reserving resources.

Qwen3-32B:

```bash
CUDA_VISIBLE_DEVICES="$GPU_IDS" "$SERVER_PYTHON" -m vllm.entrypoints.openai.api_server \
  --model "$MODEL_ROOT/Qwen3-32B" --served-model-name qwen3-32b \
  --tensor-parallel-size 4 --host 127.0.0.1 --port "$PORT" \
  --max-model-len 40960 --max-num-seqs 4 --gpu-memory-utilization 0.75 \
  --enforce-eager --max-num-batched-tokens 2048 --no-enable-prefix-caching \
  --dtype bfloat16
```

Qwen3.5-35B-A3B:

```bash
# GPU_IDS=4,5 PORT=18232 for this case.
export MAX_JOBS=3 CMAKE_BUILD_PARALLEL_LEVEL=3 TORCHINDUCTOR_COMPILE_THREADS=3
CUDA_VISIBLE_DEVICES="$GPU_IDS" "$SERVER_PYTHON" -m vllm.entrypoints.openai.api_server \
  --model "$MODEL_ROOT/Qwen3.5-35B-A3B" --served-model-name qwen35-a3b \
  --tensor-parallel-size 2 --host 127.0.0.1 --port "$PORT" \
  --max-model-len 65536 --max-num-seqs 4 --gpu-memory-utilization 0.75 \
  --enforce-eager --max-num-batched-tokens 2048 --no-enable-prefix-caching \
  --dtype auto --gdn-prefill-backend triton --safetensors-load-strategy lazy
```

DeepSeek-V4-Flash-0731:

Use `lazy` for checkpoint loading in this runtime. The first attempt used
`--safetensors-load-strategy eager`, which calls the safetensors 0.8.0 bytes
loader and raises `KeyError: 'F8_E8M0'`. The retry below uses `safe_open` through
the `lazy` strategy: all 48 shards loaded and the GPU server became ready.
The execution-mode `--enforce-eager` flag remains in effect; it is independent
of the checkpoint loader. The [first failure](v4-startup-failure.json) is
retained separately from the retry in [service receipts](receipts.json).

```bash
# GPU_IDS=0,1 PORT=18231 for the V4 retry.
export MAX_JOBS=1 CMAKE_BUILD_PARALLEL_LEVEL=1 TORCHINDUCTOR_COMPILE_THREADS=1
CUDA_VISIBLE_DEVICES="$GPU_IDS" "$SERVER_PYTHON" experiments/serving/vllm024/serve_v4.py \
  --model "$MODEL_ROOT/DeepSeek-V4-Flash-0731" --served-model-name deepseek-v4-flash-0731 \
  --tensor-parallel-size 2 --host 127.0.0.1 --port "$PORT" \
  --max-model-len 65536 --max-num-seqs 4 --gpu-memory-utilization 0.75 \
  --enforce-eager --max-num-batched-tokens 2048 --no-enable-prefix-caching \
  --kv-cache-dtype fp8 --enable-auto-tool-choice \
  --tool-call-parser deepseek_v4 --reasoning-parser deepseek_v4 \
  --safetensors-load-strategy lazy
```

GLM-5.3-Flash (native GLM runtime, not the vLLM 0.24.0 environment):

```bash
CUDA_VISIBLE_DEVICES="$GPU_IDS" "$GLM_SERVER_PYTHON" -m vllm.entrypoints.openai.api_server \
  --model "$MODEL_ROOT/GLM-5.3-Flash" --served-model-name glm-5.3-flash \
  --tensor-parallel-size 4 --host 127.0.0.1 --port "$PORT" \
  --max-model-len 65536 --max-num-seqs 4 --gpu-memory-utilization 0.75 \
  --enforce-eager --max-num-batched-tokens 2048 --no-enable-prefix-caching \
  --dtype bfloat16 --language-model-only --no-enable-flashinfer-autotune \
  --enable-auto-tool-choice --tool-call-parser glm47 --reasoning-parser glm45 \
  --safetensors-load-strategy eager
```

Wait for `/health` and confirm `/v1/models` before running a client. There was
no discarded model warmup; first-use costs are retained. Task wall time excludes
server startup and official scoring. GPU allocation time includes startup and
all intervals while the server was held.

The weight-hash inventory ran concurrently with the early Qwen tasks and used
CPU/storage resources. Recorded wall times therefore include uncontrolled
shared-host activity and must not be used as speedup estimates.

## BFCL client

Use `FAMILY=qwen`, `MODEL_NAME=qwen3-32b`, `N=10` for Qwen; other models use
`N=5` with the same first five task IDs. Corresponding families are `qwen35`,
`deepseek_v4` and `glm53`. Client-side Qwen3-32B rendering is intentional for
both Qwen models; serving always uses the target model's own tokenizer.

```bash
export FAMILY=qwen MODEL_NAME=qwen3-32b N=10
export OUT=results/gpu-smoke-qwen
CUDA_VISIBLE_DEVICES='' "$CLIENT_PYTHON" experiments/bfcl/bfcl_spork_eval.py \
  --official-root "$BFCL_ROOT" --task-ids-file experiments/configs/gpu_smoke10.json \
  --n "$N" --model-url "http://127.0.0.1:$PORT/v1" \
  --model-name "$MODEL_NAME" --model-family "$FAMILY" \
  --configs baseline,portfolio --probe-token-targets 1,16 \
  --max-tokens 4096 --context-window-tokens 32768 --max-steps-per-user-turn 10 \
  --tokenizer-path "$MODEL_ROOT/Qwen3-32B" --output "$OUT"
"$CLIENT_PYTHON" experiments/validation/summarize_smoke.py "$OUT" \
  --output results/gpu-smoke-qwen-compact.json
```

For native chat families, omit `--tokenizer-path`. Defaults also in force:
temperature 0, thinking on, seed 1234, 256 probe tokens and a 2-second tool latency floor.
The runner interleaves baseline then portfolio for each task, without repeated
runs. Every config gets an independent simulator world. Quality is scored by
the official multi-turn and irrelevance checkers; it is not inferred from
probe matching. Missing/unscored records remain null.
The task list contains condition variants of shared scenarios; different IDs
must not be treated as statistically independent samples.

The compact export omits prompts, answers, tool results and local paths. It
checks recorded baseline isolation and selected-call equality. It cannot
independently prove discarded simulator state or hidden server activity.
A BFCL error can omit the failing turn: its counters cover recorded completed
turns only, and token totals are incomplete. They must not be interpreted as
the total work of the failed request. The unified tau2 runner replaces its
partial trace on errors/timeouts; the compact export keeps those counters null.
A completion reaching its recorded token budget is flagged as an inferred
budget hit; it is not an observed server `finish_reason`.

## Qwen v1.0/v1.1 tau2 regression

Create a separate v1.0 source directory with `git archive 31d5ab6`, then run
both runners against the same plain Qwen3-32B server:

```bash
export OLD_SOURCE=/your/extracted/v1.0
export SOURCE="$OLD_SOURCE" OUT=results/tau-v10
# Repeat with SOURCE="$PWD" and a fresh OUT=results/tau-v11.
CUDA_VISIBLE_DEVICES='' "$TAU_CLIENT_PYTHON" "$SOURCE/spork_unified_eval.py" \
  --model-url "http://127.0.0.1:$PORT/v1" --model-name qwen3-32b \
  --tokenizer-path "$MODEL_ROOT/Qwen3-32B" --benchmark tau2 \
  --configs baseline,d1_d2 --n 10 --seed 1234 --max-tokens 32768 \
  --max-turns 8 --tau2-floor 2 --task-timeout-s 240 \
  --server-tp 4 --server-max-model-len 40960 \
  --server-tag gpu-smoke-eager-vllm024 --results-root "$OUT"
```

The exact ten task/domain IDs are in
`experiments/configs/tau_gpu_smoke10.json`; all ten are retail tasks. This is
a client-code regression on vLLM 0.24.0, not validation of the legacy
vLLM 0.19.1 D3 backend. The unified runner uses static user instructions and an
ACTION-match guardrail, not the full tau2 interactive user/evaluation protocol.
Timeouts and errors remain excluded from quality/latency conclusions even if
the old runner emits an action-match value for them.

Execution order is v1.0 baseline, v1.0 D1+D2, v1.1 baseline, then v1.1
D1+D2 on the same server. There are no balanced repetitions; this checks
workflow compatibility, not performance differences between code versions.
Both configurations cap each main completion at 8192 tokens. The eight-turn
limit permits the runner's default additional forced-conclusion turn with
tools removed. Legacy D1+D2 defaults remain in force: at most five probes,
160 tokens per probe, 50 stream-delta retry spacing, sentence snapping on,
confidence threshold 0.90, and stop probing after dispatch. Candidate pooling
and dispatch-all are off. These settings differ from BFCL's T1+T16 portfolio.

To make the compact export, point at the generated `tau2_...` directory:

```bash
python experiments/validation/summarize_tau_regression.py "$RAW_TAU_FOLDER" \
  --manifest experiments/configs/tau_gpu_smoke10.json \
  --code-commit "$RUNNER_COMMIT" --output results/tau-compact.json
```
