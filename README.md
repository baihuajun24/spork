# SPORK

### Self-Speculative Forking to Accelerate Agentic LLM Inference

📄 **Paper:** [arXiv:2607.03333](https://arxiv.org/abs/2607.03333)

SPORK is a self-speculative tool-call probe for LLM agents. While the main
model is still decoding its chain-of-thought, a lightweight probe speculatively
predicts the upcoming tool call, dispatches it early, and **overlaps tool
execution with the remaining main decode**. When the probe is right, the tool
result is already available by the time the main model finishes its turn,
hiding tool latency behind decode time.

<p align="center">
  <img src="docs/fig_method_overview.png" alt="A baseline turn vs. a SPORK turn" width="900">
</p>
<p align="center"><sub><b>A baseline turn vs. a SPORK turn.</b>
D1 forks a lightweight probe at decode start and dispatches the predicted tool
call early; D2 gates dispatch on probe confidence; the tool then runs under the
remaining CoT decode (<i>t_overlap</i>). On accept, the result is already
waiting at main stop; on reject, D3 recycles the probe's tokens as
speculative-decoding drafts for the serial fallback.</sub></p>

The probe runs against a vLLM OpenAI-compatible endpoint using `/v1/completions`
logprobs, and is most effective in **think mode** (Qwen3 emits multi-second
`<think>...</think>` CoT), which gives the probe enough decode time to finish
early and the tool time to overlap.

<p align="center">
  <img src="docs/fig_teaser.png" alt="Tool intent is visible early" width="620">
</p>
<p align="center"><sub><b>Why it works: tool intent is visible early.</b>
A probe forked at the main thread's first token predicts the exact tool call
that the main thread only emits ~2,000 CoT tokens later.</sub></p>

## Results (from the paper)

SPORK reduces end-to-end agent **tail latency** by overlapping tool execution with
decode. Headline P95 results **from the paper** (full serving setup — see the paper
for complete tables, models, and ablations):

| Benchmark | Model | N | P95 latency vs baseline |
|---|---|---|---|
| τ²-bench | Qwen3.5-35B-A3B | 155 | **1.19×** (−16%) |
| GAIA | Qwen3-32B | 165 | **1.22×** (−18%) |
| HotpotQA | Qwen3.5-35B-A3B | 200 | **1.25×** (−19.7%, D1+D2) |

Quality is preserved (action-match / EM·F1 neutral within ±1–2 points). τ²-bench
Qwen3-4B also shows a 1.15× mean speedup (N=155).

**What this code reproduces.** This release faithfully reproduces the SPORK
*mechanism* — the probe predicts the next tool, dispatches it early, and overlaps it
with decode (verified at ~80% probe acceptance and per-task tool-overlap on
Qwen3-32B + τ²-bench). Exact latency gains are **serving- and N-dependent**: measure
with enough tasks that per-pass CoT-length variance averages out, and serve with
adequate parallelism so the probe does not contend with the main decode.
**τ²-bench is the most directly reproducible** here (public tau2-bench tools);
the paper's GAIA/HotpotQA used internal search services, so this release ships a
built-in Wikipedia backend for HotpotQA and a bring-your-own backend for GAIA
(see **Benchmarks & tool backends**). The D3 draft-injection numbers in the paper
are from the engine-mode integration; the HTTP `launch_vllm_qwen_spork.sh` path is
provided as a reference implementation of the same idea.

## Configs

`spork_unified_eval.py` evaluates a baseline against progressively richer SPORK
variants via `--configs`:

- `baseline` — no speculation; tool calls dispatched after the turn completes.
- `d1` (alias `spork`) — speculative probe + early tool dispatch (D1).
- `d1_d2` — D1 plus adaptive re-probing during decode (D2).
- `d1_d2_d3` — D1+D2 plus draft-token injection to accelerate main decode (D3).
- `portfolio` — T1+T16 candidate pool: dispatch parsed candidates on isolated
  tool snapshots, then commit the earliest-dispatched candidate with a strict
  tool-name and typed-argument match. Other candidate states are discarded.

```bash
python3 spork_unified_eval.py \
  --model-url http://localhost:8000/v1 --model-name qwen3-32b \
  --model-family qwen --tokenizer-path Qwen/Qwen3-32B \
  --benchmark tau2 --configs baseline,portfolio --n 10 \
  --probe-token-targets 1,16
```

`--probe-after-tokens` controls the single D1 fork. For retry configs,
`--probe-token-targets` supplies explicit depths; `--candidate-pool` retains
the candidates, and `--dispatch-all-parsed` bypasses the confidence filter.
The `portfolio` preset enables both pool options and defaults to `1,16`.
Depths count output-stream deltas, **not exact tokenizer positions**. Baseline
continues to run without probes or speculative tool calls. Candidate pools
require an executor with isolated speculative state.

**Serve-mode requirement:** `baseline`/`d1`/`d1_d2` run on the **plain** server
(`launch_vllm_qwen.sh`); `d1_d2_d3` requires the **SPORK-HTTP** server
(`launch_vllm_qwen_spork.sh`, which adds ngram spec-dec + the draft-injection
sidecar). See **Serving the model**.

## Model adapters and configurable candidates

This update adds native DeepSeek/GLM adapters, configurable fork
depths, candidate pools, robust tool-call history, and main/probe usage records.
Start with `python3 examples/offline_minimal.py` for a CPU-only example.
Original config fields retain their order and defaults; new strategy fields
are appended and disabled by default. `--model-family auto` retains the v1.0
Qwen adapter selection. Streaming completion requests now explicitly ask for
usage with `stream_options.include_usage`; other default request fields remain
unchanged. Reasoning-specific request options are enabled explicitly for the
new chat model families. This update also fixes missing chat usage propagation:
both streaming main responses and nonstreaming chat probes retain usage when
the endpoint provides it; absent usage remains unknown.

| Model | Explicit family / tool grammar | Serving setup used for development |
|---|---|---|
| Qwen3 | `qwen`, JSON | Legacy vLLM 0.19.1 path; newer example path prepared for 0.24.0 |
| Qwen3.5-35B-A3B | `qwen35`, JSON with an explicitly compatible renderer tokenizer | vLLM 0.24.0; serving tokenizer owns the 248058 boundary ID |
| DeepSeek-V4-Flash | `deepseek_v4`, native chat / DSML | vLLM 0.24.0; `deepseek_v4` tool and reasoning parsers; continuation patch below |
| GLM-5.3-Flash | `glm53`, native chat / XML | vLLM `0.28.1rc1.dev451+g1970f3ed4`, Transformers 5.16.1; `glm47` tool parser and `glm45` reasoning parser |

These are separate serving environments, not a single dependency lock. Fresh
GPU smoke observations and remaining checks are recorded in the
[validation checklist](docs/VALIDATION.md) and the
[GPU report with reproduction commands](docs/validation/20261031-gpu/README.md).
The small samples include quality failures and do not establish speedups or
quality equivalence.
The existing results table above is unchanged. No new cross-model performance
claim is attached to these adapter additions.

For native chat families, pass `--model-family deepseek_v4` or `glm53` and the
corresponding served model name. A client-side tokenizer is optional there;
`--tokenizer-path none` uses server rendering. Qwen families require a tokenizer
whose rendered tool grammar matches the adapter. Explicit `qwen35` selects the
JSON adapter; `auto` retains the older name-based XML adapter selection.
The legacy `d1_d2_d3` and cross-model entrypoints remain Qwen-only.
The new request-scoped D3 bridge is deferred to a separate update. Full
GLM-5.3 is not included in the validated support matrix.

Additional examples:

- [Standalone BFCL runner and official scoring](experiments/bfcl/README.md).
- [Optional vLLM 0.24 V4 continuation patch](experiments/serving/vllm024/README.md).
- [CPU regression tests and GPU smoke observations](docs/VALIDATION.md).

## Prerequisites

- **Python 3.10+**.
- **vLLM** with an OpenAI-compatible server (see `launch_vllm_qwen.sh`).
- **A Qwen3 model** (e.g. `Qwen/Qwen3-32B`). Other Qwen3 sizes work too.
- **tau2-bench** installed and importable on `PYTHONPATH` (its `src/` on the
  path). Point `SPORK_TAU2_ROOT` at your checkout if it is not under the repo.
- Python deps from `requirements.txt` (`pip install -r requirements.txt`).
- For `hotpotqa`: a local dataset JSONL (Wikipedia backend is built in — no key).
  For `gaia`: a local dataset JSONL **and** your own web search/browse backend
  (see **Benchmarks & tool backends** below).

## Serving the model

There are **two serve modes**, depending on which SPORK configs you want to run:

**(a) Plain serve — for `baseline`, `d1`, `d1_d2`** (the HTTP probe + tool-overlap path):

```bash
MODEL_PATH=Qwen/Qwen3-32B PORT=8000 TP=4 ./launch_vllm_qwen.sh
```

**(b) SPORK-HTTP serve — additionally enables `d1_d2_d3`** (D3 = ngram speculative
decoding + a `/spork/*` sidecar that injects the probe's predicted tool-call
tokens at the boundary to accelerate the main decode):

```bash
MODEL_PATH=Qwen/Qwen3-32B PORT=8000 TP=4 ./launch_vllm_qwen_spork.sh
```

`MODEL_PATH`, `PORT`, `TP`, `MAX_LEN`, `GPU_UTIL` (and `NSPEC` for mode b) are all
overridable via environment variables. Mode (b) also serves `baseline`/`d1`/`d1_d2`
(D3 is simply unused for them), but its `baseline` runs *with* ngram self-speculation
(an "ngram baseline") — so for a clean D3 ablation compare `d1_d2_d3` against the
`baseline` measured on the **same** (mode b) server.

## Running an evaluation

```bash
PYTHONPATH=. python3 spork_unified_eval.py \
    --model-url http://localhost:8000/v1 \
    --model-name qwen3-32b \
    --benchmark tau2 \
    --configs baseline,d1_d2 \
    --n 40 \
    --enable-thinking
```

This runs 40 tau2 tasks for both the `baseline` and `d1_d2` configs and writes
per-config results under the results root (override with `--results-root`).
Add `d1` and/or `d1_d2_d3` to `--configs` to compare more variants.

`--benchmark` accepts `tau2` (default), `gaia`, and `hotpotqa`. **HotpotQA runs
out-of-the-box** via a built-in Wikipedia backend (no API key); **GAIA** needs a
dataset path **and** your own search/browse backend (see **Benchmarks & tool
backends**):

```bash
PYTHONPATH=. python3 spork_unified_eval.py \
    --model-url http://localhost:8000/v1 --model-name qwen3-32b \
    --benchmark gaia --configs baseline,d1_d2 --n 165 \
    --gaia-dataset-path ./datasets/gaia_validation.jsonl

PYTHONPATH=. python3 spork_unified_eval.py \
    --model-url http://localhost:8000/v1 --model-name qwen3-32b \
    --benchmark hotpotqa --configs baseline,d1_d2 --n 200 \
    --hotpotqa-dataset-path ./datasets/hotpotqa_validation.jsonl
```

## Benchmarks & tool backends

This release demonstrates SPORK's **latency-overlap mechanism**. We use these
benchmarks as realistic tool-agent **workloads** for latency measurement, not as
official leaderboard scores — the reported success is only a *same-harness
`baseline` ≈ `d1_d2` quality-preservation check* (see the note on tau2 below).

| benchmark | out of the box? | tool backend |
|---|---|---|
| **tau2** | ✅ yes — install tau2-bench | tau2-bench's own tool environment (real DB ops). We run a **single-turn, no-user-simulator** harness, so success is **action-match**, *not* τ²-bench's official user-simulator (DB×communicate) reward — don't compare it to the τ²-bench leaderboard. |
| **HotpotQA** | ✅ yes — **no API key** | built-in **Wikipedia** backend (`WikipediaSearchBackend`). HotpotQA is Wikipedia QA, so this is faithful and reproducible. Uses the public MediaWiki API (honors `http(s)_proxy` env vars). |
| **GAIA** | ⚠️ bring your own | placeholder `WebSearchBackend` / `WebBrowseBackend` (raise `NotImplementedError`). GAIA needs general web search/browse — plug in SerpAPI / Bing / a headless browser in [`tool_backends.py`](tool_backends.py) and it will run. The paper used internal search services not included here. |

The GAIA/HotpotQA tool **schemas**, system prompts, task loaders (local JSONL),
and the EM/F1 scorer are all included. To add your own backend, subclass
`WebSearchBackend.search(query, top_k) -> list[{title, url, snippet}]` and/or
`WebBrowseBackend.browse(url, goal) -> str` in `tool_backends.py`.

Datasets are read from local JSONL: HotpotQA expects `question`/`answer` fields;
GAIA expects `Question`/`Final answer`. Paths default to
`./datasets/{gaia,hotpotqa}_validation.jsonl` and are overridable via
`--gaia-dataset-path` / `--hotpotqa-dataset-path` (or `SPORK_GAIA_DATASET` /
`SPORK_HOTPOTQA_DATASET`).

## Citation

If you use SPORK in your research, please cite:

```bibtex
@misc{bai2026spork,
  title         = {SPORK: Self-Speculative Forking to Accelerate Agentic LLM Inference},
  author        = {Bai, Huajun and Lv, Weiwei and Zheng, Huichuan and Lu, Youyou and Shu, Jiwu},
  year          = {2026},
  eprint        = {2607.03333},
  archivePrefix = {arXiv},
  primaryClass  = {cs.DC}
}
```
