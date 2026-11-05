# BFCL v3 multi-turn example

This runner reads a separately obtained official BFCL checkout. It has no
dependency on a private adapter package or a server scheduler. It starts no
model server. Use a model endpoint you have already provisioned.

The expected source layout is `bfcl_eval/data/` and
`bfcl_eval/eval_checker/multi_turn_eval/` below `--official-root`. The example
was prepared against the official Gorilla/BFCL snapshot
`ea13468e4423454d0c213704fb87cf7cb3990433`. Obtain the
[official source](https://github.com/ShishirPatil/gorilla/tree/main/berkeley-function-call-leaderboard)
and install the client/simulator dependencies with
`pip install -r requirements-bfcl.txt` in the client environment.
Data and upstream code retain their own licenses and are not vendored here.

Obtain the pinned source separately (the repository root and BFCL root differ):

```bash
git clone https://github.com/ShishirPatil/gorilla.git "$BFCL_CHECKOUT"
git -C "$BFCL_CHECKOUT" checkout ea13468e4423454d0c213704fb87cf7cb3990433
export BFCL_ROOT="$BFCL_CHECKOUT/berkeley-function-call-leaderboard"
python3 -m pip install -r requirements-bfcl.txt
```

For an offline starting point, run `python3 examples/offline_minimal.py`.
Before the GPU example below, provision a plain server using the root README
or the [tested model-specific commands](../../docs/validation/20261031-gpu/REPRODUCE.md#model-servers).

`../configs/bfcl_holdout100.json` contains public task IDs only: 25 tasks per
multi-turn category, indices 64–88. No task text, answers or model outputs are
included. A CPU-only preflight checks the selected IDs and hashes the data,
simulators and evaluator sources:

```bash
python3 experiments/bfcl/bfcl_spork_eval.py \
  --official-root "$BFCL_ROOT" --n 100 --describe
```

For a small functional run against a plain Qwen server:

```bash
python3 experiments/bfcl/bfcl_spork_eval.py \
  --official-root "$BFCL_ROOT" \
  --model-url http://localhost:8000/v1 --model-name qwen3-32b \
  --model-family qwen --tokenizer-path Qwen/Qwen3-32B \
  --configs baseline,portfolio --probe-token-targets 1,16 \
  --n 10 --tool-latency-s 2 --output results/bfcl-smoke
```

The output directory must be new. Each task/config gets an independent tool
world. Baseline does not execute probes or speculative work; portfolio uses
isolated snapshots and strict name/typed-argument matching. For chat families,
use `--model-family deepseek_v4` or `glm53`; no client-side tokenizer is needed.
For `qwen35`, explicitly provide a compatible JSON renderer tokenizer and serve
the model with its own tokenizer. The model name alone does not choose a tool
format. The defaults use thinking on, seed 1234, output-stream trigger depths
1/16 and a controlled 2-second tool floor; record changes explicitly.

Tasks/configs run serially. Task wall time includes model and tool execution
and excludes scoring. `protocol.json` records source hashes, task selection,
generation settings and code commit/dirty state. Set
`SPORK_HARDWARE_DESCRIPTION` to record the endpoint's hardware and TP allocation;
the client cannot infer remote hardware. Preserve the server launch command,
runtime version and model/tokenizer revision alongside the run when comparing
performance. The example does not establish reproducible performance by itself.

The runner calls the unmodified official `multi_turn_checker` and
`multi_turn_irrelevance_checker` on the recorded canonical calls. Errors and
truncated tasks are unscored; scorer exceptions are retained separately.
`correct/scored` is reported alongside complete/error/incomplete/unscored counts.
Strict probe matching does not establish task quality. This is a local example
protocol, not an official leaderboard submission harness.

The frozen MessageAPI schema includes an optional `message_id` that its
`delete_message` implementation does not accept. When that exact mismatch is
present, the example removes the optional schema field and records the change
in every affected task record. It does not alter the simulator or gold calls.

Baseline supports native multi-call decisions. The portfolio example explicitly
rejects native multi-call responses rather than evaluating a truncated trace.
The source checkout must be trusted: simulator code and the official evaluator
execute locally. Raw result directories are not intended for publication.
