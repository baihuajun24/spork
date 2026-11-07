# Model adapter and candidate portfolio validation

This update is based on public v1.0 (`31d5ab6`). Original results and figures
are unchanged. New strategies are opt-in. The new D3 bridge is outside this
update; the original public D3 implementation is unchanged.

## CPU checks

On 2026-11-05, all **18 tests passed**, including the optional official BFCL
integration. All **31 Python files** compiled. The offline example, both CLI
help commands and the 100-task BFCL source preflight passed. No GPU or model
server was started for these checks.

```bash
CUDA_VISIBLE_DEVICES='' python3 -m unittest discover -s tests -v
CUDA_VISIBLE_DEVICES='' python3 examples/offline_minimal.py
python3 spork_unified_eval.py --help
python3 experiments/bfcl/bfcl_spork_eval.py --help

# Optional integration against a trusted, separately obtained official source.
SPORK_BFCL_OFFICIAL="$BFCL_ROOT" CUDA_VISIBLE_DEVICES='' \
  python3 -m unittest discover -s tests -p test_bfcl_official.py -v
python3 experiments/bfcl/bfcl_spork_eval.py \
  --official-root "$BFCL_ROOT" --n 100 --describe
```

Without `SPORK_BFCL_OFFICIAL`, two official-source tests are skipped. See the
[BFCL example](../experiments/bfcl/README.md) for source setup and runnable
baseline/portfolio commands. The preflight loads no model and calls no server.

Tests cover original configuration field order/defaults, default request bodies
with the documented streaming-usage extension, baseline isolation, strict
name/typed-argument commit matching, mismatch fallback, later matching
candidates, native baseline multi-calls, malformed argument history,
reasoning/usage capture, and the optional V4 continuation patch. The official
CPU checks compare 12 tasks' teacher-call results, states and complete traces
with the upstream simulators and checkers. These are implementation checks,
not measured model accuracy.

## Recorded GPU checks

The [2026-10-31 GPU report](validation/20261031-gpu/README.md) includes
[reproduction commands](validation/20261031-gpu/REPRODUCE.md), runtime and
checkpoint fingerprints, and text-free per-task records. Original JSON
receipts are preserved byte-for-byte; the
[source-equivalence manifest](validation/20261031-gpu/code-equivalence.json)
checks the included runtime against the historical execution commits. These
are historical checks, not new measurements on the extracted branch.

| Path | Observed functional evidence | Quality and validation limits |
|---|---|---|
| Qwen3-32B portfolio | BFCL 10 IDs per configuration; 88 exact commits, 12 fallbacks, zero recorded decision violations | Baseline 3 pass / 6 fail / 1 unscored; portfolio 4 / 6 / 0. A paired regression is retained. GPU discarded-state isolation is not independently audited. |
| Qwen3.5-35B-A3B | BFCL 5 IDs per configuration; 42 exact commits, 1 fallback, usage retained | Baseline 2 / 3 / 0; portfolio 3 / 2 / 0. Requires an explicit compatible JSON renderer. |
| DeepSeek-V4-Flash | Lazy-loader retry loaded 48 shards; BFCL 5 IDs per configuration, 108 parsed probes, 46 exact commits | Each configuration: 2 / 2 / 1. Initial loader failure and a slower portfolio case are retained. |
| GLM-5.3-Flash | BFCL 5 IDs per configuration; native calls, 108 parsed probes, 39 exact commits, 14 fallbacks | Baseline 0 / 3 / 2; portfolio 0 / 4 / 1. All scored cases failed the state check; cause unresolved. Full GLM-5.3 is not validated. |
| Qwen3 v1.0/current tau2 client | Same 10 retail IDs, baseline and D1+D2 under each version; 39 returned, 1 old-run timeout | All returned ACTION guardrail scores are zero. Static-user workflow checks do not establish full interactive tau2 quality or preservation. |

Each ID/configuration was run once. Condition variants can share an underlying
scenario. Calls and observations are not counts of independent tasks.
Task quality is scored separately from call matching; incomplete/error cases
remain unscored. Small shared-host samples do not establish speedups, quality
equivalence, latency CDF improvements or concurrency capacity.

## Remaining work

- Investigate scored failures and trajectory divergence, including GLM state
  mismatches and the Qwen paired regression.
- Evaluate task quality and latency with balanced repetitions before making
  performance or quality-preservation claims.
- Keep the new request-scoped D3 bridge and its further qualification in a
  separate update.

Before running GPU examples, check GPU ownership, available VRAM and shared
host memory, preserve model/runtime revisions and use fresh output directories.
Stop only services created for the run. Raw requests, datasets, weights and
logs remain outside Git.
