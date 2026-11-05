# GPU smoke validation — 2026-10-31

This directory records the 2026-10-31 functional checks of model adapters and
candidate portfolios.
It does **not** establish latency gains, quality equivalence or full-model
support. The original paper's results and figures are unchanged.

## Provenance and reproduction

- [Commands and measurement boundaries](REPRODUCE.md).
- [Extraction and runtime-source equivalence](code-equivalence.json): historical
  commit IDs are preserved; the included runtime files match those used in
  the checks. The new D3 implementation and its dedicated records are excluded.
- [Per-service and per-client receipts](receipts.json): code versions, launch
  arguments, runtimes, timestamps, allocation and host-memory budgets.
- [Checkpoint fingerprints](model-fingerprints.json): config/tokenizer and all
  weight shards. Upstream revisions were unavailable and remain `null`;
  hashes identify the actual local checkpoints without inventing revisions.
- Runtime/source fingerprints: [vLLM 0.24](vllm024-runtime.json),
  [native GLM](glm-native-runtime.json), [BFCL client](bfcl-client-runtime.json),
  [tau2 client](tau-client-runtime.json), [tau2 data/tools](tau2-source-hashes.json).
- [Artifact checks and GPU release state](artifact-checks.json): all 80
  per-task raw hashes matched, both export CLIs reproduced their compact
  records, and all 24 cooperative physical-GPU locks were available.
- [V4 retry checks](v4-retry-artifact-checks.json): ten additional raw hashes
  and the compact export matched; its two GPUs and six associated locks
  were released. The initial audit remains a separate historical record.

The BFCL task list was fixed before generation; Qwen3 uses all ten IDs and the
other families the first five. IDs include condition variants of shared
scenarios. Each task/configuration was run once. The tau2 regression uses the
same ten retail task IDs under both code versions and configurations.
Observations, repeats and independent tasks must not be conflated.

Raw requests, generated text, simulator outputs and server logs remain local.
The compact observations retain each raw result's SHA256 for author-side
verification. Reproduction means rerunning the recorded commands with the
recorded dependencies; exact regeneration is not guaranteed by seed alone.

## BFCL observations

The independent `baseline` does not launch SPORK probes or candidate workers.
`portfolio` uses the same task/model/generation conditions with T1+T16 probes.
Each configuration starts from an independent simulator world. Quality is
scored separately by the unmodified official multi-turn **and** irrelevance
checkers; a matched speculative call is not a task-quality score.

<!-- BFCL_TABLE_START -->

| Run | Config | Recorded / planned | Quality pass / fail / unscored | Commits | Fallbacks | Decision violations |
|---|---|---|---|---|---|---|
| glm-01-bfcl | baseline | 5 / 5 | 0 / 3 / 2 | 0 | n/a | 0 |
| glm-01-bfcl | portfolio | 5 / 5 | 0 / 4 / 1 | 39 | 14 | 0 |
| qwen-01-bfcl | baseline | 10 / 10 | 3 / 6 / 1 | 0 | n/a | 0 |
| qwen-01-bfcl | portfolio | 10 / 10 | 4 / 6 / 0 | 88 | 12 | 0 |
| qwen35-01-bfcl | baseline | 5 / 5 | 2 / 3 / 0 | 0 | n/a | 0 |
| qwen35-01-bfcl | portfolio | 5 / 5 | 3 / 2 / 0 | 42 | 1 | 0 |
| v4-02-bfcl | baseline | 5 / 5 | 2 / 2 / 1 | 0 | n/a | 0 |
| v4-02-bfcl | portfolio | 5 / 5 | 2 / 2 / 1 | 46 | 0 | 0 |

<!-- BFCL_TABLE_END -->

The machine-readable [counts](bfcl-counts.json) and per-task exports preserve
successes, scored failures and unscored/incomplete observations. A missing
record is `missing`, with null quality, rather than a measured zero.

For Qwen3, the missing-function baseline task `multi_turn_miss_func_65` reached
the step limit and remains unscored. Both configurations also have scored
failures in missing-function/parameter conditions. At least one completion
reached its token cap; compact rows flag this as an **inferred** budget hit,
not a recorded server finish reason. Main generations and trajectories differ
between some baseline/portfolio pairs despite temperature 0 and seed 1234.
These observations do not support a deterministic-trajectory or quality
equivalence claim.

The Qwen3 pair `multi_turn_miss_param_65` is a [quality regression](paired-quality-regression.json): baseline
passes, portfolio fails both state and relevance checks. Its first canonical
decision already differs before any speculative result has been committed;
the portfolio also has an inferred token-budget hit. The cause of the
generation divergence is unresolved. The aggregate pass counts must not hide
this per-task regression.

Decision audits compare the recorded selected call with the canonical main
call using the full name and typed JSON arguments. They also check that
baseline records contain no speculative work. This is evidence about recorded
decisions; it does not independently inspect hidden server activity or prove
that every discarded simulator state is isolated.

GLM returned all ten observations without a client/runtime error. Its main
responses contained 104 structured tool calls; all 108 recorded probes parsed
from reasoning text, with main/probe usage present. The portfolio recorded
39 exact commits and 14 canonical fallbacks. However, every scored observation
failed the official state check, and three observations hit the step limit
and remain unscored. Native transport and probe parsing worked in this sample;
task quality did not pass. The cause of the state mismatches is unresolved.

### DeepSeek V4 startup failure

The first V4 attempt (`v4-01`) exited before readiness while loading checkpoint weights.
`safetensors.torch.load` raised `KeyError: 'F8_E8M0'` in both TP workers in the
recorded vLLM 0.24 / safetensors 0.8.0 runtime. No BFCL task or probe ran in
that attempt; its task-quality result remains null.
[Loader diagnosis](v4-startup-failure.json) records the failing loader stack,
its raw-log hash, and a CPU-only synthetic dtype check. The same synthetic
tensor loads through `safe_open`. The
[upstream 0.8.0 Python loader](https://github.com/huggingface/safetensors/blob/v0.8.0/bindings/python/py_src/safetensors/torch.py)
also lacks `F8_E8M0` in the bytes-loader dtype map. This is a loader
compatibility failure; it is not evidence of corrupted checkpoint files.

The follow-up (`v4-02`, code `3028801`) changes only the weight-loading strategy
from `eager` to `lazy`, which uses `safe_open`. A real checkpoint scale tensor
read as `torch.float8_e8m0fnu` with bytes identical to the file. All 48 shards
then loaded on TP2 GPUs 0–1; worker 0 reported 47.99 seconds of weight reading
and 73.88 GiB of model memory. The server passed readiness after kernel warmup.
The original checkpoint and installed packages are unchanged. This is a
file-loading strategy change; vLLM execution still uses `--enforce-eager`.

The [retry receipt](v4-loader-retry.json) records the actual checkpoint read,
unchanged runtime/source checks and GPU loading. The
[ten BFCL observations](v4-02-bfcl.json) returned without runtime errors.
Baseline and portfolio each have two quality passes, two scored failures and
one incomplete/unscored case (`multi_turn_base_65`, step limit). The 90
recorded native main tool calls and 108 successfully parsed probes have
nonempty usage records. All 46 speculative commits match the canonical full
call, with zero recorded fallbacks or decision violations. Probe output was
parsed from reasoning text; this does not establish full-model support.

The V4 missing-function case fails relevance in both configurations while
passing state checks. Its observed task wall times are 291.0 seconds for
baseline and 486.9 seconds for portfolio; this slower portfolio observation is
retained. These single, unbalanced runs are not a speedup estimate. Loader
readiness and matching speculative calls do not establish task quality or
latency gains.

## tau2 client workflow regression

| Runner | Config | Returned / planned | Timeout / missing | ACTION reward = 1 / eligible | Recorded probes / commits |
|---|---|---|---|---|---|
| v1.0 `31d5ab6` | baseline | 10 / 10 | 0 / 0 | 0 / 10 | 0 / 0 |
| v1.0 `31d5ab6` | D1+D2 | 9 / 10 | 1 / 0 | 0 / 9 | 67 / 29 |
| v1.1 `d6a3ef6` | baseline | 10 / 10 | 0 / 0 | 0 / 10 | 0 / 0 |
| v1.1 `d6a3ef6` | D1+D2 | 10 / 10 | 0 / 0 | 0 / 10 | 68 / 32 |

The [v1.0](v1.0-tau2.json) and [v1.1](v1.1-tau2.json) exports contain
40 observations of the same ten tasks. All 39 returned observations received
ACTION reward 0. The v1.0 D1+D2 run of retail task 39 timed out at 240 seconds;
the legacy runner discarded its partial trace. Its quality, probe and commit
counts are therefore null and excluded above. Both baselines contain no
recorded probes or speculative dispatches. v1.1 completed the workflow, but
these zero guardrail scores do not establish task-quality preservation or a
successful end-to-end tau2 regression. Investigating the scores is a TODO.

The tau2 runner uses static user instructions and an ACTION-match guardrail,
not the full interactive tau2 protocol. Errors/timeouts are unscored in these
exports even if the legacy summary reports an action reward. This check
compares client code against vLLM 0.24, not the legacy 0.19.1 D3 backend.
The pinned benchmark checkout includes newer τ³-bench task fixes; its
equivalence to the original paper's dataset revision is unverified.

## Limits and cost

All servers use eager execution and disabled prefix caching. These are small
functional samples on a shared host. Early Qwen requests overlapped with
checkpoint fingerprinting, and some services overlap on separate GPUs.
Recorded task wall times are descriptive observations, not fair speedup
estimates, CDF evidence or concurrency-capacity measurements.

| Service | Allocated GPUs | Allocation hours (GPU·h) | Outcome |
|---|---|---|---|
| Qwen3-32B | 4 | 9.639 | BFCL and both tau2 runner versions ended |
| Qwen3.5-35B-A3B | 2 | 1.073 | BFCL ended |
| DeepSeek V4 initial attempt | 2 | 0.022 | Startup failed; no client requests |
| DeepSeek V4 lazy-loader retry | 2 | 1.324 | BFCL ended: 8 complete, 2 incomplete/unscored |
| GLM-5.3-Flash | 4 | 5.683 | BFCL ended; no scored quality passes |
| Included total | — | **17.742** | All five included service allocations released |

The [admission log summary](resource-admission.json) records the initial V4 attempt waiting for
the shared host-memory budget until the primary stopped. Those CPU-only
checks held no GPUs and are not extra task trials. The allocation total covers
the five included services. The unmodified source receipt also contains a
separate D3 allocation (0.315 GPU·h), excluded from this report and PR. CPU
metadata/hash work is outside this service-allocation metric. At final inspection all eight
GPUs had no compute processes and 4 MiB reported memory each.

Allocation hours include model loading, CPU scoring and intervals when GPUs
were reserved but not busy. Model weights, datasets, raw prompts and logs are
not included in this directory. No publication or release is implied by these
local checks.
