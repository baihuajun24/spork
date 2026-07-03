#!/bin/bash
# SPORK-HTTP serve for a Qwen3 model — adds ngram speculative decoding + the
# /spork/* draft-injection sidecar, which enables the **d1_d2_d3** config (D3 =
# inject the probe's predicted tool-call tokens at the boundary to accelerate the
# main decode). For `baseline` / `d1` / `d1_d2` use the plain `launch_vllm_qwen.sh`
# instead (no spec-dec / no sidecar).
#
# Qwen's tool-call boundary is the `<tool_call>` token (id 151657), which the
# SporkProposer uses by default — no SPORK_BOUNDARY_TOKEN_IDS env needed. (For a
# model whose tool call opens with a different token sequence, set
# SPORK_BOUNDARY_TOKEN_IDS='[...]' to that sequence.)
set -e
MODEL_PATH=${MODEL_PATH:-Qwen/Qwen3-32B}
PORT=${PORT:-8000}
TP=${TP:-1}
MAX_LEN=${MAX_LEN:-40960}
GPU_UTIL=${GPU_UTIL:-0.85}
NSPEC=${NSPEC:-20}              # ngram num_speculative_tokens / prompt_lookup window
export NO_PROXY=localhost,127.0.0.1 no_proxy=localhost,127.0.0.1
export VLLM_WORKER_MULTIPROC_METHOD=fork
exec python3 launch_vllm_spork_http.py \
  --model "$MODEL_PATH" --served-model-name qwen3-32b \
  --tensor-parallel-size "$TP" --dtype bfloat16 --max-model-len "$MAX_LEN" \
  --gpu-memory-utilization "$GPU_UTIL" --trust-remote-code --max-logprobs 20 \
  --speculative-config "{\"method\":\"ngram\",\"num_speculative_tokens\":${NSPEC},\"prompt_lookup_max\":${NSPEC},\"prompt_lookup_min\":2}" \
  --port "$PORT"
