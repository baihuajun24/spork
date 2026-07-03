#!/bin/bash
# Launch a Qwen3 model with vLLM's OpenAI-compatible server for SPORK.
#
# SPORK needs /v1/completions with logprobs (for the speculative tool-call
# probe) and think-mode CoT (Qwen3 emits <think>...</think>), so we serve the
# base Qwen3 chat model and drive thinking from the request side.
#
# Configure via environment variables:
#   MODEL_PATH  path or HF id of the Qwen3 model   (default: Qwen/Qwen3-32B)
#   PORT        server port                        (default: 8000)
#   TP          tensor-parallel size / #GPUs       (default: 1)
#   MAX_LEN     max model length                   (default: 40960)
#   GPU_UTIL    gpu memory utilization             (default: 0.85)

set -euo pipefail

MODEL_PATH="${MODEL_PATH:-Qwen/Qwen3-32B}"
PORT="${PORT:-8000}"
TP="${TP:-1}"
MAX_LEN="${MAX_LEN:-40960}"
GPU_UTIL="${GPU_UTIL:-0.85}"

# vLLM serves localhost; keep loopback out of any HTTP proxy.
export NO_PROXY="localhost,127.0.0.1${NO_PROXY:+,$NO_PROXY}"
export no_proxy="$NO_PROXY"

python3 -m vllm.entrypoints.openai.api_server \
    --model "$MODEL_PATH" \
    --tensor-parallel-size "$TP" \
    --dtype bfloat16 \
    --max-model-len "$MAX_LEN" \
    --gpu-memory-utilization "$GPU_UTIL" \
    --trust-remote-code \
    --max-logprobs 20 \
    --port "$PORT"
