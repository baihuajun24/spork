# Optional DeepSeek-V4 continuation patch for vLLM 0.24.0

This extension maps `continue_final_message=True` to the native V4 encoder's
per-message `wo_eos` flag. It copies the conversation and installs once,
without editing the installed vLLM package. It checks for vLLM **0.24.0** and
is separate from the GLM runtime and the original vLLM 0.19.1 D3 integration.
The new request-scoped D3 bridge is deferred to a separate update.

Install vLLM 0.24.0 in a dedicated environment. After allocating GPUs and
checking available VRAM and shared host memory:

```bash
CUDA_VISIBLE_DEVICES="$GPU_IDS" python3 experiments/serving/vllm024/serve_v4.py \
  --model "$MODEL_PATH" --served-model-name deepseek-v4-flash \
  --host 127.0.0.1 --port 8000 --tensor-parallel-size "$TP" \
  --enable-auto-tool-choice --tool-call-parser deepseek_v4 \
  --reasoning-parser deepseek_v4 --safetensors-load-strategy lazy
```

Wait for `/health` and confirm the model name at `/v1/models` before running
the [BFCL example](../../bfcl/README.md) with `--model-family deepseek_v4`.
This server does not enable D3. See the [GPU reproduction instructions](../../../docs/validation/20261031-gpu/REPRODUCE.md)
for the tested runtime, checkpoint hashes and complete launch settings.

The [GPU report](../../../docs/validation/20261031-gpu/README.md#deepseek-v4-startup-failure)
retains the initial loader failure and successful retry. In the recorded
safetensors 0.8.0 runtime, `lazy` avoids the bytes loader's missing `F8_E8M0`
dtype mapping. It does not rewrite weights or change installed packages.
Model loading and tool-call matching do not establish task quality or speedups.
