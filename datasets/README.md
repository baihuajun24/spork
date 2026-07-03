# Datasets

- `hotpotqa_validation.jsonl` — a 500-question sample of the **HotpotQA**
  validation set (Yang et al., 2018), distractor setting. HotpotQA is released
  under **CC BY-SA 4.0**. Full set: https://hotpotqa.github.io/ .
  Records: `{"question": ..., "answer": ..., "type": ...}`.
- `gaia_validation.jsonl` — **not included**. GAIA (Mialon et al., 2023) is on
  Hugging Face (`gaia-benchmark/GAIA`, gated). Provide your own JSONL with
  `Question` / `Final answer` fields, and a web search/browse backend
  (see `../tool_backends.py`).
