#!/usr/bin/env python3
"""Run vLLM's OpenAI server with the V4 continuation compatibility patch."""
from __future__ import annotations

import runpy

from vllm_v4_continue_patch import install_v4_continue_final_message_patch


if __name__ == "__main__":
    install_v4_continue_final_message_patch()
    runpy.run_module("vllm.entrypoints.openai.api_server", run_name="__main__")
