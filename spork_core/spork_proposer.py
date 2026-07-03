"""SPORK D3 Proposer: context-aware draft injection at tool-call boundary.

Monkey-patches vLLM's ngram proposer. Delegates to the original ngram proposer
at all times EXCEPT the one step where the tool-call boundary is detected and
probe tokens are available — then injects probe-predicted tool-call tokens as
draft for verification. After firing once, delegates back to ngram.

Qwen3 uses a single <tool_call> token (151657). DeepSeek-V4 DSML uses a
multi-token string boundary, so SporkProposer also accepts a boundary token
sequence and searches for the last occurrence in the generated suffix.
"""
from __future__ import annotations

import numpy as np
import os
import threading
from typing import Any

TOOL_CALL_TOKEN_ID = 151657  # Qwen3 <tool_call> token (default)


def _common_prefix_len(a: list[int], b: list[int]) -> int:
    n = 0
    for x, y in zip(a, b):
        if x != y:
            break
        n += 1
    return n


class SporkProposer:

    def __init__(
        self,
        original_drafter=None,
        tool_call_token_id: int | None = None,
        boundary_token_ids: list[int] | None = None,
    ):
        self.k = getattr(original_drafter, 'k', 20)
        self.max_model_len = getattr(original_drafter, 'max_model_len', 4096)
        self.min_n = getattr(original_drafter, 'min_n', 1)
        self.max_n = getattr(original_drafter, 'max_n', 20)
        self._original = original_drafter
        self.valid_ngram_draft = np.zeros((1024, self.k), dtype=np.int32)
        self.valid_ngram_num_drafts = np.zeros(1024, dtype=np.int32)
        self.num_tokens_threshold = getattr(original_drafter, 'num_tokens_threshold', 8192)
        self.num_numba_thread_available = getattr(original_drafter, 'num_numba_thread_available', 1)
        self.tool_call_token_id = tool_call_token_id or TOOL_CALL_TOKEN_ID
        self.boundary_token_ids = list(boundary_token_ids or [])

        self._lock = threading.Lock()
        self._request_state: dict[Any, dict[str, Any]] = {}
        self.enabled = True
        # Max tokens the decode may be PAST the <tool_call> boundary and still
        # inject. ngram spec-dec advances multiple tokens/step, so a too-narrow
        # window is frequently leapt over (the boundary is detected but the args
        # are already being generated). A wide window + the skip/remaining logic
        # below lets D3 still inject whatever args are not yet generated.
        self.inject_window = int(os.environ.get("SPORK_D3_INJECT_WINDOW", "200"))
        self.inject_count = 0
        self.empty_count = 0
        self.draft_token_count = 0
        self.last_inject_info: dict[str, int] | None = None
        self.debug = os.environ.get("SPORK_PROPOSER_DEBUG", "").lower() in {"1", "true", "yes"}

    def set_probe_tokens(self, req_idx: int, token_ids: list[int], prompt_len: int = 0) -> None:
        self.set_request_probe_tokens(req_idx, token_ids, prompt_len)

    def set_request_probe_tokens(
        self,
        request_id: Any,
        token_ids: list[int],
        prompt_len: int = 0,
    ) -> None:
        """Register D3 draft tokens for one request.

        `request_id` is the vLLM batch index for the legacy monkey-patch path.
        A future explicit vLLM hook can pass stable request ids through the same
        method without sharing state across requests.
        """
        with self._lock:
            self._request_state[request_id] = {
                "probe_tokens": list(token_ids[:self.k]),
                "fired": False,
                "prompt_len": prompt_len,
            }

    def clear_request(self, request_id: Any) -> None:
        with self._lock:
            self._request_state.pop(request_id, None)
            self.last_inject_info = None

    @property
    def _probe_tokens(self) -> list[int] | None:
        state = self._request_state.get(0)
        return state.get("probe_tokens") if state else None

    @property
    def _fired(self) -> bool:
        return all(bool(state.get("fired")) for state in self._request_state.values()) if self._request_state else False

    def snapshot(self) -> dict[str, int]:
        with self._lock:
            return {
                "inject_count": self.inject_count,
                "empty_count": self.empty_count,
                "draft_token_count": self.draft_token_count,
                "active_requests": len(self._request_state),
            }

    def delta_since(self, snapshot: dict[str, int]) -> dict[str, int]:
        now = self.snapshot()
        return {k: now.get(k, 0) - snapshot.get(k, 0) for k in now}

    def clear_all(self) -> None:
        with self._lock:
            self._request_state.clear()
            self.last_inject_info = None

    def propose(
        self,
        sampled_token_ids: list[list[int]],
        *args: Any,
        **kwargs: Any,
    ) -> list[list[int]]:
        # Delegate to original ngram proposer when SPORK injection is inactive
        if not self.enabled or not self._request_state or self._fired:
            if self._original and hasattr(self._original, '_orig_propose'):
                return self._original._orig_propose(sampled_token_ids, *args, **kwargs)
            return [[] for _ in sampled_token_ids]

        if len(args) >= 4:
            _, num_tokens_no_spec, token_ids_cpu, _ = args[:4]
        elif len(args) >= 2:
            num_tokens_no_spec, token_ids_cpu = args[:2]
        else:
            raise TypeError("SporkProposer.propose requires vLLM ngram proposer inputs")

        # Run original ngram proposer first to get baseline draft proposals
        ngram_drafts = None
        if self._original and hasattr(self._original, '_orig_propose'):
            ngram_drafts = self._original._orig_propose(sampled_token_ids, *args, **kwargs)

        num_reqs = len(sampled_token_ids)
        draft_token_ids: list[list[int]] = []

        with self._lock:
            for i in range(num_reqs):
                state = self._request_state.get(i)
                if not state or state.get("fired"):
                    draft_token_ids.append(ngram_drafts[i] if ngram_drafts else [])
                    self.empty_count += 1
                    continue

                seq_len = int(num_tokens_no_spec[i])
                if seq_len <= 0:
                    draft_token_ids.append(ngram_drafts[i] if ngram_drafts else [])
                    self.empty_count += 1
                    continue

                gen_start = max(int(state.get("prompt_len", 0)), 0)
                if seq_len <= gen_start:
                    draft_token_ids.append(ngram_drafts[i] if ngram_drafts else [])
                    self.empty_count += 1
                    continue

                gen_tokens = token_ids_cpu[i, gen_start:seq_len]
                if self.boundary_token_ids:
                    boundary = self.boundary_token_ids
                    boundary_len = len(boundary)
                    gen_list = gen_tokens.tolist()
                    tc_positions = [
                        pos for pos in range(0, len(gen_list) - boundary_len + 1)
                        if gen_list[pos:pos + boundary_len] == boundary
                    ]
                else:
                    tc_positions = np.where(gen_tokens == self.tool_call_token_id)[0]
                    boundary_len = 1

                if self.debug:
                    self._dbg = getattr(self, '_dbg', 0) + 1
                    last5 = gen_tokens[-5:].tolist() if len(gen_tokens) >= 5 else gen_tokens.tolist()
                    print(f"  [PROPOSER] call={self._dbg} seq={seq_len} gen_start={gen_start} gen_len={len(gen_tokens)} tc_found={len(tc_positions)} last5={last5}", flush=True)

                if len(tc_positions) == 0:
                    # No <tool_call> yet — use ngram draft
                    draft_token_ids.append(ngram_drafts[i] if ngram_drafts else [])
                    self.empty_count += 1
                    continue

                tc_pos_in_gen = int(tc_positions[-1])
                gen_len = seq_len - gen_start
                tokens_after_tc = gen_len - tc_pos_in_gen - boundary_len + 1

                if tokens_after_tc <= self.inject_window:
                    draft = state["probe_tokens"]
                    generated_body = gen_tokens[tc_pos_in_gen + boundary_len:].tolist()
                    skip = _common_prefix_len(generated_body, draft)
                    remaining = draft[skip:]
                    if remaining:
                        proposed = remaining[:self.k]
                        draft_token_ids.append(proposed)
                        state["fired"] = True
                        self.inject_count += 1
                        self.draft_token_count += len(proposed)
                        self.last_inject_info = {
                            "request_id": i,
                            "seq_len": seq_len,
                            "gen_len": gen_len,
                            "tokens_after_tc": tokens_after_tc,
                            "skip": skip,
                            "generated_body_tokens": len(generated_body),
                            "draft_tokens": len(proposed),
                        }
                    else:
                        draft_token_ids.append(ngram_drafts[i] if ngram_drafts else [])
                        self.empty_count += 1
                else:
                    draft_token_ids.append(ngram_drafts[i] if ngram_drafts else [])
                    state["fired"] = True
                    self.empty_count += 1

        return draft_token_ids

    def load_model(self, *args, **kwargs):
        pass


def patch_drafter(gpu_model_runner, proposer: SporkProposer) -> None:
    original = gpu_model_runner.drafter
    proposer._original = original
    original._spork = proposer
    original._orig_propose = original.propose
    original.propose = proposer.propose
