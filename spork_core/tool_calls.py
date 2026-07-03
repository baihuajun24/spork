"""Model-agnostic helpers for structured tool-call comparison and confidence."""
from __future__ import annotations

import json
import math
from typing import Any


EMPTY_LOGPROBS = {"tokens": [], "token_logprobs": [], "top_logprobs": []}


def normalize_args(args: Any) -> dict:
    if args is None:
        return {}
    if isinstance(args, dict):
        return args
    if isinstance(args, str):
        try:
            obj = json.loads(args)
            return obj if isinstance(obj, dict) else {"value": obj}
        except json.JSONDecodeError:
            return {"value": args}
    return {"value": args}


def args_exact_match(a: dict | None, b: dict | None) -> bool:
    try:
        return json.dumps(a or {}, sort_keys=True, default=str) == json.dumps(
            b or {}, sort_keys=True, default=str
        )
    except Exception:
        return False


def args_partial_overlap(a: dict | None, b: dict | None) -> float:
    if not a or not b:
        return 0.0
    keys = set(a) | set(b)
    if not keys:
        return 1.0
    return sum(1 for k in keys if k in a and k in b and a[k] == b[k]) / len(keys)


def extract_full_logprobs(logprobs_obj: dict | None, max_positions: int = 21) -> dict:
    if not logprobs_obj:
        return dict(EMPTY_LOGPROBS)
    return {
        "tokens": list(logprobs_obj.get("tokens") or [])[:max_positions],
        "token_logprobs": list(logprobs_obj.get("token_logprobs") or [])[:max_positions],
        "top_logprobs": list(logprobs_obj.get("top_logprobs") or [])[:max_positions],
    }


def first_token_top1_prob(logprobs: dict) -> float | None:
    top = logprobs.get("top_logprobs") or []
    if not top or not top[0]:
        return None
    return math.exp(max(top[0].values()))


def span_min_prob(logprobs: dict, *, skip_first: bool = True) -> float | None:
    token_lps = list(logprobs.get("token_logprobs") or [])
    if skip_first:
        token_lps = token_lps[1:]
    token_lps = [lp for lp in token_lps if lp is not None]
    if not token_lps:
        return None
    return math.exp(min(token_lps))


def span_scores(logprobs: dict, *, span_len: int = 5, skip_first: bool = True) -> dict:
    """Multi-metric aggregation over the first `span_len` tokens of a probe decode.

    Returns a dict with four scalars in [0, 1], all missing → None:
      - mean_top1: arithmetic mean of top-1 probs
      - geo_mean:  exp(mean of log top-1 probs) = geometric mean of top-1 probs
      - min_top1:  min of top-1 probs over the span (most pessimistic)
      - mean_top1_margin: mean of (top1 - top2) probability margins; null when top-2 absent

    Used by Adaptive's confidence gate post-hoc selection (logged per turn in replay).
    """
    tokens = list(logprobs.get("tokens") or [])
    token_lps = list(logprobs.get("token_logprobs") or [])
    top_lps = list(logprobs.get("top_logprobs") or [])
    if skip_first:
        tokens = tokens[1:]; token_lps = token_lps[1:]; top_lps = top_lps[1:]
    tokens = tokens[:span_len]; token_lps = token_lps[:span_len]; top_lps = top_lps[:span_len]

    valid_lps = [lp for lp in token_lps if lp is not None]
    result: dict[str, float | None] = {
        "mean_top1": None, "geo_mean": None, "min_top1": None, "mean_top1_margin": None,
        "span_len_used": len(valid_lps),
    }
    if valid_lps:
        probs = [math.exp(lp) for lp in valid_lps]
        result["mean_top1"] = sum(probs) / len(probs)
        result["geo_mean"] = math.exp(sum(valid_lps) / len(valid_lps))
        result["min_top1"] = min(probs)
    margins = []
    for pos_top in top_lps:
        if not pos_top:
            continue
        sorted_lps = sorted(pos_top.values(), reverse=True)
        if len(sorted_lps) >= 2:
            margins.append(math.exp(sorted_lps[0]) - math.exp(sorted_lps[1]))
    if margins:
        result["mean_top1_margin"] = sum(margins) / len(margins)
    return result


def preview(text: str | None, limit: int = 200) -> str:
    if not text:
        return ""
    if len(text) <= limit:
        return text
    return text[:limit] + "...[truncated]"
