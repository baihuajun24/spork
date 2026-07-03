"""SPORK acceptance and confidence gates."""
from __future__ import annotations

from dataclasses import dataclass

from . import tool_calls


@dataclass(frozen=True)
class GateDecision:
    dispatch_spec: bool
    accept_after_main: bool
    name_match: bool
    args_exact: bool
    args_partial: float
    confidence: float | None


def decide_gate(
    gate: str,
    probe_call: dict | None,
    main_call: dict | None,
    probe_logprobs: dict,
    *,
    confidence_threshold: float = 0.90,
) -> GateDecision:
    name_match = bool(
        probe_call and main_call and probe_call.get("name") == main_call.get("name")
    )
    args_exact = bool(
        name_match
        and tool_calls.args_exact_match(probe_call.get("arguments"), main_call.get("arguments"))
    )
    args_partial = (
        tool_calls.args_partial_overlap(probe_call.get("arguments"), main_call.get("arguments"))
        if name_match and probe_call and main_call
        else 0.0
    )
    confidence = tool_calls.span_min_prob(probe_logprobs, skip_first=True)

    if gate == "name_only_loose":
        return GateDecision(True, name_match, name_match, args_exact, args_partial, confidence)
    if gate == "args_exact_strict":
        return GateDecision(True, args_exact, name_match, args_exact, args_partial, confidence)
    if gate == "confidence_strict":
        dispatch = confidence is not None and confidence >= confidence_threshold
        return GateDecision(dispatch, dispatch and args_exact, name_match, args_exact,
                            args_partial, confidence)
    if gate == "per_tool_loose_search":
        # Tencent M6 precedent: name-only accept for web_search (free-form query args rarely
        # match exactly), strict args_exact for all other tools (URL/code args should match).
        # Decision keyed on MAIN's tool name (probe must match main by name first).
        if name_match and main_call and main_call.get("name") == "search":
            accept = True
        else:
            accept = args_exact
        return GateDecision(True, accept, name_match, args_exact, args_partial, confidence)
    if gate == "confidence_name_loose":
        # D1+D2 for GAIA: confidence-gated dispatch + name-only acceptance.
        dispatch = confidence is not None and confidence >= confidence_threshold
        return GateDecision(dispatch, dispatch and name_match, name_match, args_exact,
                            args_partial, confidence)
    raise ValueError(f"unknown gate: {gate}")

