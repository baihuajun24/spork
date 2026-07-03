"""Reusable SPORK runtime pieces for benchmark runners."""


def extract_answer(text: str) -> str:
    """Extract the final answer from model output, stripping <think> CoT.

    If the text starts with <think> but never closes it, the model ran out of
    tokens during reasoning — there is no final answer, so return empty string.
    """
    if "</think>" in text:
        return text.split("</think>")[-1].strip()
    if text.lstrip().startswith("<think>"):
        return ""
    return text.strip()
