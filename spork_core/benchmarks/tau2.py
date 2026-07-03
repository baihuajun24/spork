"""tau2-bench loading and execution helpers."""
from __future__ import annotations

import json
from pathlib import Path

from spork_core import paths


paths.add_runtime_paths()


def load_system_prompt(domain: str) -> str:
    policy = (paths.TAU2_ROOT / f"data/tau2/domains/{domain}/policy.md").read_text()
    return (
        "You are a customer service agent. Follow the policy below exactly. "
        "Use the available tools to help the user. Make one tool call at a time. "
        "When the task is complete, reply to the user with a final message summarizing "
        "the outcome (do not call any more tools).\n\n"
        f"<policy>\n{policy}\n</policy>"
    )


def load_tools(domain: str) -> list[dict]:
    from tau2_tool_parser import parse_tools_py

    return parse_tools_py(paths.TAU2_ROOT / f"src/tau2/domains/{domain}/tools.py")


def load_tasks(domain: str) -> list[dict]:
    tasks_path = paths.TAU2_ROOT / f"data/tau2/domains/{domain}/tasks.json"
    tasks = json.loads(tasks_path.read_text())
    return [t for t in tasks if (t.get("evaluation_criteria") or {}).get("actions")]


def build_user_message(task: dict) -> str:
    ins = task["user_scenario"]["instructions"]
    parts = []
    for key in ("known_info", "reason_for_call", "task_instructions"):
        if ins.get(key):
            parts.append(ins[key].strip())
    return "\n\n".join(parts)


def get_environment(domain: str):
    if domain == "airline":
        from tau2.domains.airline.environment import get_environment as get_env
    elif domain == "retail":
        from tau2.domains.retail.environment import get_environment as get_env
    else:
        raise ValueError(f"unknown tau2 domain: {domain}")
    return get_env()


def write_json(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False, default=str))

