"""Parse tau2 domain tools.py files to extract OpenAI-format tool schemas.

Uses AST on the source file to avoid importing tau2 (heavy deps). Returns a
list of {"type": "function", "function": {"name", "description", "parameters"}}
dicts that match what tokenizer.apply_chat_template(tools=...) expects.

We extract per-method:
- name
- docstring's first paragraph as description
- parameters (excluding self): JSON-schema type from annotation; required = no default;
  per-parameter description from docstring "Args:" block.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path
from typing import Any


# ---- type annotation → JSON schema fragment ----

def _ann_to_schema(node: ast.AST | None) -> dict[str, Any]:
    if node is None:
        return {"type": "string"}
    # str / int / float / bool / list / dict
    if isinstance(node, ast.Name):
        return {
            "str": {"type": "string"},
            "int": {"type": "integer"},
            "float": {"type": "number"},
            "bool": {"type": "boolean"},
            "list": {"type": "array", "items": {}},
            "dict": {"type": "object"},
            "Any": {},
        }.get(node.id, {"type": "string"})
    # Optional[T] / List[T] / List[dict | T] / Union[...]
    if isinstance(node, ast.Subscript):
        base = node.value.id if isinstance(node.value, ast.Name) else None
        inner = node.slice
        # tuple of args (e.g. dict[str, int])
        if base in ("List", "list"):
            items = _ann_to_schema(inner)
            return {"type": "array", "items": items}
        if base in ("Dict", "dict"):
            return {"type": "object"}
        if base in ("Optional",):
            return _ann_to_schema(inner)
        if base in ("Union",):
            # If only one non-None, use it
            elts = inner.elts if isinstance(inner, ast.Tuple) else [inner]
            for e in elts:
                if isinstance(e, ast.Constant) and e.value is None:
                    continue
                return _ann_to_schema(e)
            return {"type": "string"}
        return {"type": "string"}
    # X | Y (PEP 604)
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.BitOr):
        for side in (node.left, node.right):
            if isinstance(side, ast.Constant) and side.value is None:
                continue
            return _ann_to_schema(side)
        return {"type": "string"}
    # unknown — string fallback
    return {"type": "string"}


# ---- docstring parameter doc parsing ----

def _parse_args_block(doc: str) -> dict[str, str]:
    """Extract {param_name: description} from a Google-style Args: block."""
    if not doc:
        return {}
    # find "Args:" then lines like "    name: description" until next section or end
    m = re.search(r"(?ms)^\s*Args\s*:\s*\n(.*?)(\n\s*(Returns?|Raises?|Yields?|Note|Example)s?\s*:|\Z)", doc)
    if not m:
        return {}
    block = m.group(1)
    out = {}
    cur_name: str | None = None
    cur_parts: list[str] = []
    for raw in block.splitlines():
        line = raw.rstrip()
        if not line.strip():
            continue
        m2 = re.match(r"^\s{2,}(\w+)\s*(?:\([^)]*\))?\s*:\s*(.*)$", line)
        if m2:
            if cur_name:
                out[cur_name] = " ".join(cur_parts).strip()
            cur_name = m2.group(1)
            cur_parts = [m2.group(2).strip()]
        elif cur_name and line.startswith((" ", "\t")):
            cur_parts.append(line.strip())
    if cur_name:
        out[cur_name] = " ".join(cur_parts).strip()
    return out


def _short_desc(doc: str) -> str:
    if not doc:
        return ""
    # take everything before first "Args:" / blank+"Args:"
    doc = doc.strip()
    m = re.split(r"\n\s*(?:Args?|Returns?|Raises?)\s*:", doc, maxsplit=1)
    head = m[0].strip()
    return head


# ---- main extraction ----

def parse_tools_py(path: Path) -> list[dict]:
    tree = ast.parse(path.read_text())
    tools: list[dict] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.ClassDef):
            continue
        for item in node.body:
            if not isinstance(item, ast.FunctionDef):
                continue
            # must have @is_tool decorator (active, not commented)
            has_tool_dec = False
            for dec in item.decorator_list:
                if isinstance(dec, ast.Call) and isinstance(dec.func, ast.Name) and dec.func.id == "is_tool":
                    has_tool_dec = True
                    break
                if isinstance(dec, ast.Name) and dec.id == "is_tool":
                    has_tool_dec = True
                    break
            if not has_tool_dec:
                continue

            # gather args
            args = item.args.args[1:]  # drop self
            defaults = item.args.defaults
            n_defaults = len(defaults)
            n_args = len(args)
            # defaults line up with the last n_defaults args
            has_default = [False] * n_args
            for i in range(n_defaults):
                has_default[n_args - n_defaults + i] = True

            doc = ast.get_docstring(item) or ""
            desc = _short_desc(doc)
            arg_docs = _parse_args_block(doc)

            props: dict[str, dict] = {}
            required: list[str] = []
            for i, a in enumerate(args):
                schema = _ann_to_schema(a.annotation)
                if a.arg in arg_docs:
                    schema = {**schema, "description": arg_docs[a.arg]}
                props[a.arg] = schema
                if not has_default[i]:
                    required.append(a.arg)

            tools.append({
                "type": "function",
                "function": {
                    "name": item.name,
                    "description": desc,
                    "parameters": {
                        "type": "object",
                        "properties": props,
                        "required": required,
                    },
                },
            })
    return tools


if __name__ == "__main__":
    import os, sys
    root = os.environ.get("SPORK_TAU2_ROOT", sys.argv[1] if len(sys.argv) > 1 else ".")
    for d in ["airline", "retail"]:
        p = Path(root) / f"src/tau2/domains/{d}/tools.py"
        tools = parse_tools_py(p)
        print(f"{d}: {len(tools)} tools -> {[t['function']['name'] for t in tools]}")
