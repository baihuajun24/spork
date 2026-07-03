#!/usr/bin/env python3
"""Unified SPORK multi-benchmark evaluation runner (HTTP mode).

Runs baseline and SPORK (D1, D1+D2, D1+D2+D3) configs against an external vLLM
server. Reuses spork_core/ infrastructure (turn_runner, QwenToolCallAdapter,
executors).

Supported benchmarks: tau2 (out of the box, via tau2-bench), plus gaia and
hotpotqa. The GAIA/HotpotQA web-search and web-browse tools require a
user-supplied backend (see tool_backends.py); the SPORK paper used internal
services that are not part of this public release.

Usage:
    python3 spork_unified_eval.py \
        --model-url http://localhost:8001/v1 --model-name qwen3-4b \
        --benchmark tau2 --configs baseline,d1 --seed 42

    python3 spork_unified_eval.py \
        --model-url http://localhost:8001/v1 --model-name qwen3-4b \
        --benchmark gaia --configs baseline,d1 --n 165 --seed 42 \
        --gaia-dataset-path ./datasets/gaia_validation.jsonl

    python3 spork_unified_eval.py \
        --model-url http://localhost:8001/v1 --model-name qwen3-4b \
        --benchmark hotpotqa --configs baseline,d1 --n 200 --seed 42 \
        --hotpotqa-dataset-path ./datasets/hotpotqa_validation.jsonl
"""
import argparse
import asyncio
import json
import os
import random
import re
import statistics
import string
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import aiohttp

from spork_core import extract_answer, paths
from spork_core.executors import (
    CachedToolExecutor,
    LatencyFloorToolExecutor,
    Tau2ToolExecutor,
)
from spork_core.qwen import QwenToolCallAdapter, Qwen35ToolCallAdapter
from spork_core.turn_runner import TurnConfig, RetryTurnConfig, run_baseline_turn, run_spork_turn, run_spork_retry_turn, run_spork_cross_turn
from spork_core.vllm_client import VllmClient
from tool_backends import WebSearchBackend, WebBrowseBackend, WikipediaSearchBackend

paths.add_runtime_paths()

# Results root: env override, else a local ./results dir next to this script.
RESULTS_ROOT = Path(os.environ.get("SPORK_RESULTS_ROOT", str(Path(__file__).parent / "results")))


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def write_json(path: Path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False, default=str))


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw in (None, ""):
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def _result_has_error(result: dict) -> bool:
    return any(t.get("error") for t in result.get("turns", []))


def _p95_nearest_rank(values: list[float]) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    idx = min(len(ordered) - 1, int(len(ordered) * 0.95))
    return ordered[idx]


async def run_discarded_warmup(
    args,
    client: VllmClient,
    benchmark: str,
    tasks: list[dict],
    system_prompt: str | None,
    tools: list[dict] | None,
) -> None:
    if getattr(args, "warmup_requests", 0) <= 0:
        return

    warm_messages = [{"role": "user", "content": "Say OK."}]
    if benchmark == "tau2" and tasks:
        from spork_core.benchmarks.tau2 import load_system_prompt, load_tools
        domain = tasks[0]["domain"]
        warm_messages = [
            {"role": "system", "content": load_system_prompt(domain)},
            {"role": "user", "content": tasks[0]["user_message"]},
        ]
    elif benchmark == "gaia" and tasks:
        warm_messages = [{"role": "user", "content": tasks[0].get("Question", "Say OK.")}]
        if system_prompt:
            warm_messages.insert(0, {"role": "system", "content": system_prompt})
    elif benchmark == "hotpotqa" and tasks:
        warm_messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": tasks[0]["question"]},
        ]

    short_max = max(1, min(32, args.completion_max_tokens, args.max_tokens))
    long_max = max(1, min(args.warmup_max_tokens, args.completion_max_tokens, args.max_tokens))
    log(f"Running discarded warmup: {args.warmup_requests} round(s), max_tokens={long_max}")
    async with aiohttp.ClientSession(trust_env=False) as session:
        for _ in range(args.warmup_requests):
            await client.complete(
                session,
                warm_messages[-1]["content"],
                max_tokens=short_max,
                logprobs=None,
            )


class _SporkSidecar:
    """D3-in-HTTP sidecar client.

    Talks to the SPORK-enabled vLLM HTTP server's /spork/* endpoints (registered
    by launch_vllm_spork_http.py). Given a probe's predicted tool call, it
    tokenizes the canonical body and POSTs the draft tokens; the in-engine
    SporkProposer injects them at the <tool_call> boundary of the main decode.

    Mirrors engine-mode set_probe_tokens: body = "\\n" + JSON tool-call body,
    prompt_len = main_prompt token length (so the proposer searches generated
    tokens only). Acceptance is decided downstream by the turn runner; this only
    accelerates decode. State is per-server (request_id index 0), so the eval
    must run a single concurrent request against this server (workers=1).
    """

    def __init__(
        self,
        session,
        base_url: str,
        tokenizer,
        adapter,
        *,
        model: str | None = None,
        tool_choice: str | None = None,
    ):
        self.session = session
        root = base_url.rstrip("/")
        if root.endswith("/v1"):
            root = root[: -len("/v1")]
        self.root = root
        self.tokenizer = tokenizer
        self.adapter = adapter
        self.model = model
        self.tool_choice = tool_choice
        self.xml_mode = isinstance(adapter, Qwen35ToolCallAdapter)

    async def _chat_prompt_len(self, messages: list[dict]) -> int:
        payload = {
            "model": self.model,
            "messages": messages,
        }
        tools = getattr(self.adapter, "tools", None) or None
        if tools is not None:
            payload["tools"] = tools
        if self.tool_choice is not None:
            payload["tool_choice"] = self.tool_choice
        async with self.session.post(
            self.root + "/v1/chat/completions/render",
            json=payload,
            timeout=aiohttp.ClientTimeout(total=60),
        ) as resp:
            resp.raise_for_status()
            rendered = await resp.json()
        return len(rendered.get("token_ids") or [])

    async def set_from_probe_call(self, probe_call: dict, main_prompt_or_messages):
        from spork_core import qwen as _qwen
        name = probe_call.get("name")
        args = probe_call.get("arguments", {})
        if hasattr(self.adapter, "format_d3_draft"):
            body = self.adapter.format_d3_draft(probe_call)
        elif self.xml_mode:
            body = f"<function={name}>"
            for k, v in (args or {}).items():
                body += f"\n<parameter={k}>\n{v}\n</parameter>"
            body += "\n</function>"
        else:
            body = "\n" + json.dumps({"name": name, "arguments": args}, ensure_ascii=False)
        if self.tokenizer is None:
            return {"error": "missing tokenizer for D3 draft encoding"}
        draft_tokens = self.tokenizer.encode(body, add_special_tokens=False)
        if isinstance(main_prompt_or_messages, list):
            prompt_len = await self._chat_prompt_len(main_prompt_or_messages)
        else:
            prompt_len = len(self.tokenizer.encode(main_prompt_or_messages, add_special_tokens=False))
        payload = {"request_id": "0", "draft_tokens": draft_tokens, "prompt_len": prompt_len}
        try:
            async with self.session.post(
                self.root + "/spork/set_tokens", json=payload,
                timeout=aiohttp.ClientTimeout(total=30),
            ) as resp:
                resp.raise_for_status()
                r = await resp.json()
            return {"n_draft_tokens": len(draft_tokens), "prompt_len": prompt_len,
                    "status": r.get("status")}
        except Exception as e:
            return {"error": str(e), "n_draft_tokens": len(draft_tokens)}

    async def clear(self):
        try:
            async with self.session.post(
                self.root + "/spork/clear", json={},
                timeout=aiohttp.ClientTimeout(total=30),
            ) as resp:
                resp.raise_for_status()
            async with self.session.get(
                self.root + "/spork/status",
                timeout=aiohttp.ClientTimeout(total=30),
            ) as resp:
                return await resp.json()
        except Exception as e:
            return {"error": str(e)}


def maybe_wrap_tool_executor(args, executor, config_key: str):
    if not args.tool_cache_path:
        if args.simulated_tool_latency_s is not None:
            return LatencyFloorToolExecutor(
                executor,
                floor_s=max(0.0, args.simulated_tool_latency_s),
            )
        return executor
    cache_path = Path(args.tool_cache_path)
    if args.tool_cache_per_config:
        cache_path = cache_path.with_name(
            f"{cache_path.stem}.{config_key}{cache_path.suffix or '.json'}"
        )
    return CachedToolExecutor(
        executor,
        cache_path,
        mode=args.tool_cache_mode,
        replay_latency_s=args.simulated_tool_latency_s,
        strict=not args.tool_cache_allow_miss,
    )


# ── Benchmark loaders ──────────────────────────────────────────────────────

# Default dataset locations (relative to this script). Override via the
# --gaia-dataset-path / --hotpotqa-dataset-path args or the SPORK_GAIA_DATASET /
# SPORK_HOTPOTQA_DATASET environment variables.
DEFAULT_GAIA_DATASET = os.environ.get(
    "SPORK_GAIA_DATASET", str(Path(__file__).parent / "datasets" / "gaia_validation.jsonl")
)
DEFAULT_HOTPOTQA_DATASET = os.environ.get(
    "SPORK_HOTPOTQA_DATASET", str(Path(__file__).parent / "datasets" / "hotpotqa_validation.jsonl")
)


def _load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found: {path}. Provide a local JSONL via the "
            f"--*-dataset-path arg or the corresponding SPORK_*_DATASET env var."
        )
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def load_gaia_tasks(n: int, seed: int, dataset_path: str | os.PathLike) -> list[dict]:
    """Load GAIA tasks from a local JSONL.

    Each line is a GAIA record; the question field is "Question" and the gold
    answer field is "Final answer" (the standard GAIA validation schema).
    """
    tasks = _load_jsonl(Path(dataset_path))
    rng = random.Random(seed)
    rng.shuffle(tasks)
    return tasks[:n] if n and n < len(tasks) else tasks


def load_hotpotqa_tasks(n: int, seed: int, dataset_path: str | os.PathLike) -> list[dict]:
    """Load HotpotQA tasks from a local JSONL.

    Each line carries at least "question" and "answer" (the distractor
    validation split schema); an optional "type" is preserved if present.
    """
    rows = _load_jsonl(Path(dataset_path))
    items = [
        {"question": r["question"], "answer": r["answer"], "type": r.get("type")}
        for r in rows
    ]
    rng = random.Random(seed)
    rng.shuffle(items)
    return items[:n] if n and n < len(items) else items


def load_tau2_tasks(seed: int) -> list[dict]:
    from spork_core.benchmarks.tau2 import load_tasks, build_user_message
    tasks = []
    for domain in ("airline", "retail"):
        raw = load_tasks(domain)
        for t in raw:
            tasks.append({
                "domain": domain,
                "task_id": t.get("task_id", t.get("id")),
                "user_message": build_user_message(t),
                "raw_task": t,
            })
    rng = random.Random(seed)
    rng.shuffle(tasks)
    return tasks


# ── GAIA / HotpotQA tool schemas, prompts, and executors ────────────────────
# The tool SCHEMAS below are plain JSON exposed to the model. The EXECUTORS wire
# the model-issued tool calls to a user-supplied WebSearchBackend/WebBrowseBackend
# (see tool_backends.py). The placeholder backends raise NotImplementedError with
# a clear message; subclass them to evaluate GAIA/HotpotQA.

GAIA_SYSTEM = """You are a helpful assistant that can use tools to answer questions.

Available tools:
- search(query): Search the web and return top results with snippets
- browse_with_goal(url, goal): Fetch and read the text content of a webpage

When you need to call a tool, output it in this exact format:
<tool_call>
{"name": "tool_name", "arguments": {"param": "value"}}
</tool_call>

When you have the final answer, state it clearly prefixed with "FINAL ANSWER: "."""

GAIA_TOOLS = [
    {"type": "function", "function": {"name": "search", "description": "Search the web and return top results with title, url, and snippet.", "parameters": {"type": "object", "properties": {"query": {"type": "string", "description": "Search query"}}, "required": ["query"]}}},
    {"type": "function", "function": {"name": "browse_with_goal", "description": "Fetch a webpage and return its text content.", "parameters": {"type": "object", "properties": {"url": {"type": "string", "description": "URL to visit"}, "goal": {"type": "string", "description": "What to look for on the page"}}, "required": ["url"]}}},
]

HOTPOTQA_SYSTEM = """You are a research assistant that answers multi-hop questions by searching the web for evidence.
Use the search tool to find relevant information. You may search multiple times.
When you have enough information, provide your final answer in <final_answer>YOUR ANSWER</final_answer> tags.
Keep your answer concise — usually a short phrase or entity name."""

HOTPOTQA_TOOLS = [
    {"type": "function", "function": {"name": "search", "description": "Search the web for evidence.", "parameters": {"type": "object", "properties": {"query": {"type": "string", "description": "Search query"}}, "required": ["query"]}}},
]


def _format_search_results(results: list[dict], top_k: int) -> str:
    """Render a WebSearchBackend result list into model-visible text."""
    parts = []
    for r in (results or [])[:top_k]:
        title = r.get("title") or "Untitled"
        url = r.get("url") or ""
        snippet = r.get("snippet") or ""
        head = f"**{title}**" + (f" ({url})" if url else "")
        parts.append(f"{head}: {snippet}")
    return "\n\n".join(parts) or "No results found."


class GaiaToolExecutor:
    """GAIA web tools wired to user-supplied search/browse backends.

    `search` -> WebSearchBackend.search; `browse_with_goal` -> WebBrowseBackend.browse.
    With the placeholder backends these raise NotImplementedError (caught and
    surfaced as the tool result), so plug in real backends to actually run GAIA.
    """

    def __init__(self, search_backend: WebSearchBackend, browse_backend: WebBrowseBackend,
                 search_top_k: int = 8):
        self.search_backend = search_backend
        self.browse_backend = browse_backend
        self.search_top_k = search_top_k
        self.last_real_wall_s = 0.0
        self.last_floor_sleep_s = 0.0

    def execute(self, tool_call: dict) -> tuple[str, float]:
        name = tool_call["name"]
        args = tool_call.get("arguments", {}) or {}
        t0 = time.time()
        try:
            if name == "search":
                results = self.search_backend.search(args.get("query", ""), top_k=self.search_top_k)
                result = _format_search_results(results, self.search_top_k)
            elif name == "browse_with_goal":
                result = self.browse_backend.browse(args.get("url", ""), args.get("goal", ""))
            else:
                result = f"Unknown tool: {name}"
        except Exception as e:
            result = f"Error: {type(e).__name__}: {e}"
        wall = time.time() - t0
        self.last_real_wall_s = wall
        self.last_floor_sleep_s = 0.0
        return str(result)[:10000], wall


class HotpotQAToolExecutor:
    """HotpotQA search wired to a user-supplied WebSearchBackend.

    With the placeholder backend `search` raises NotImplementedError (caught and
    surfaced as the tool result); plug in a real backend to actually run HotpotQA.
    """

    def __init__(self, search_backend: WebSearchBackend, search_top_k: int = 5):
        self.search_backend = search_backend
        self.search_top_k = search_top_k
        self.last_real_wall_s = 0.0
        self.last_floor_sleep_s = 0.0

    def execute(self, tool_call: dict) -> tuple[str, float]:
        args = tool_call.get("arguments", {}) or {}
        t0 = time.time()
        try:
            results = self.search_backend.search(args.get("query", ""), top_k=self.search_top_k)
            result = _format_search_results(results, self.search_top_k)
        except Exception as e:
            result = f"Error: {type(e).__name__}: {e}"
        wall = time.time() - t0
        self.last_real_wall_s = wall
        self.last_floor_sleep_s = 0.0
        return str(result)[:10000], wall


# ── Evaluation metrics ─────────────────────────────────────────────────────

def normalize_answer(s: str) -> str:
    s = s.lower()
    s = re.sub(r'\b(a|an|the)\b', ' ', s)
    s = ''.join(ch for ch in s if ch not in string.punctuation)
    return ' '.join(s.split())


def em_score(pred: str, gold: str) -> bool:
    return normalize_answer(pred) == normalize_answer(gold)


def f1_score(pred: str, gold: str) -> float:
    pred_tokens = normalize_answer(pred).split()
    gold_tokens = normalize_answer(gold).split()
    if not pred_tokens or not gold_tokens:
        return float(pred_tokens == gold_tokens)
    common = set(pred_tokens) & set(gold_tokens)
    if not common:
        return 0.0
    prec = len(common) / len(pred_tokens)
    rec = len(common) / len(gold_tokens)
    return 2 * prec * rec / (prec + rec)


def extract_final_answer(text: str) -> str:
    text = extract_answer(text)
    if m := re.search(r'<final_answer>(.*?)</final_answer>', text, re.DOTALL):
        return m.group(1).strip()
    if m := re.search(r'FINAL ANSWER:\s*(.+)', text, re.IGNORECASE):
        return m.group(1).strip()
    lines = text.strip().split('\n')
    return lines[-1].strip() if lines else ""


# ── tau2 task-success (ACTION reward) scoring ───────────────────────────────
# Wire tau2-bench's per-task ACTION reward as a guardrail. We reuse the OFFICIAL
# tau2 data models + comparison logic
# (tau2.data_model.tasks.Action.compare_with_tool_call), reproducing
# ActionEvaluator.calculate_reward exactly: reward = 1.0 iff every golden
# assistant action is matched by some predicted assistant tool call (name +
# args, restricted to compare_args). This is computed from our recorded turns'
# baseline_tool_name / baseline_tool_args (the calls the agent actually
# committed), so it is deterministic and adds no model calls. We avoid importing
# tau2.evaluator.* (which pulls in pandas, absent from the eval venv) and call
# the dependency-free Action/ToolCall models directly. ADDITIVE: returns None on
# any failure so the existing path is never broken.

def tau2_action_reward(raw_task: dict, turns: list[dict]) -> dict | None:
    """Compute tau2 ACTION reward (task success) for one task from its turns.

    Returns {"reward": 0.0/1.0, "n_matched": int, "n_golden": int,
             "n_pred_tool_calls": int} or None if it cannot be computed.
    """
    try:
        from tau2.data_model.tasks import Task
        from tau2.data_model.message import ToolCall
    except Exception:
        return None
    try:
        task = Task(**raw_task)
    except Exception:
        return None
    ec = task.evaluation_criteria
    # Reconstruct predicted assistant tool calls from committed turns.
    pred = []
    for t in turns:
        if not isinstance(t, dict):
            continue
        if t.get("baseline_has_tool_call") and t.get("baseline_tool_name"):
            try:
                pred.append(ToolCall(
                    id=f"c{len(pred)}",
                    name=t["baseline_tool_name"],
                    arguments=t.get("baseline_tool_args") or {},
                    requestor="assistant",
                ))
            except Exception:
                continue
    if ec is None or not ec.actions:
        # No criteria / no actions → tau2 treats reward as 1.0 (vacuous).
        return {"reward": 1.0, "n_matched": 0, "n_golden": 0,
                "n_pred_tool_calls": len(pred)}
    golden = ec.actions
    n_matched = 0
    for g in golden:
        if any(g.compare_with_tool_call(tc) for tc in pred):
            n_matched += 1
    reward = 1.0 if n_matched == len(golden) else 0.0
    return {"reward": reward, "n_matched": n_matched, "n_golden": len(golden),
            "n_pred_tool_calls": len(pred)}


# ── Main runner ────────────────────────────────────────────────────────────

async def run_task(
    args,
    session: aiohttp.ClientSession,
    client: VllmClient,
    adapter: QwenToolCallAdapter,
    executor,
    messages: list[dict],
    config: str,
    cfg: TurnConfig,
    max_turns: int = 8,
    use_qwen35_messages: bool = False,
    probe_client: VllmClient | None = None,
    probe_adapter=None,
) -> list[dict]:
    turns = []
    for turn_idx in range(1, max_turns + 1):
        if config == "baseline":
            entry = await run_baseline_turn(session, client, adapter, messages, executor, turn_idx, cfg)
        elif config in ("spork_d2", "d1_d2", "d1_d2_d3"):
            retry_cfg = RetryTurnConfig(
                max_retries=args.d2_max_retries,
                confidence_threshold=args.d2_confidence_threshold,
                retry_token_step=args.d2_retry_token_step,
                snap_to_sentence=args.d2_snap_to_sentence,
                probe_max_tokens=args.d2_probe_max_tokens,
                probe_timeout_s=args.d2_probe_timeout_s,
                baseline_max_tokens=cfg.baseline_max_tokens,
                context_window_tokens=cfg.context_window_tokens,
                enable_thinking=cfg.enable_thinking,
                first_token_timeout_s=cfg.first_token_timeout_s,
                seed=cfg.seed,
                main_stop=cfg.main_stop,
                theta_decay_per_retry=args.d2_theta_decay_per_retry,
                min_tokens_first_probe=args.d2_min_tokens_first_probe,
                continue_after_dispatch=args.d2_continue_after_dispatch,
            )
            # D3-in-HTTP: wire sidecar token-injection hooks for d1_d2_d3.
            # Acceptance metrics (gate_match/hit_name) are computed identically to
            # d1_d2; D3 only injects probe draft tokens to accelerate main decode.
            if config == "d1_d2_d3":
                sidecar = _SporkSidecar(
                    session,
                    client.base_url,
                    adapter.tokenizer,
                    adapter,
                    model=client.model,
                    tool_choice=getattr(client, "tool_choice", None),
                )
                retry_cfg.d3_set = sidecar.set_from_probe_call
                retry_cfg.d3_clear = sidecar.clear
            entry = await run_spork_retry_turn(session, client, adapter, messages, executor, turn_idx, retry_cfg)
        elif config == "spork_cross":
            if probe_client is None or probe_adapter is None:
                raise ValueError("spork_cross requires probe_client and probe_adapter")
            entry = await run_spork_cross_turn(
                session, client, adapter, probe_client, probe_adapter,
                messages, executor, turn_idx, cfg,
            )
        elif config in ("d1", "spork"):
            entry = await run_spork_turn(session, client, adapter, messages, executor, turn_idx, cfg)
        else:
            raise ValueError(f"unknown config: {config}")
        turns.append(entry)
        if not entry.get("baseline_has_tool_call"):
            break
    # Forced conclusion: if we hit max_turns and model is still calling tools,
    # do one more turn with tools removed so model must give a final answer
    if (
        len(turns) == max_turns
        and turns[-1].get("baseline_has_tool_call")
        and not getattr(args, "no_forced_conclusion", False)
    ):
        conclude_msg = list(messages)
        conclude_msg.append({"role": "user", "content": "You have used all your available tool calls. Based on the information you have gathered, please provide your final answer now. State it as: FINAL ANSWER: <your concise answer>"})
        no_tool_adapter = type(adapter)(adapter.tokenizer, [])
        conclude_entry = await run_baseline_turn(session, client, no_tool_adapter, conclude_msg, executor, max_turns + 1, cfg)
        conclude_entry["mode"] = "forced_conclusion"
        turns.append(conclude_entry)
    return turns


async def run_benchmark(args):
    from transformers import AutoTokenizer
    if args.tokenizer_path and str(args.tokenizer_path).lower() == "none":
        tokenizer = None
    else:
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path, trust_remote_code=True)

    client = VllmClient(
        base_url=args.model_url,
        model=args.model_name,
        timeout_s=600,
    )
    if "3.5" in args.model_name or "qwen3.5" in args.model_name.lower():
        client.main_stop_token_ids = (248058,)
    elif "a3b" in args.model_name.lower() and "35" in args.model_name.lower():
        # Qwen3.5-35B-A3B emits JSON tool calls (uses the JSON adapter, since "3.5"
        # is not a substring of "qwen35-a3b"), but its <tool_call> token id is 248058,
        # not Qwen3's 151657. Set it explicitly so main decode stops at the tool-call
        # boundary, matching the Qwen3-8B/4B methodology.
        client.main_stop_token_ids = (248058,)
    elif "phi" in args.model_name.lower():
        # Phi-4 has no special <tool_call> token (vocab_size=100352) — rely on string stop only
        client.main_stop_token_ids = ()
    log(f"Waiting for vLLM at {args.model_url}...")
    await client.wait_ready(deadline_s=60)
    log("vLLM ready.")

    # Optional cross-model probe client/tokenizer (for spork_cross config)
    probe_client = None
    probe_tokenizer = None
    if args.probe_model_url:
        probe_tok_path = args.probe_tokenizer_path or args.tokenizer_path
        probe_tokenizer = AutoTokenizer.from_pretrained(probe_tok_path, trust_remote_code=True)
        probe_client = VllmClient(base_url=args.probe_model_url, model=args.probe_model_name)
        # Qwen3-4B uses the same JSON <tool_call> token id as Qwen3-32B (151657).
        log(f"Waiting for PROBE vLLM at {args.probe_model_url} (model={args.probe_model_name})...")
        await probe_client.wait_ready(deadline_s=60)
        log("Probe vLLM ready.")

    today = time.strftime("%Y%m%d")
    out_dir = Path(args.results_root) / f"{args.benchmark}_{args.model_name}_n{args.n}_{today}"
    out_dir.mkdir(parents=True, exist_ok=True)
    log(f"Output: {out_dir}")

    # Tool backends for the web-search benchmarks.
    #   HotpotQA: defaults to the built-in free Wikipedia backend (HotpotQA is
    #     Wikipedia QA), so it runs out-of-the-box with no API key.
    #   GAIA: uses the placeholder backends (raise NotImplementedError) — plug in
    #     your own web search/browse in tool_backends.py.
    if args.benchmark == "hotpotqa":
        search_backend = WikipediaSearchBackend()
    else:
        search_backend = WebSearchBackend()
    browse_backend = WebBrowseBackend()

    # Load tasks
    if args.benchmark == "gaia":
        tasks = load_gaia_tasks(args.n, args.seed, args.gaia_dataset_path)
        system_prompt = GAIA_SYSTEM
        tools = GAIA_TOOLS
    elif args.benchmark == "hotpotqa":
        tasks = load_hotpotqa_tasks(args.n, args.seed, args.hotpotqa_dataset_path)
        system_prompt = HOTPOTQA_SYSTEM
        tools = HOTPOTQA_TOOLS
    elif args.benchmark == "tau2":
        all_tau2 = load_tau2_tasks(args.seed)
        tasks = all_tau2[:args.n] if args.n < len(all_tau2) else all_tau2
        system_prompt = None
        tools = None
    else:
        raise ValueError(f"Unknown benchmark: {args.benchmark}")

    log(f"Loaded {len(tasks)} tasks for {args.benchmark}")

    # Apply shard slicing
    if args.end > 0:
        tasks = tasks[args.start:args.end]
    elif args.start > 0:
        tasks = tasks[args.start:]
    log(f"Running shard [{args.start}:{args.end or 'end'}] = {len(tasks)} tasks")

    await run_discarded_warmup(
        args, client, args.benchmark, tasks, system_prompt, tools
    )

    configs = [c.strip() for c in args.configs.split(",")]
    main_completion_max_tokens = min(args.max_tokens, 8192)
    cfg = TurnConfig(
        enable_thinking=args.enable_thinking,
        baseline_max_tokens=main_completion_max_tokens,
        context_window_tokens=args.max_tokens,
        seed=args.seed,
    )

    all_results = {}
    for config in configs:
        config_key = config
        log(f"=== Config: {config_key} ===")
        config_results = []

        for task_idx, task in enumerate(tasks):
            task_id = str(task.get("task_id", task.get("question", f"task_{task_idx}")))[:80]
            log(f"  [{task_idx+1}/{len(tasks)}] {config_key} :: {str(task_id)[:50]}")

            # Build messages + executor per benchmark
            use_qwen35_adapter = "3.5" in args.model_name or "qwen3.5" in args.model_name.lower()
            adapter_cls = Qwen35ToolCallAdapter if use_qwen35_adapter else QwenToolCallAdapter
            if args.benchmark == "tau2":
                from spork_core.benchmarks.tau2 import load_system_prompt, load_tools, get_environment
                domain = task["domain"]
                tau2_sys = load_system_prompt(domain)
                tau2_tools = load_tools(domain)
                adapter = adapter_cls(tokenizer, tau2_tools)
                env = get_environment(domain)
                executor = Tau2ToolExecutor(env, floor_s=args.tau2_floor)
                executor = maybe_wrap_tool_executor(args, executor, config_key)
                messages = [
                    {"role": "system", "content": tau2_sys},
                    {"role": "user", "content": task["user_message"]},
                ]
            elif args.benchmark == "gaia":
                adapter = adapter_cls(tokenizer, tools)
                executor = GaiaToolExecutor(
                    search_backend, browse_backend, search_top_k=args.gaia_search_top_k
                )
                executor = maybe_wrap_tool_executor(args, executor, config_key)
                messages = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": task["Question"]},
                ]
            elif args.benchmark == "hotpotqa":
                adapter = adapter_cls(tokenizer, tools)
                executor = HotpotQAToolExecutor(search_backend)
                executor = maybe_wrap_tool_executor(args, executor, config_key)
                messages = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": task["question"]},
                ]
            else:
                raise ValueError(f"Unknown benchmark: {args.benchmark}")

            t0 = time.time()
            max_turns = args.max_turns if args.max_turns > 0 else 8
            use_qwen35 = "3.5" in args.model_name or "qwen3.5" in args.model_name.lower()
            # Build a per-task probe adapter (same class + tools, probe tokenizer)
            probe_adapter = None
            if probe_tokenizer is not None:
                probe_adapter = type(adapter)(probe_tokenizer, adapter.tools)
            try:
                async with aiohttp.ClientSession(trust_env=False) as task_session:
                    task_coro = run_task(
                        args, task_session, client, adapter, executor,
                        messages, config_key, cfg, max_turns=max_turns,
                        use_qwen35_messages=use_qwen35,
                        probe_client=probe_client, probe_adapter=probe_adapter,
                    )
                    if getattr(args, "task_timeout_s", 0) and args.task_timeout_s > 0:
                        turns = await asyncio.wait_for(task_coro, timeout=args.task_timeout_s)
                    else:
                        turns = await task_coro
            except asyncio.TimeoutError:
                log(f"    ERROR: task timeout after {args.task_timeout_s}s")
                turns = [{"error": f"task timeout after {args.task_timeout_s}s"}]
            except Exception as e:
                log(f"    ERROR: {e}")
                turns = [{"error": str(e)}]
            task_wall_s = time.time() - t0

            # Extract answer and compute metrics
            final_text = ""
            if turns and not turns[-1].get("error"):
                last = turns[-1]
                final_text = last.get("final_answer", last.get("baseline_text", ""))
            pred = extract_final_answer(final_text)
            gold = ""
            if args.benchmark == "gaia":
                gold = task.get("Final answer", "")
            elif args.benchmark == "hotpotqa":
                gold = task.get("answer", "")

            em = em_score(pred, gold) if gold else None
            f1 = f1_score(pred, gold) if gold else None

            result_entry = {
                "task_idx": task_idx,
                "task_id": task_id,
                "config": config_key,
                "benchmark": args.benchmark,
                "n_turns": len(turns),
                "wall_s": round(task_wall_s, 3),
                "pred": pred,
                "gold": gold,
                "em": em,
                "f1": f1,
                "turns": turns,
            }
            if args.benchmark == "tau2":
                result_entry["domain"] = task["domain"]
                # tau2 ACTION-reward task-success (guardrail signal).
                ts = tau2_action_reward(task.get("raw_task") or {}, turns)
                result_entry["tau2_success"] = ts

            config_results.append(result_entry)

            # Write per-task JSON
            safe_id = re.sub(r'[^a-zA-Z0-9_-]', '_', str(task_id))[:60]
            if args.benchmark == "tau2":
                safe_id = f"{task.get('domain', 'x')}_{safe_id}"
            write_json(out_dir / config_key / f"{safe_id}.json", result_entry)

        all_results[config_key] = config_results
        log(f"  Config {config_key} done: {len(config_results)} tasks")

    # Write summary
    summary = {
        "benchmark": args.benchmark,
        "model": args.model_name,
        "n": len(tasks),
        "run": {
            "seed": args.seed,
            "configs": configs,
            "max_tokens": args.max_tokens,
            "completion_max_tokens": args.completion_max_tokens,
            "main_completion_max_tokens": main_completion_max_tokens,
            "tau2_floor": args.tau2_floor,
            "task_timeout_s": args.task_timeout_s,
            "warmup_requests": args.warmup_requests,
            "warmup_max_tokens": args.warmup_max_tokens,
        },
        "server": {
            "model_url": args.model_url,
            "tag": args.server_tag or None,
            "tp": args.server_tp or None,
            "max_model_len": args.server_max_model_len or None,
            "max_num_batched_tokens": args.server_max_num_batched_tokens or None,
            "speculative_config": args.server_speculative_config or None,
            "parser_mode": args.server_parser_mode or None,
        },
        "configs": {},
    }
    for config_key, results in all_results.items():
        errored = [r for r in results if _result_has_error(r)]
        walls = [r["wall_s"] for r in results if r.get("wall_s") and not _result_has_error(r)]
        ems = [r["em"] for r in results if r.get("em") is not None]
        f1s = [r["f1"] for r in results if r.get("f1") is not None]
        n_errors = len(errored)
        wall_p95_s = _p95_nearest_rank(walls)

        summary["configs"][config_key] = {
            "n_tasks": len(results),
            "n_errors": n_errors,
            "n_wall_samples": len(walls),
            "n_wall_excluded_errors": len(errored),
            "em_mean": round(statistics.mean(ems), 4) if ems else None,
            "f1_mean": round(statistics.mean(f1s), 4) if f1s else None,
            "wall_mean_s": round(statistics.mean(walls), 3) if walls else None,
            "wall_p50_s": round(statistics.median(walls), 3) if walls else None,
            "wall_p95_s": round(wall_p95_s, 3) if wall_p95_s is not None else None,
            "turns_mean": round(statistics.mean(r["n_turns"] for r in results), 2) if results else 0,
        }
        # tau2 ACTION-reward task-success aggregate (guardrail).
        if args.benchmark == "tau2":
            rewards = [r["tau2_success"]["reward"] for r in results
                       if r.get("tau2_success") is not None]
            scored = [r for r in results if r.get("tau2_success") is not None]
            summary["configs"][config_key]["tau2_success_rate"] = (
                round(statistics.mean(rewards), 4) if rewards else None)
            summary["configs"][config_key]["tau2_n_scored"] = len(scored)
            summary["configs"][config_key]["tau2_actions_matched"] = sum(
                r["tau2_success"]["n_matched"] for r in scored)
            summary["configs"][config_key]["tau2_actions_total"] = sum(
                r["tau2_success"]["n_golden"] for r in scored)

    write_json(out_dir / "summary.json", summary)
    log(f"\n{'='*60}")
    log(f"SUMMARY: {args.benchmark} / {args.model_name}")
    log(json.dumps(summary["configs"], indent=2))
    log(f"Results at: {out_dir}")
    return summary


def main():
    parser = argparse.ArgumentParser(description="Unified SPORK multi-benchmark eval")
    parser.add_argument("--model-url", default="http://localhost:8001/v1")
    parser.add_argument("--model-name", default="qwen3-4b")
    parser.add_argument("--probe-model-url", default=None,
                        help="Optional separate vLLM endpoint for the cross-model probe (spork_cross config)")
    parser.add_argument("--probe-model-name", default="qwen3-4b",
                        help="Served model name for the probe endpoint")
    parser.add_argument("--probe-tokenizer-path", default=None,
                        help="Tokenizer path for the probe model (defaults to --tokenizer-path)")
    parser.add_argument("--benchmark", choices=["tau2", "gaia", "hotpotqa"], default="tau2")
    parser.add_argument("--configs", default="baseline,d1")
    parser.add_argument("--n", type=int, default=0,
                        help="Number of tasks (0=auto: all tau2, 165 gaia, 200 hotpotqa).")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-tokens", type=int, default=32768)
    parser.add_argument("--completion-max-tokens", type=int, default=2048,
                        help="Max main completion tokens per turn.")
    parser.add_argument("--enable-thinking", action="store_true", default=True)
    parser.add_argument("--no-thinking", dest="enable_thinking", action="store_false")
    parser.add_argument("--results-root", default=str(RESULTS_ROOT))
    parser.add_argument("--tau2-floor", type=float, default=2.0)
    parser.add_argument("--gaia-dataset-path", default=DEFAULT_GAIA_DATASET,
                        help="Path to the GAIA validation JSONL (question field 'Question', "
                             "gold field 'Final answer'). Env: SPORK_GAIA_DATASET.")
    parser.add_argument("--hotpotqa-dataset-path", default=DEFAULT_HOTPOTQA_DATASET,
                        help="Path to the HotpotQA validation JSONL ('question'/'answer' fields). "
                             "Env: SPORK_HOTPOTQA_DATASET.")
    parser.add_argument("--gaia-search-top-k", type=int, default=8,
                        help="Number of GAIA web-search results exposed to the model.")
    parser.add_argument("--start", type=int, default=0, help="Start index for sharding")
    parser.add_argument("--end", type=int, default=0, help="End index for sharding (0=all)")
    parser.add_argument("--max-turns", type=int, default=0,
                        help="Max turns per task (0=auto: 8).")
    parser.add_argument("--task-timeout-s", type=float, default=0,
                        help="Wall-clock timeout per task; 0 disables.")
    parser.add_argument("--warmup-requests", type=int,
                        default=_env_int("SPORK_WARMUP_REQUESTS", 0),
                        help="Discarded warmup rounds before timed tasks.")
    parser.add_argument("--warmup-max-tokens", type=int,
                        default=_env_int("SPORK_WARMUP_MAX_TOKENS", 512),
                        help="Max tokens for the longer discarded warmup request.")
    parser.add_argument("--server-tag", default=os.environ.get("SPORK_SERVER_TAG", ""),
                        help="Free-form server provenance tag stamped into summary.json.")
    parser.add_argument("--server-tp", type=int,
                        default=_env_int("SPORK_SERVER_TP", 0),
                        help="Tensor parallel size of the served model.")
    parser.add_argument("--server-max-model-len", type=int,
                        default=_env_int("SPORK_SERVER_MAX_MODEL_LEN", 0),
                        help="Served --max-model-len stamped into summary.json.")
    parser.add_argument("--server-max-num-batched-tokens", type=int,
                        default=_env_int("SPORK_SERVER_MAX_NUM_BATCHED_TOKENS", 0),
                        help="Served --max-num-batched-tokens stamped into summary.json.")
    parser.add_argument("--server-speculative-config",
                        default=os.environ.get("SPORK_SERVER_SPECULATIVE_CONFIG", ""),
                        help="Active vLLM speculative config stamped into summary.json.")
    parser.add_argument("--server-parser-mode",
                        default=os.environ.get("SPORK_SERVER_PARSER_MODE", ""),
                        help="Tool parser mode stamped into summary.json.")
    parser.add_argument("--no-forced-conclusion", action="store_true",
                        help="Skip final no-tool answer turn after max_turns.")
    parser.add_argument("--tokenizer-path", default=None,
                        help="Path to tokenizer (defaults to a Qwen3-4B HF id)")
    parser.add_argument("--d2-max-retries", type=int, default=5,
                        help="Maximum D2 probe attempts per turn for d1_d2.")
    parser.add_argument("--d2-retry-token-step", type=int, default=50,
                        help="Main decoded-token cadence between D2 retry probes.")
    parser.add_argument("--d2-min-tokens-first-probe", type=int, default=0,
                        help="Delay first D2 probe until main has decoded at least this many tokens.")
    parser.add_argument("--d2-confidence-threshold", type=float, default=0.90,
                        help="Probe logprob confidence threshold for D2 commit.")
    parser.add_argument("--d2-theta-decay-per-retry", type=float, default=0.0,
                        help="Subtract this threshold amount after each D2 retry.")
    parser.add_argument("--d2-probe-max-tokens", type=int, default=160,
                        help="Max tokens for each D2 probe completion.")
    parser.add_argument("--d2-probe-timeout-s", type=float, default=0.0,
                        help="Optional wall-clock timeout for each D2 probe request; 0 disables.")
    parser.add_argument("--d2-snap-to-sentence", action=argparse.BooleanOptionalAction,
                        default=True,
                        help="Wait for sentence/newline boundary before retries after retry 0.")
    parser.add_argument("--d2-continue-after-dispatch", action="store_true",
                        help="Keep probing after an early confident dispatch and replace it with later confident probes.")
    parser.add_argument("--tool-cache-path", default=None,
                        help="JSON cache path for deterministic tool record/replay.")
    parser.add_argument("--tool-cache-mode", choices=["record", "replay"], default="record",
                        help="record executes real tools and writes cache; replay serves cached outputs.")
    parser.add_argument("--simulated-tool-latency-s", type=float, default=None,
                        help="Replay-mode fixed latency per tool call; default replays recorded wall time.")
    parser.add_argument("--tool-cache-allow-miss", action="store_true",
                        help="In replay mode, fall back to the real executor on cache miss.")
    parser.add_argument("--tool-cache-per-config", action="store_true",
                        help="Write/read separate cache files per config instead of sharing one cache.")
    args = parser.parse_args()

    # Default N per benchmark (tau2 = all).
    if args.n == 0:
        args.n = {"tau2": 999, "gaia": 165, "hotpotqa": 200}[args.benchmark]

    # Default tokenizer path
    if args.tokenizer_path is None:
        model_map = {
            "qwen3-4b": os.environ.get("QWEN3_4B_PATH", "Qwen/Qwen3-4B"),
            "qwen3-8b": os.environ.get("QWEN3_8B_PATH", "Qwen/Qwen3-8B"),
            "qwen3-32b": os.environ.get("QWEN3_32B_PATH", "Qwen/Qwen3-32B"),
        }
        args.tokenizer_path = model_map.get(args.model_name, model_map["qwen3-4b"])

    # Default probe tokenizer path from probe model name
    if args.probe_model_url and args.probe_tokenizer_path is None:
        model_map_probe = {
            "qwen3-4b": os.environ.get("QWEN3_4B_PATH", "Qwen/Qwen3-4B"),
            "qwen3-8b": os.environ.get("QWEN3_8B_PATH", "Qwen/Qwen3-8B"),
            "qwen3-32b": os.environ.get("QWEN3_32B_PATH", "Qwen/Qwen3-32B"),
        }
        args.probe_tokenizer_path = model_map_probe.get(args.probe_model_name,
                                                        model_map_probe["qwen3-4b"])

    asyncio.run(run_benchmark(args))


if __name__ == "__main__":
    main()
