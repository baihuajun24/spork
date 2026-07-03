"""Tool executors used by SPORK turn runners."""
from __future__ import annotations

import copy
import json
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol


@dataclass
class SpeculativeToolExecution:
    text: str
    wall_s: float
    real_wall_s: float
    artificial_sleep_s: float
    state: Any = None


class ToolExecutor(Protocol):
    last_real_wall_s: float
    last_floor_sleep_s: float

    def execute(self, tool_call: dict) -> tuple[str, float]:
        ...


class SimulatedToolExecutor:
    def __init__(self, latency_s: float = 0.5, result: str = "<mock-result>"):
        self.latency_s = latency_s
        self.result = result
        self.last_real_wall_s = 0.0
        self.last_floor_sleep_s = latency_s

    def execute(self, tool_call: dict) -> tuple[str, float]:
        t0 = time.time()
        time.sleep(max(0.0, self.latency_s))
        wall = time.time() - t0
        self.last_real_wall_s = 0.0
        self.last_floor_sleep_s = wall
        return self.result, wall

    def execute_speculative(self, tool_call: dict) -> SpeculativeToolExecution:
        text, wall = self.execute(tool_call)
        return SpeculativeToolExecution(
            text=text,
            wall_s=wall,
            real_wall_s=self.last_real_wall_s,
            artificial_sleep_s=self.last_floor_sleep_s,
        )


class LatencyFloorToolExecutor:
    """Delegate to a real executor while enforcing a minimum observed latency."""

    def __init__(self, executor: ToolExecutor, floor_s: float = 0.0):
        self.executor = executor
        self.floor_s = floor_s
        self.last_real_wall_s = 0.0
        self.last_floor_sleep_s = 0.0

    def execute(self, tool_call: dict) -> tuple[str, float]:
        t0 = time.time()
        text, _wall = self.executor.execute(tool_call)
        real_wall = time.time() - t0
        sleep_s = max(0.0, self.floor_s - real_wall)
        if sleep_s > 0:
            time.sleep(sleep_s)
        wall = time.time() - t0
        self.last_real_wall_s = getattr(self.executor, "last_real_wall_s", real_wall)
        self.last_floor_sleep_s = (
            getattr(self.executor, "last_floor_sleep_s", 0.0) + max(0.0, wall - real_wall)
        )
        return text, wall

    def execute_speculative(self, tool_call: dict) -> SpeculativeToolExecution:
        if hasattr(self.executor, "execute_speculative"):
            t0 = time.time()
            execution = self.executor.execute_speculative(tool_call)
            elapsed = time.time() - t0
            sleep_s = max(0.0, self.floor_s - elapsed)
            if sleep_s > 0:
                time.sleep(sleep_s)
            wall = time.time() - t0
            execution.wall_s = wall
            execution.artificial_sleep_s += max(0.0, wall - elapsed)
            self.last_real_wall_s = execution.real_wall_s
            self.last_floor_sleep_s = execution.artificial_sleep_s
            return execution
        text, wall = self.execute(tool_call)
        return SpeculativeToolExecution(
            text=text,
            wall_s=wall,
            real_wall_s=self.last_real_wall_s,
            artificial_sleep_s=self.last_floor_sleep_s,
        )

    def commit_speculative(self, execution: SpeculativeToolExecution) -> None:
        if hasattr(self.executor, "commit_speculative"):
            self.executor.commit_speculative(execution)
        self.last_real_wall_s = execution.real_wall_s
        self.last_floor_sleep_s = execution.artificial_sleep_s


def tool_call_cache_key(tool_call: dict) -> str:
    return json.dumps(
        {
            "name": tool_call.get("name") or "",
            "arguments": tool_call.get("arguments") or {},
        },
        sort_keys=True,
        ensure_ascii=False,
        default=str,
    )


class CachedToolExecutor:
    """Record/replay wrapper for deterministic tool latency experiments.

    `mode="record"` delegates to the wrapped executor, stores the result by
    normalized tool call, and returns the real result. `mode="replay"` returns
    the cached result and sleeps for either `replay_latency_s` or the recorded
    wall time. This is intended for read-only web/search benchmarks where the
    model-visible tool output should be held fixed across SPORK variants.
    """

    def __init__(
        self,
        executor: ToolExecutor,
        cache_path: str | Path,
        *,
        mode: str = "record",
        replay_latency_s: float | None = None,
        strict: bool = True,
    ):
        if mode not in {"record", "replay"}:
            raise ValueError(f"unknown cache mode: {mode}")
        self.executor = executor
        self.cache_path = Path(cache_path)
        self.mode = mode
        self.replay_latency_s = replay_latency_s
        self.strict = strict
        self.last_real_wall_s = 0.0
        self.last_floor_sleep_s = 0.0
        self._lock = threading.Lock()
        self._cache = self._load_cache()

    def _load_cache(self) -> dict[str, dict]:
        if not self.cache_path.exists():
            return {}
        try:
            data = json.loads(self.cache_path.read_text())
            return data if isinstance(data, dict) else {}
        except json.JSONDecodeError:
            return {}

    def _save_cache(self) -> None:
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.cache_path.with_suffix(self.cache_path.suffix + ".tmp")
        tmp.write_text(json.dumps(self._cache, indent=2, ensure_ascii=False, default=str))
        tmp.replace(self.cache_path)

    def execute(self, tool_call: dict) -> tuple[str, float]:
        key = tool_call_cache_key(tool_call)
        if self.mode == "replay":
            with self._lock:
                record = self._cache.get(key)
            if record is None:
                if self.strict:
                    raise KeyError(f"tool call not found in replay cache: {key}")
                text, wall = self.executor.execute(tool_call)
                self.last_real_wall_s = self.executor.last_real_wall_s
                self.last_floor_sleep_s = self.executor.last_floor_sleep_s
                return text, wall
            sleep_s = (
                self.replay_latency_s
                if self.replay_latency_s is not None
                else float(record.get("wall_s", 0.0) or 0.0)
            )
            t0 = time.time()
            if sleep_s > 0:
                time.sleep(sleep_s)
            wall = time.time() - t0
            self.last_real_wall_s = 0.0
            self.last_floor_sleep_s = wall
            return str(record.get("text", "")), wall

        text, wall = self.executor.execute(tool_call)
        record = {
            "tool_call": {
                "name": tool_call.get("name") or "",
                "arguments": tool_call.get("arguments") or {},
            },
            "text": text,
            "wall_s": wall,
            "real_wall_s": self.executor.last_real_wall_s,
            "artificial_sleep_s": self.executor.last_floor_sleep_s,
            "recorded_at": time.time(),
        }
        with self._lock:
            self._cache[key] = record
            self._save_cache()
        self.last_real_wall_s = self.executor.last_real_wall_s
        self.last_floor_sleep_s = self.executor.last_floor_sleep_s
        return text, wall

    def execute_speculative(self, tool_call: dict) -> SpeculativeToolExecution:
        text, wall = self.execute(tool_call)
        return SpeculativeToolExecution(
            text=text,
            wall_s=wall,
            real_wall_s=self.last_real_wall_s,
            artificial_sleep_s=self.last_floor_sleep_s,
        )

    def commit_speculative(self, execution: SpeculativeToolExecution) -> None:
        self.last_real_wall_s = execution.real_wall_s
        self.last_floor_sleep_s = execution.artificial_sleep_s


class Tau2ToolExecutor:
    def __init__(self, env: Any, floor_s: float = 0.5, extra_s: float = 0.0):
        self.env = env
        self.floor_s = floor_s
        self.extra_s = extra_s
        self.last_real_wall_s = 0.0
        self.last_floor_sleep_s = 0.0

    def execute(self, tool_call: dict) -> tuple[str, float]:
        text, wall, real_wall, artificial_sleep = self._execute_on_env(self.env, tool_call)
        self.last_real_wall_s = real_wall
        self.last_floor_sleep_s = artificial_sleep
        return text, wall

    def execute_speculative(self, tool_call: dict) -> SpeculativeToolExecution:
        spec_env = copy.deepcopy(self.env)
        text, wall, real_wall, artificial_sleep = self._execute_on_env(spec_env, tool_call)
        return SpeculativeToolExecution(
            text=text,
            wall_s=wall,
            real_wall_s=real_wall,
            artificial_sleep_s=artificial_sleep,
            state=spec_env,
        )

    def commit_speculative(self, execution: SpeculativeToolExecution) -> None:
        if execution.state is not None:
            self.env = execution.state
        self.last_real_wall_s = execution.real_wall_s
        self.last_floor_sleep_s = execution.artificial_sleep_s

    def _execute_on_env(self, env: Any, tool_call: dict) -> tuple[str, float, float, float]:
        name = tool_call["name"]
        args = tool_call.get("arguments", {}) or {}
        t0 = time.time()
        func = getattr(env.tools, name, None)
        if func is None:
            result_obj: Any = {"error": f"unknown tool: {name}"}
        else:
            try:
                result_obj = func(**args)
            except Exception as exc:  # noqa: BLE001
                result_obj = {"error": f"{type(exc).__name__}: {exc}"}
        real_wall = time.time() - t0
        sleep_s = 0.0
        if self.extra_s > 0:
            sleep_s = self.extra_s
        elif self.floor_s > 0 and real_wall < self.floor_s:
            sleep_s = self.floor_s - real_wall
        if sleep_s > 0:
            time.sleep(sleep_s)
        wall = time.time() - t0
        artificial_sleep = max(0.0, wall - real_wall)
        if hasattr(result_obj, "model_dump_json"):
            text = result_obj.model_dump_json()
        elif isinstance(result_obj, (dict, list)):
            text = json.dumps(result_obj, ensure_ascii=False, default=str)
        else:
            text = str(result_obj)
        if len(text) > 15000:
            text = text[:15000] + "...[truncated]"
        return text, wall, real_wall, artificial_sleep


def execute_speculative(executor: ToolExecutor, tool_call: dict) -> SpeculativeToolExecution:
    method = getattr(executor, "execute_speculative", None)
    if callable(method):
        return method(tool_call)
    text, wall = executor.execute(tool_call)
    return SpeculativeToolExecution(
        text=text,
        wall_s=wall,
        real_wall_s=executor.last_real_wall_s,
        artificial_sleep_s=executor.last_floor_sleep_s,
    )


def commit_speculative(executor: ToolExecutor, execution: SpeculativeToolExecution) -> None:
    method = getattr(executor, "commit_speculative", None)
    if callable(method):
        method(execution)
        return
    executor.last_real_wall_s = execution.real_wall_s
    executor.last_floor_sleep_s = execution.artificial_sleep_s
