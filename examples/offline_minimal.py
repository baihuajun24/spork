#!/usr/bin/env python3
"""Synthetic GLM-format fixture through the real core. No model/network/benchmark."""
import asyncio
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from spork_core.executors import SpeculativeToolExecution
from spork_core.glm53 import Glm53ToolCallAdapter
from spork_core.turn_runner import TurnConfig, run_baseline_turn, run_spork_turn

TOOLS = [{"type": "function", "function": {"name": "lookup", "parameters": {
    "type": "object", "properties": {"number": {"type": "integer"}}, "required": ["number"]}}}]


class FixtureClient:
    def __init__(self, main_number=2):
        self.main_number = main_number
        self.probes = 0

    async def render_chat_token_count(self, *args, **kwargs):
        return 10

    async def stream_chat(self, session, messages, **kwargs):
        await kwargs["on_first_token"](.001, "inspect")
        if kwargs.get("on_token"):
            await kwargs["on_token"]("inspect", 1)
        await asyncio.sleep(.05)  # Allow the speculative worker to run in this fixture.
        return dict(text="", reasoning="inspect", first_token_s=.001, wall_s=.05,
                    tool_calls=[dict(id="fixture-call", type="function", function=dict(
                        name="lookup", arguments=json.dumps({"number": self.main_number})))])

    async def complete_chat(self, *args, **kwargs):
        self.probes += 1
        return {"message": {"content": "lookup<arg_key>number</arg_key><arg_value>2</arg_value></tool_call>"}}, .001


class FixtureExecutor:
    last_real_wall_s = 0.0
    last_floor_sleep_s = 0.0

    def __init__(self):
        self.state = []
        self.speculative_calls = 0
        self.canonical_calls = 0

    def execute(self, call):
        self.canonical_calls += 1
        self.state.append(call["arguments"]["number"])
        return json.dumps(self.state), 0.0

    def execute_many(self, calls):
        return [self.execute(call) for call in calls]

    def execute_speculative(self, call):
        self.speculative_calls += 1
        isolated = self.state + [call["arguments"]["number"]]
        return SpeculativeToolExecution(json.dumps(isolated), 0.0, 0.0, 0.0, state=isolated)

    def commit_speculative(self, execution):
        self.state = execution.state


async def run_case(mode, main_number=2):
    client, executor = FixtureClient(main_number), FixtureExecutor()
    messages = [{"role": "user", "content": "Use lookup."}]
    runner = run_baseline_turn if mode == "baseline" else run_spork_turn
    result = await runner(None, client, Glm53ToolCallAdapter(tools=TOOLS), messages,
                          executor, 0, TurnConfig(enable_thinking=True, probe_after_tokens=1))
    return result, client, executor, messages


async def main():
    output = {}
    for mode in ["baseline", "spork"]:
        result, client, executor, _ = await run_case(mode)
        assert executor.state == [2]
        if mode == "baseline":
            assert client.probes == executor.speculative_calls == 0
        else:
            assert result["spec_used"] and executor.canonical_calls == 0
        output[mode] = dict(final_state=executor.state, probe_requests=client.probes,
                            speculative_committed=result.get("spec_used", False))
    print(json.dumps(dict(kind="synthetic_offline_fixture", benchmark_score=None,
                          measured_speedup=None, cases=output), indent=2))


if __name__ == "__main__":
    asyncio.run(main())
