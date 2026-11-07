import asyncio
from dataclasses import fields
import json
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from examples.offline_minimal import FixtureClient, FixtureExecutor, TOOLS, run_case
from spork_core.glm53 import Glm53ToolCallAdapter
from spork_core.model_setup import configure_client, make_adapter, probe_targets
from spork_core.turn_runner import (TurnConfig, RetryTurnConfig, run_baseline_turn,
    run_spork_turn, run_spork_retry_turn, _portfolio_calls_exact, _append_tool_messages)
from spork_core.vllm_client import VllmClient


class CompatibilityTests(unittest.IsolatedAsyncioTestCase):
    def test_original_field_order_and_defaults(self):
        contract = json.loads((ROOT / 'tests/v1_contract.json').read_text())
        for cls in (TurnConfig, RetryTurnConfig):
            actual = [{'name': f.name, 'default': f.default} for f in fields(cls)]
            actual = json.loads(json.dumps(actual))
            original = contract['config_fields'][cls.__name__]
            self.assertEqual(actual[:len(original)], original)
        self.assertFalse(RetryTurnConfig().retain_candidate_pool)
        self.assertFalse(RetryTurnConfig().dispatch_all_parsed)
        self.assertEqual(RetryTurnConfig().probe_token_targets, ())
        self.assertEqual(TurnConfig().probe_after_tokens, 1)

    def test_original_default_requests_with_documented_usage_extension(self):
        client = VllmClient(model='fixture')
        contract = json.loads((ROOT / 'tests/v1_contract.json').read_text())['requests']
        for stream in (False, True):
            actual = client.request_body('fixture', max_tokens=16, stop=['<tool_call>'], logprobs=5, stream=stream)
            if stream:
                self.assertEqual(actual.pop('stream_options'), {'include_usage': True})
            self.assertEqual(actual, contract['completion_' + str(stream)])
            actual = client.chat_request_body([{'role': 'user', 'content': 'fixture'}], max_tokens=16, stream=stream)
            if stream:
                self.assertEqual(actual.pop('stream_options'), {'include_usage': True})
            self.assertEqual(actual, contract['chat_' + str(stream)])

    async def test_independent_baseline_and_exact_commit(self):
        for mode in ('baseline', 'spork'):
            row, client, executor, _ = await run_case(mode)
            self.assertEqual(executor.state, [2])
            if mode == 'baseline':
                self.assertEqual((client.probes, executor.speculative_calls), (0, 0))
                self.assertEqual(executor.canonical_calls, 1)
            else:
                self.assertTrue(row['spec_used'])
                self.assertEqual(executor.canonical_calls, 0)

    async def test_mismatch_discards_speculative_state(self):
        row, _, executor, _ = await run_case('spork', main_number=3)
        self.assertFalse(row['spec_used'])
        self.assertEqual(executor.state, [3])
        self.assertEqual(executor.canonical_calls, 1)

    async def test_native_baseline_preserves_multiple_calls(self):
        class MultiClient(FixtureClient):
            async def stream_chat(self, *args, **kwargs):
                row = await super().stream_chat(*args, **kwargs)
                row['tool_calls'].append(dict(id='second', type='function', function=dict(
                    name='lookup', arguments='{"number":3}')))
                return row
        client, executor, messages = MultiClient(), FixtureExecutor(), []
        row = await run_baseline_turn(None, client, Glm53ToolCallAdapter(tools=TOOLS),
                                     messages, executor, 0, TurnConfig())
        self.assertEqual(executor.state, [2, 3])
        self.assertEqual(len(row['baseline_tool_calls']), 2)
        self.assertEqual([m['tool_call_id'] for m in messages if m['role'] == 'tool'], ['fixture-call', 'second'])
        self.assertEqual(client.probes, 0)

    def test_invalid_arguments_remain_visible(self):
        messages = []
        _append_tool_messages(messages, '', 'error', tool_calls_structured=[dict(
            id='invalid', type='function', function=dict(name='lookup', arguments='{broken'))])
        self.assertEqual(json.loads(messages[0]['tool_calls'][0]['function']['arguments']), {'_raw': '{broken'})

    def test_strict_types_and_names(self):
        canonical = dict(name='lookup', arguments={'number': 2})
        self.assertTrue(_portfolio_calls_exact(canonical, canonical))
        self.assertFalse(_portfolio_calls_exact(dict(name='lookup', arguments={'number': '2'}), canonical))
        self.assertFalse(_portfolio_calls_exact(dict(name='other', arguments={'number': 2}), canonical))

    async def test_portfolio_selects_later_exact_match_or_falls_back(self):
        class TwoProbeClient(FixtureClient):
            async def stream_chat(self, session, messages, **kwargs):
                await kwargs['on_first_token'](.001, 'inspect')
                await kwargs['on_token']('inspect', 1)
                await asyncio.sleep(.04)
                await kwargs['on_token']('more', 16)
                await asyncio.sleep(.08)
                return dict(text='', reasoning='inspect', first_token_s=.001, wall_s=.12,
                    tool_calls=[dict(id='canonical', type='function', function=dict(
                        name='lookup', arguments=json.dumps({'number': self.main_number})))])

            async def complete_chat(self, *args, **kwargs):
                self.probes += 1
                return {'message': {'content': 'lookup<arg_key>number</arg_key><arg_value>' + str(self.probes + 1) + '</arg_value></tool_call>'}}, .001

        for main_number in (3, 4):
            client, executor = TwoProbeClient(main_number), FixtureExecutor()
            cfg = RetryTurnConfig(retain_candidate_pool=True, dispatch_all_parsed=True,
                max_retries=2, probe_token_targets=(1, 16), enable_thinking=True)
            row = await run_spork_retry_turn(None, client, Glm53ToolCallAdapter(tools=TOOLS),
                [{'role': 'user', 'content': 'lookup'}], executor, 0, cfg)
            self.assertEqual(client.probes, 2)
            self.assertEqual(executor.speculative_calls, 2)
            self.assertEqual(executor.state, [main_number])
            self.assertEqual(row['spec_used'], main_number == 3)
            self.assertEqual(executor.canonical_calls, int(main_number == 4))
            self.assertEqual(sum(p['selected'] for p in row['portfolio_candidates']), int(main_number == 3))

    def test_explicit_models_and_legacy_selection(self):
        from spork_core.qwen import QwenToolCallAdapter, Qwen35ToolCallAdapter
        self.assertIsInstance(make_adapter('auto', None, [], 'Qwen3.5'), Qwen35ToolCallAdapter)
        self.assertIsInstance(make_adapter('qwen35', None, []), QwenToolCallAdapter)
        for family in ('deepseek_v4', 'glm53'):
            client = configure_client(VllmClient(), family)
            self.assertTrue(make_adapter(family, None, []).uses_chat_api)
            self.assertTrue(client.include_reasoning)
            self.assertFalse(client.parallel_tool_calls)
        for invalid in ('0,16', '16,1', '1,1', ''):
            with self.assertRaises(ValueError):
                probe_targets(invalid)

    async def test_cli_depths_reach_retry_configuration(self):
        from unittest.mock import AsyncMock, patch
        import spork_unified_eval as cli
        args = cli.build_parser().parse_args(['--configs', 'portfolio', '--probe-token-targets', '8,32'])
        args.no_forced_conclusion = True
        runner = AsyncMock(return_value={'baseline_has_tool_call': False})
        with patch.object(cli, 'run_spork_retry_turn', runner):
            await cli.run_task(args, None, FixtureClient(), None, FixtureExecutor(), [], 'portfolio', TurnConfig())
        cfg = runner.call_args.args[-1]
        self.assertEqual(cfg.probe_token_targets, (8, 32))
        self.assertEqual(cfg.min_tokens_first_probe, 8)
        self.assertEqual(cfg.max_retries, 2)
        self.assertTrue(cfg.retain_candidate_pool and cfg.dispatch_all_parsed)


if __name__ == '__main__':
    unittest.main()
