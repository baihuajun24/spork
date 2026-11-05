"""Optional CPU checks against an explicitly provided official BFCL checkout."""
import ast
import importlib
import inspect
import json
import os
from pathlib import Path
import sys
import unittest
import uuid

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from experiments.bfcl.bfcl_spork_eval import CATEGORIES, WorldExecutor, load_entries, official_score
from experiments.bfcl import bfcl_spork_eval as harness


class HarnessTests(unittest.IsolatedAsyncioTestCase):
    def setup_case(self):
        args = harness.build_parser().parse_args(['--official-root', 'fixture', '--model-family', 'glm53'])
        entry = dict(id='multi_turn_base_0', functions=[], releases={}, involved_classes=[], initial_config={},
            schema_adjustments=[], question=[[dict(role='user', content='first')], [dict(role='user', content='second')]])
        return args, entry

    async def test_assistant_answers_are_retained_between_user_turns(self):
        from unittest.mock import patch
        args, entry = self.setup_case()
        histories = []
        async def runner(session, client, adapter, messages, executor, turn, config):
            histories.append(json.loads(json.dumps(messages)))
            return dict(baseline_has_tool_call=False, final_answer='answer', baseline_text='answer')
        with patch.object(harness, 'run_baseline_turn', runner), patch.object(harness, 'official_score', return_value={'valid': True}):
            row = await harness.run_one(args, entry, 'baseline', None)
        self.assertEqual(row['status'], 'complete')
        self.assertEqual(histories[1][-2], dict(role='assistant', content='answer'))
        self.assertTrue(row['quality']['valid'])

    async def test_truncated_tasks_are_unscored(self):
        from unittest.mock import AsyncMock, patch
        args, entry = self.setup_case()
        args.max_steps_per_user_turn = 1
        runner = AsyncMock(return_value=dict(baseline_has_tool_call=True,
            baseline_tool_name='lookup', baseline_tool_args={'number': 1}))
        with patch.object(harness, 'run_baseline_turn', runner), patch.object(harness, 'official_score') as scorer:
            row = await harness.run_one(args, entry, 'baseline', None)
        self.assertEqual(row['status'], 'incomplete')
        self.assertIsNone(row['quality'])
        scorer.assert_not_called()


@unittest.skipUnless(os.environ.get('SPORK_BFCL_OFFICIAL'), 'set SPORK_BFCL_OFFICIAL for official CPU checks')
class OfficialBFCLTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        source = Path(os.environ['SPORK_BFCL_OFFICIAL'])
        sys.path.insert(0, str(source))
        ids = json.loads((ROOT / 'experiments/configs/bfcl_holdout100.json').read_text())['task_ids']
        selected = [task for category in CATEGORIES for task in
                    [item for item in ids if item.rsplit('_', 1)[0] == category][:3]]
        cls.entries = load_entries(source, selected)

    def test_teacher_calls_match_official_results_and_state(self):
        utils = importlib.import_module('bfcl_eval.eval_checker.multi_turn_eval.multi_turn_utils')
        checker = importlib.import_module('bfcl_eval.eval_checker.multi_turn_eval.multi_turn_checker')
        for entry in self.entries:
            world = WorldExecutor(entry, 0)
            identity = 'public_cpu_' + uuid.uuid4().hex
            try:
                for user_calls in entry['ground_truth']:
                    for expression in user_calls:
                        call = ast.parse(expression, mode='eval').body
                        self.assertIsInstance(call, ast.Call)
                        self.assertIsInstance(call.func, ast.Name)
                        methods = [getattr(obj, call.func.id) for obj in world.instances.values()
                                   if inspect.ismethod(getattr(obj, call.func.id, None))]
                        bound = inspect.signature(methods[0]).bind(
                            *[ast.literal_eval(value) for value in call.args],
                            **{kw.arg: ast.literal_eval(kw.value) for kw in call.keywords})
                        actual, _ = world.execute(dict(name=call.func.id, arguments=bound.arguments))
                        expected, instances = utils.execute_multi_turn_func_call([expression],
                            entry['initial_config'], entry['involved_classes'], identity, entry['id'],
                            long_context='long_context' in entry['id'])
                        self.assertEqual(actual, expected[0], entry['id'] + ': ' + expression)
                        self.assertTrue(checker.state_checker(world.instances, instances)['valid'], entry['id'])
            finally:
                for key in list(vars(utils)):
                    if key.startswith(identity + '_'):
                        delattr(utils, key)

    def test_official_checker_accepts_complete_teacher_trace(self):
        for entry in self.entries:
            decoded = [[[call] for call in calls] for calls in entry['ground_truth']]
            self.assertTrue(official_score(entry, decoded)['valid'], entry['id'])


if __name__ == '__main__':
    unittest.main()
