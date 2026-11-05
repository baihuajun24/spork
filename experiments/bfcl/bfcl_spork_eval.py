#!/usr/bin/env python3
"""Small, serial BFCL v3 multi-turn example using an explicit official checkout.

No services are started. Each task/config owns a fresh tool world. Task text,
model responses and execution records are written only to the requested output.
"""
import argparse
import asyncio
import copy
from dataclasses import replace
import hashlib
import importlib
import inspect
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import time
import uuid

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import aiohttp
from spork_core.executors import SpeculativeToolExecution
from spork_core.model_setup import configure_client, make_adapter, probe_targets
from spork_core.turn_runner import TurnConfig, RetryTurnConfig, run_baseline_turn, run_spork_retry_turn
from spork_core.vllm_client import VllmClient

CATEGORIES = ('multi_turn_base', 'multi_turn_miss_func', 'multi_turn_miss_param', 'multi_turn_long_context')
CLASS_FILES = dict(GorillaFileSystem='gorilla_file_system', MathAPI='math_api',
    MessageAPI='message_api', TwitterAPI='posting_api', TicketAPI='ticket_api',
    TradingBot='trading_bot', TravelAPI='travel_booking', VehicleControlAPI='vehicle_control')
TOOL_LOCK = threading.RLock()


def read_lines(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def normalize_schema(schema):
    mapping = {'dict': 'object', 'list': 'array', 'float': 'number', 'tuple': 'array'}
    if isinstance(schema, dict):
        if isinstance(schema.get('type'), str) and schema['type'] in mapping:
            schema['type'] = mapping[schema['type']]
        for value in schema.values():
            if isinstance(value, (dict, list)):
                normalize_schema(value)
    elif isinstance(schema, list):
        for value in schema:
            normalize_schema(value)


def load_entries(source, task_ids):
    data = source / 'bfcl_eval/data'
    questions, answers = {}, {}
    for category in CATEGORIES:
        questions.update({row['id']: row for row in read_lines(data / f'BFCL_v3_{category}.json')})
        answers.update({row['id']: row['ground_truth'] for row in
                        read_lines(data / 'possible_answer' / f'BFCL_v3_{category}.json')})
    entries = []
    for task_id in task_ids:
        if task_id not in questions or task_id not in answers:
            raise ValueError(f'official task or answer missing: {task_id}')
        entry = copy.deepcopy(questions[task_id])
        functions = []
        for cls in entry['involved_classes']:
            if cls not in CLASS_FILES:
                raise ValueError(f'unsupported official simulator: {cls}')
            functions.extend(read_lines(data / 'multi_turn_func_doc' / (CLASS_FILES[cls] + '.json')))
        releases = {}
        for turn, names in entry.get('missed_function', {}).items():
            releases[str(turn)] = [fn for fn in functions if fn['name'] in names]
            if len(releases[str(turn)]) != len(names):
                raise ValueError('held-out function not uniquely identified')
            functions = [fn for fn in functions if fn['name'] not in names]
        normalize_schema(functions)
        normalize_schema(releases)
        # This frozen BFCL schema advertises an optional argument absent from
        # its simulator. Record the adjustment rather than silently changing it.
        patches = []
        for fn in functions + [fn for group in releases.values() for fn in group]:
            params = fn.get('parameters', {})
            if fn['name'] == 'delete_message' and 'MessageAPI' in entry['involved_classes']:
                module = importlib.import_module(
                    'bfcl_eval.eval_checker.multi_turn_eval.func_source_code.message_api')
                if ('message_id' not in inspect.signature(module.MessageAPI.delete_message).parameters
                        and 'message_id' in params.get('properties', {})
                        and 'message_id' not in params.get('required', [])):
                    del params['properties']['message_id']
                    patches.append('delete_message: removed unsupported optional message_id')
        entry.update(functions=functions, releases=releases, ground_truth=answers[task_id],
                     schema_adjustments=patches)
        entries.append(entry)
    return entries


class WorldExecutor:
    """Invoke only public simulator methods; speculate on isolated snapshots."""
    last_real_wall_s = 0.0
    last_floor_sleep_s = 0.0

    def __init__(self, entry, floor_s):
        self.floor_s = floor_s
        self.instances = {}
        for name in entry['involved_classes']:
            module = importlib.import_module('bfcl_eval.eval_checker.multi_turn_eval.func_source_code.' + CLASS_FILES[name])
            obj = getattr(module, name)()
            if name != 'MathAPI':
                obj._load_scenario(copy.deepcopy(entry['initial_config'].get(name, {})),
                                   long_context='long_context' in entry['id'])
            self.instances[name] = obj

    def invoke(self, instances, call):
        name = call['name']
        methods = [getattr(obj, name) for obj in instances.values()
                   if not name.startswith('_') and inspect.ismethod(getattr(obj, name, None))]
        if len(methods) != 1:
            return 'Error during execution: Unknown or ambiguous function: ' + name
        try:
            with TOOL_LOCK:
                bound = inspect.signature(methods[0]).bind(**copy.deepcopy(call.get('arguments') or {}))
                bound.apply_defaults()
                value = methods[0](**copy.deepcopy(bound.arguments))
            if isinstance(value, str):
                return value
            if isinstance(value, dict):
                try:
                    return json.dumps(value)
                except (TypeError, ValueError):
                    pass
            return str(value)
        except Exception as exc:
            return 'Error during execution: ' + str(exc)

    def execute_on(self, instances, call):
        started = time.perf_counter()
        text = self.invoke(instances, call)
        real = time.perf_counter() - started
        time.sleep(max(0, self.floor_s - real))
        wall = time.perf_counter() - started
        return text, wall, real

    def execute(self, call):
        text, wall, real = self.execute_on(self.instances, call)
        self.last_real_wall_s, self.last_floor_sleep_s = real, max(0, wall - real)
        return text, wall

    def execute_many(self, calls):
        return [self.execute(call) for call in calls]

    def execute_speculative(self, call):
        snapshot = copy.deepcopy(self.instances)
        text, wall, real = self.execute_on(snapshot, call)
        return SpeculativeToolExecution(text, wall, real, max(0, wall - real), state=snapshot)

    def commit_speculative(self, execution):
        self.instances = execution.state
        self.last_real_wall_s = execution.real_wall_s
        self.last_floor_sleep_s = execution.artificial_sleep_s


class SingleCallClient(VllmClient):
    async def stream_chat(self, *args, **kwargs):
        row = await super().stream_chat(*args, **kwargs)
        if len(row.get('tool_calls') or []) > 1:
            raise ValueError('portfolio example supports one canonical call per decision; refusing truncation')
        return row


def expression(call):
    name, arguments = call['name'], call.get('arguments') or {}
    if not name.isidentifier() or name.startswith('_') or any(
            not key.isidentifier() or key.startswith('_') for key in arguments):
        raise ValueError('call cannot be represented safely for the official checker')
    return name + '(' + ','.join(key + '=' + repr(value) for key, value in arguments.items()) + ')'


def official_score(entry, decoded):
    from bfcl_eval.eval_checker.multi_turn_eval.multi_turn_checker import (
        multi_turn_checker, multi_turn_irrelevance_checker)
    # The official evaluator caches simulator instances under model names.
    # A fresh identity prevents state leaking between methods or repetitions.
    identity = 'spork_' + uuid.uuid4().hex
    try:
        state = multi_turn_checker(decoded, entry['ground_truth'], entry,
                                   entry['id'].rsplit('_', 1)[0], identity)
        relevance = multi_turn_irrelevance_checker(decoded, entry['ground_truth'])
        return dict(valid=bool(state['valid'] and relevance['valid']), state=state, relevance=relevance)
    finally:
        from bfcl_eval.eval_checker.multi_turn_eval import multi_turn_utils
        for key in list(vars(multi_turn_utils)):
            if key.startswith(identity + '_'):
                delattr(multi_turn_utils, key)


async def run_one(args, entry, config, tokenizer):
    executor = WorldExecutor(entry, args.tool_latency_s)
    tools = [dict(type='function', function=copy.deepcopy(fn)) for fn in entry['functions']]
    adapter = make_adapter(args.model_family, tokenizer, tools, args.model_name)
    cls = VllmClient if config == 'baseline' else SingleCallClient
    client = configure_client(cls(base_url=args.model_url, model=args.model_name), args.model_family,
                              enable_thinking=args.thinking)
    messages, turns, decoded = [], [], []
    cfg = TurnConfig(enable_thinking=args.thinking, baseline_max_tokens=args.max_tokens,
                     context_window_tokens=args.context_window_tokens, seed=args.seed)
    retry = RetryTurnConfig(enable_thinking=args.thinking, baseline_max_tokens=args.max_tokens,
        context_window_tokens=args.context_window_tokens, seed=args.seed,
        max_retries=len(args.probe_token_targets), probe_token_targets=args.probe_token_targets,
        min_tokens_first_probe=args.probe_token_targets[0],
        retain_candidate_pool=True, dispatch_all_parsed=True, probe_max_tokens=args.probe_max_tokens)
    row = dict(task_id=entry['id'], config=config, status='running', turns=turns,
               schema_adjustments=entry['schema_adjustments'], force_quit=False,
               quality=None, score_error=None)
    started = time.perf_counter()
    try:
        async with aiohttp.ClientSession(trust_env=False) as session:
            for user_index, user_messages in enumerate(entry['question']):
                added = entry['releases'].get(str(user_index), [])
                if added:
                    tools.extend(dict(type='function', function=copy.deepcopy(fn)) for fn in added)
                    messages.append(dict(role='user', content='I have updated some more functions you can choose from. What about now?'))
                else:
                    messages.extend(copy.deepcopy(user_messages))
                adapter.tools = tools
                steps = []
                decoded.append(steps)
                for _ in range(args.max_steps_per_user_turn):
                    effective = cfg if config == 'baseline' else retry
                    if not getattr(adapter, 'uses_chat_api', False):
                        prompt = adapter.render_main_prompt(messages, enable_thinking=args.thinking)
                        budget = max(1, args.context_window_tokens - len(tokenizer.encode(prompt, add_special_tokens=False)))
                        effective = replace(effective, baseline_max_tokens=min(args.max_tokens, budget))
                    runner = run_baseline_turn if config == 'baseline' else run_spork_retry_turn
                    turn = await runner(session, client, adapter, messages, executor, len(turns), effective)
                    turn['user_turn_index'] = user_index
                    turns.append(turn)
                    calls = turn.get('baseline_tool_calls')
                    if calls is None:
                        calls = [dict(name=turn['baseline_tool_name'], arguments=turn['baseline_tool_args'])] if turn['baseline_has_tool_call'] else []
                    steps.append([expression(call) for call in calls])
                    if not turn['baseline_has_tool_call']:
                        answer = dict(role='assistant', content=turn.get('final_answer', turn.get('baseline_text', '')))
                        if turn.get('main_reasoning'):
                            answer[getattr(adapter, 'reasoning_message_field', 'reasoning_content')] = turn['main_reasoning']
                        messages.append(answer)
                        break
                else:
                    row['force_quit'] = True
                    break
        row['status'] = 'incomplete' if row['force_quit'] else 'complete'
    except Exception as exc:
        row.update(status='error', error=f'{type(exc).__name__}: {exc}')
    row['wall_s'] = time.perf_counter() - started
    row['decoded_calls'] = decoded
    if row['status'] == 'complete':
        try:
            row['quality'] = official_score(entry, decoded)
        except Exception as exc:
            row['score_error'] = f'{type(exc).__name__}: {exc}'
    return row


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--official-root', type=Path, required=True)
    parser.add_argument('--task-ids-file', type=Path, default=ROOT / 'experiments/configs/bfcl_holdout100.json')
    parser.add_argument('--n', type=int, default=10)
    parser.add_argument('--model-url', default='http://localhost:8000/v1')
    parser.add_argument('--model-name', default='qwen3-32b')
    parser.add_argument('--model-family', choices=['qwen', 'qwen35', 'deepseek_v4', 'glm53'], default='qwen')
    parser.add_argument('--tokenizer-path')
    parser.add_argument('--configs', default='baseline,portfolio')
    parser.add_argument('--probe-token-targets', type=probe_targets, default=(1, 16))
    parser.add_argument('--probe-max-tokens', type=int, default=256)
    parser.add_argument('--max-tokens', type=int, default=8192)
    parser.add_argument('--context-window-tokens', type=int, default=32768)
    parser.add_argument('--max-steps-per-user-turn', type=int, default=21)
    parser.add_argument('--tool-latency-s', type=float, default=2)
    parser.add_argument('--seed', type=int, default=1234)
    parser.add_argument('--thinking', action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument('--describe', action='store_true', help='CPU-only source/task preflight')
    parser.add_argument('--output', type=Path)
    return parser


async def evaluate(args, entries, configs, protocol):
    tokenizer = None
    if args.model_family in ('qwen', 'qwen35'):
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path, trust_remote_code=True)
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / 'protocol.json').write_text(json.dumps(protocol, indent=2) + '\n')
    observations = []
    for entry in entries:
        for config in configs:
            row = await run_one(args, entry, config, tokenizer)
            folder = args.output / config
            folder.mkdir(exist_ok=True)
            (folder / (entry['id'] + '.json')).write_text(json.dumps(row, indent=2, default=str) + '\n')
            observations.append(row)
            print(entry['id'], config, row['status'], flush=True)
    summary = {}
    for config in configs:
        rows = [row for row in observations if row['config'] == config]
        scored = [row for row in rows if row['quality'] is not None]
        summary[config] = dict(task_observations=len(rows), unique_tasks=len({row['task_id'] for row in rows}),
            complete=sum(row['status'] == 'complete' for row in rows),
            errors=sum(row['status'] == 'error' for row in rows),
            incomplete=sum(row['status'] == 'incomplete' for row in rows),
            scored=len(scored), unscored=len(rows) - len(scored),
            correct=sum(row['quality']['valid'] for row in scored),
            quality_rate=sum(row['quality']['valid'] for row in scored) / len(scored) if scored else None)
    (args.output / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps(summary, indent=2))


def main():
    parser = build_parser()
    args = parser.parse_args()
    configs = args.configs.split(',')
    if len(set(configs)) != len(configs) or not configs or set(configs) - {'baseline', 'portfolio'}:
        parser.error('--configs accepts baseline,portfolio, each at most once')
    if min(args.n, args.max_tokens, args.probe_max_tokens, args.context_window_tokens, args.max_steps_per_user_turn) < 1 or args.tool_latency_s < 0:
        parser.error('counts must be positive and tool latency nonnegative')
    if not args.describe and (args.output is None or args.output.exists()):
        parser.error('a fresh --output directory is required')
    if not args.describe and args.model_family in ('qwen', 'qwen35') and not args.tokenizer_path:
        parser.error('Qwen requires an explicit --tokenizer-path matching the tool grammar')
    source = args.official_root.resolve()
    sys.path.insert(0, str(source))
    ids = json.loads(args.task_ids_file.read_text())
    ids = ids['task_ids'] if isinstance(ids, dict) else ids
    if len(ids) != len(set(ids)) or args.n > len(ids):
        parser.error('task IDs must be unique and cover --n')
    entries = load_entries(source, ids[:args.n])
    relevant = set()
    for entry in entries:
        category = entry['id'].rsplit('_', 1)[0]
        relevant.update([source / f'bfcl_eval/data/BFCL_v3_{category}.json',
                         source / f'bfcl_eval/data/possible_answer/BFCL_v3_{category}.json'])
        for cls in entry['involved_classes']:
            relevant.update([source / f'bfcl_eval/data/multi_turn_func_doc/{CLASS_FILES[cls]}.json',
                source / f'bfcl_eval/eval_checker/multi_turn_eval/func_source_code/{CLASS_FILES[cls]}.py'])
    relevant.update((source / 'bfcl_eval/eval_checker/multi_turn_eval').glob('*.py'))
    protocol = dict(task_ids=[entry['id'] for entry in entries], configs=configs,
        source_sha256={str(p.relative_to(source)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(relevant)},
        task_ids_sha256=hashlib.sha256(args.task_ids_file.read_bytes()).hexdigest(),
        settings={key: value for key, value in vars(args).items() if key not in ('official_root', 'task_ids_file', 'output')},
        code_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
        working_tree_dirty=bool(subprocess.check_output(['git', 'status', '--porcelain', '--untracked-files=no'], cwd=ROOT, text=True)),
        hardware=os.environ.get('SPORK_HARDWARE_DESCRIPTION'),
        scoring='Unmodified official multi_turn_checker AND multi_turn_irrelevance_checker; errors/incomplete unscored',
        measurement='Task wall time includes model and tool execution, excludes official scoring; serial task/config order',
        trigger_units='Output-stream deltas, not exact tokenizer positions')
    if args.describe:
        print(json.dumps(protocol, indent=2))
        return
    asyncio.run(evaluate(args, entries, configs, protocol))


if __name__ == '__main__':
    main()
