#!/usr/bin/env python3
"""Export compact, text-free observations from the standalone BFCL runner.

Raw prompts, answers, tool outputs and local paths are deliberately omitted.
This checks recorded decisions, not hidden server activity or simulator state.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path


def summarize(folder: Path) -> dict:
    protocol = json.loads((folder / 'protocol.json').read_text())
    observations = []
    for config in protocol['configs']:
        for task_id in protocol['task_ids']:
            path = folder / config / (task_id + '.json')
            if not path.exists():
                observations.append(dict(task_id=task_id, config=config, status='missing', quality=None))
                continue
            row = json.loads(path.read_text())
            turns = row['turns']
            probes = [p for t in turns for p in t.get('probes', [])]
            tool_probes = [(t, p) for t in turns if t.get('baseline_has_tool_call')
                           for p in t.get('probes', [])]
            candidates = [c for t in turns for c in t.get('portfolio_candidates', [])]
            selected = [c for c in candidates if c['selected']]
            violations = []
            for i, turn in enumerate(turns):
                if config == 'baseline' and (turn.get('num_probes', 0) or turn.get('spec_dispatched')):
                    violations.append(dict(turn=i, check='baseline_has_speculative_work'))
                chosen = [c for c in turn.get('portfolio_candidates', []) if c['selected']]
                if len(chosen) > 1:
                    violations.append(dict(turn=i, check='multiple_selected_candidates'))
                for candidate in chosen:
                    call = candidate['call']
                    same = (call['name'] == turn.get('baseline_tool_name') and
                            json.dumps(call.get('arguments', {}), sort_keys=True) ==
                            json.dumps(turn.get('baseline_tool_args') or {}, sort_keys=True))
                    if not same:
                        violations.append(dict(turn=i, check='selected_call_differs_from_canonical'))
                if turn.get('spec_used') and not chosen:
                    violations.append(dict(turn=i, check='spec_used_without_selection'))
            main_known = [t['main_usage'] for t in turns if isinstance(t.get('main_usage'), dict)]
            probe_known = [p['usage'] for p in probes if isinstance(p.get('usage'), dict)]
            def measured_total(records, key):
                values = [item[key] for item in records if isinstance(item.get(key), (int, float))]
                return sum(values) if values else None
            observations.append(dict(task_id=task_id, config=config, status=row['status'],
                failed_turn_trace_missing=row['status'] == 'error',
                counts_scope='Completed recorded turns only; failed current-turn work is unknown',
                quality=row['quality']['valid'] if row.get('quality') is not None else None,
                quality_components=({name: value['valid'] for name, value in row['quality'].items()
                                     if isinstance(value, dict) and 'valid' in value}
                                    if row.get('quality') is not None else None),
                quality_error_types=({name: value['error_type'] for name, value in row['quality'].items()
                                      if isinstance(value, dict) and value.get('error_type')}
                                     if row.get('quality') is not None else None),
                score_error=bool(row.get('score_error')), error_type=(row.get('error') or '').split(':', 1)[0] or None,
                wall_s=row['wall_s'], turns=len(turns), tool_turns=sum(bool(t.get('baseline_has_tool_call')) for t in turns),
                probes=len(probes), parsed_probes=sum(bool(p.get('probe_parse_ok')) for p in probes),
                probes_on_tool_turns=len(tool_probes),
                probe_name_matches_on_tool_turns=sum(
                    p.get('probe_tool_name') == t.get('baseline_tool_name') for t, p in tool_probes),
                probe_full_call_matches_on_tool_turns=sum(
                    p.get('probe_tool_name') == t.get('baseline_tool_name')
                    and json.dumps(p.get('probe_tool_args'), sort_keys=True) ==
                    json.dumps(t.get('baseline_tool_args'), sort_keys=True) for t, p in tool_probes),
                probe_dispatch_depths=sorted({p['tokens_at_dispatch'] for p in probes
                                             if isinstance(p.get('tokens_at_dispatch'), int)}),
                probe_response_sources=dict(Counter(p.get('probe_response_source') or 'unknown' for p in probes)),
                native_main_call_count=sum(t.get('main_structured_tool_call_count', 0) for t in turns),
                main_reasoning_observed_turns=sum(bool(t.get('main_reasoning')) for t in turns),
                candidates=len(candidates), selected_candidates=len(selected),
                selected_exact=sum(bool(c['exact_match']) for c in selected),
                speculative_commits=sum(bool(t.get('spec_used')) for t in turns),
                selected_worker_errors=sum(bool(c.get('worker_error')) for c in selected),
                unselected_candidates=len(candidates) - len(selected),
                canonical_fallbacks=(sum(bool(t.get('baseline_has_tool_call')) and not t.get('spec_used', False)
                                         for t in turns) if config == 'portfolio' else None),
                main_usage_known=len(main_known), main_usage_missing=len(turns) - len(main_known),
                probe_usage_known=len(probe_known), probe_usage_missing=len(probes) - len(probe_known),
                probe_errors=sum(bool(p.get('probe_error')) for p in probes),
                inferred_main_budget_hits=sum(
                    isinstance(t.get('main_usage'), dict)
                    and isinstance(t['main_usage'].get('completion_tokens'), int)
                    and isinstance(t.get('main_max_tokens'), int)
                    and t['main_usage']['completion_tokens'] >= t['main_max_tokens']
                    for t in turns),
                measured_main_completion_tokens=measured_total(main_known, 'completion_tokens'),
                measured_probe_completion_tokens=measured_total(probe_known, 'completion_tokens'),
                decision_violations=violations,
                raw_sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
    public_settings = ('n', 'model_name', 'model_family', 'configs', 'probe_token_targets',
                       'probe_max_tokens', 'max_tokens', 'context_window_tokens',
                       'max_steps_per_user_turn', 'tool_latency_s', 'seed', 'thinking')
    return dict(schema_version=1, code_commit=protocol['code_commit'], code_dirty=protocol['working_tree_dirty'],
        task_ids=protocol['task_ids'], task_ids_sha256=protocol['task_ids_sha256'],
        settings={name: protocol['settings'][name] for name in public_settings if name in protocol['settings']},
        source_sha256=protocol['source_sha256'],
        scoring=protocol['scoring'], measurement=protocol['measurement'],
        trigger_units=protocol['trigger_units'], observations=observations,
        audit_scope='Recorded baseline isolation and selected-call matching; does not prove simulator state isolation or latency gains')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('folder', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error('a fresh --output file is required')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summarize(args.folder), indent=2) + '\n')


if __name__ == '__main__':
    main()
