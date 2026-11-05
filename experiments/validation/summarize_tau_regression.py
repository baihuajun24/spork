#!/usr/bin/env python3
"""Export text-free observations from the unified runner's tau2 smoke checks.

ACTION matching is a guardrail, not the full interactive tau2 evaluation.
Errors/timeouts remain unscored even when the legacy runner assigns a reward.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def summarize(folder: Path, manifest: Path, code_commit: str) -> dict:
    tasks = json.loads(manifest.read_text())['tasks']
    observations = []
    for config in ('baseline', 'd1_d2'):
        for task in tasks:
            path = folder / config / f"{task['domain']}_{task['task_id']}.json"
            base = dict(task, config=config)
            if not path.exists():
                observations.append(dict(base, status='missing', action_reward=None))
                continue
            row = json.loads(path.read_text())
            if (row['task_id'], row['domain'], row['config']) != (
                    task['task_id'], task['domain'], config):
                raise ValueError(f'task identity mismatch: {path.name}')
            turns = row['turns']
            errors = [turn['error'] for turn in turns if turn.get('error')]
            scoring = row.get('tau2_success')
            eligible = not errors and scoring is not None
            observations.append(dict(base,
                status=('timeout' if any('timeout' in str(e).lower() for e in errors)
                        else 'error') if errors else 'returned',
                action_reward=scoring.get('reward') if eligible else None,
                actions_matched=scoring.get('n_matched') if eligible else None,
                actions_total=scoring.get('n_golden') if eligible else None,
                legacy_score_excluded=bool(errors and scoring is not None),
                error_count=len(errors), wall_s=row['wall_s'],
                turns=len(turns) if not errors else None,
                turn_trace_available=not errors,
                last_turn_has_tool_call=(bool(turns[-1].get('baseline_has_tool_call'))
                                         if turns and not errors else None),
                recorded_probes=(sum(turn.get('num_probes', 0) for turn in turns)
                                 if not errors else None),
                speculative_commits=(sum(bool(turn.get('spec_used')) for turn in turns)
                                     if not errors else None),
                baseline_recorded_isolation_violation=(any(
                    turn.get('num_probes') or turn.get('spec_dispatched')
                    for turn in turns) if config == 'baseline' and not errors else None),
                raw_sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
    summary_path = folder / 'summary.json'
    summary = json.loads(summary_path.read_text()) if summary_path.exists() else None
    return dict(schema_version=1, code_commit=code_commit,
        manifest_sha256=hashlib.sha256(manifest.read_bytes()).hexdigest(),
        task_ids=tasks, run=summary.get('run') if summary else None,
        raw_summary_sha256=(hashlib.sha256(summary_path.read_bytes()).hexdigest()
                            if summary_path.exists() else None),
        observations=observations,
        scoring='Static instructions + official Action.compare_with_tool_call only; not full interactive tau2',
        measurement='Task wall time excludes server startup; errors/timeouts excluded from quality and latency claims',
        sampling='Same ten retail tasks, once per configuration and code version; forty observations are not forty independent tasks')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('folder', type=Path)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--code-commit', required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error('a fresh --output file is required')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summarize(args.folder, args.manifest, args.code_commit), indent=2) + '\n')


if __name__ == '__main__':
    main()
