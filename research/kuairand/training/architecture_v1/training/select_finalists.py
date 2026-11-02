"""Select validation winners from screening runs; sealed test is never read."""
import argparse
import json
from pathlib import Path

from summarize_ablation import rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runs', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    screened = [row for row in rows(args.runs) if row['seed'] == 42 and row['group'] != 'control']
    winners = {}
    control = [row for row in rows(args.runs) if row['seed'] == 42 and row['group'] == 'control' and row['variant'] == 'control']
    if not control:
        raise RuntimeError('No completed matched control screening run')
    for group in ('architecture', 'hyperparameters', 'sampling'):
        candidates = [row for row in screened if row['group'] == group] + control
        if not candidates:
            raise RuntimeError(f'No completed screening run for {group}')
        winner = max(candidates, key=lambda row: (row['recall_at_100'], row['ndcg_at_100']))
        winners[group] = {'variant': winner['variant'], 'screening_recall_at_100': winner['recall_at_100'], 'screening_ndcg_at_100': winner['ndcg_at_100']}
    payload = {'selection_split': 'validation', 'test_read': False, 'seeds_for_confirmation': [42, 2026, 3407], 'winners': winners}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + '\n')
    print(json.dumps(payload, indent=2))


if __name__ == '__main__':
    main()
