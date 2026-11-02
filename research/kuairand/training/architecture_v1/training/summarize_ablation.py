"""Create a CSV and Markdown summary from completed ablation run folders."""
import argparse
import csv
import json
import math
from pathlib import Path


def rows(root):
    for manifest_path in sorted(Path(root).glob('**/run/run_manifest.json')):
        run = manifest_path.parent
        config = json.loads((run / 'config.json').read_text())
        manifest = json.loads(manifest_path.read_text())
        best_epoch = manifest['best_epoch']
        metrics = json.loads((run / f'evaluation-epoch-{best_epoch:03d}.json').read_text()) if best_epoch else {}
        yield {
            'group': config['ablation']['group'], 'variant': config['ablation']['variant'],
            'seed': config['experiment']['seed'], 'best_epoch': best_epoch,
            'recall_at_100': metrics.get('recall_at_100'), 'ndcg_at_100': metrics.get('ndcg_at_100'),
            'stopped_early': manifest['stopped_early'], 'run_id': manifest['run_id'],
            'run_path': str(run),
        }


def grouped(rows):
    buckets = {}
    for row in rows:
        key = (row['group'], row['variant'])
        buckets.setdefault(key, []).append(row)
    for (group, variant), values in sorted(buckets.items()):
        recalls = [row['recall_at_100'] for row in values if row['recall_at_100'] is not None]
        ndcgs = [row['ndcg_at_100'] for row in values if row['ndcg_at_100'] is not None]
        yield {'group': group, 'variant': variant, 'n_seeds': len(values),
               'recall_mean': sum(recalls) / len(recalls),
               'recall_std': math.sqrt(sum((x - sum(recalls) / len(recalls)) ** 2 for x in recalls) / len(recalls)) if len(recalls) > 1 else 0.0,
               'ndcg_mean': sum(ndcgs) / len(ndcgs),
               'ndcg_std': math.sqrt(sum((x - sum(ndcgs) / len(ndcgs)) ** 2 for x in ndcgs) / len(ndcgs)) if len(ndcgs) > 1 else 0.0}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runs', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = list(rows(args.runs))
    args.output.mkdir(parents=True, exist_ok=True)
    fields = ['group', 'variant', 'seed', 'best_epoch', 'recall_at_100', 'ndcg_at_100', 'stopped_early', 'run_id', 'run_path']
    with (args.output / 'ablation_summary.csv').open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader(); writer.writerows(result)
    aggregate = list(grouped(result))
    aggregate_fields = ['group', 'variant', 'n_seeds', 'recall_mean', 'recall_std', 'ndcg_mean', 'ndcg_std']
    with (args.output / 'ablation_seed_summary.csv').open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=aggregate_fields); writer.writeheader(); writer.writerows(aggregate)
    lines = ['# KuaiRand-1K Ablation Summary', '', '## Per-run results', '', '| Group | Variant | Seed | Best epoch | Recall@100 | NDCG@100 |', '| --- | --- | ---: | ---: | ---: | ---: |']
    for row in result:
        lines.append(f"| {row['group']} | {row['variant']} | {row['seed']} | {row['best_epoch']} | {row['recall_at_100'] or ''} | {row['ndcg_at_100'] or ''} |")
    lines.extend(['', '## Seed aggregate', '', '| Group | Variant | Seeds | Recall@100 mean ± std | NDCG@100 mean ± std |', '| --- | --- | ---: | ---: | ---: |'])
    for row in aggregate:
        lines.append(f"| {row['group']} | {row['variant']} | {row['n_seeds']} | {row['recall_mean']:.8f} ± {row['recall_std']:.8f} | {row['ndcg_mean']:.8f} ± {row['ndcg_std']:.8f} |")
    (args.output / 'ablation_summary.md').write_text('\n'.join(lines) + '\n')
    print(f'Wrote {len(result)} rows to {args.output}')


if __name__ == '__main__':
    main()
