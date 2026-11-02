"""Write a data-grounded Markdown report and SVG charts from completed validation runs."""
import argparse
import csv
import html
from pathlib import Path


LABELS = {'architecture': 'Architecture', 'hyperparameters': 'Hyperparameters', 'sampling': 'Sampling correction', 'control': 'Matched control'}


def read_csv(path):
    with Path(path).open() as handle:
        return list(csv.DictReader(handle))


def chart(rows, output):
    values = [float(row['recall_mean']) for row in rows]
    maximum = max(values) if values else 1.0
    width, bar, gap, top = 1000, 42, 30, 80
    height = top + len(rows) * (bar + gap) + 40
    elements = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">', '<style>text{font-family:Arial,sans-serif;font-size:14px}.title{font-size:20px;font-weight:bold}</style>', '<text x="30" y="35" class="title">KuaiRand-1K validation Recall@100</text>']
    for index, row in enumerate(rows):
        y = top + index * (bar + gap)
        value = float(row['recall_mean']); std = float(row['recall_std'])
        scaled = 700 * value / maximum if maximum else 0
        label = f"{row['group']} / {row['variant']}"
        elements.extend([f'<text x="30" y="{y + 26}">{html.escape(label)}</text>', f'<rect x="270" y="{y}" width="{scaled}" height="{bar}" fill="#4f46e5"/>', f'<text x="{280 + scaled}" y="{y + 26}">{value:.6%} ± {std:.6%}</text>'])
    elements.append('</svg>')
    Path(output).write_text('\n'.join(elements))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--summary-dir', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(); args.output.mkdir(parents=True, exist_ok=True)
    rows = read_csv(args.summary_dir / 'ablation_seed_summary.csv')
    if not rows:
        raise RuntimeError('No completed runs to report')
    chart(rows, args.output / 'recall_at_100.svg')
    lines = ['# KuaiRand-1K Ablation Study', '', '## Scope', '', 'This report is generated from completed validation runs only. The sealed test split was not read for model selection or this report.', '', '## Results', '', '| Group | Variant | Seeds | Recall@100 (mean ± std) | NDCG@100 (mean ± std) |', '| --- | --- | ---: | ---: | ---: |']
    for row in rows:
        lines.append(f"| {LABELS.get(row['group'], row['group'])} | {row['variant']} | {row['n_seeds']} | {float(row['recall_mean']):.6%} ± {float(row['recall_std']):.6%} | {float(row['ndcg_mean']):.6%} ± {float(row['ndcg_std']):.6%} |")
    lines.extend(['', '## Interpretation', ''])
    for group in ('architecture', 'hyperparameters', 'sampling'):
        # The shared matched control is the valid baseline for every group;
        # include it so the report never implies that a changed variant won
        # merely because it was the best non-control variant.
        candidates = [row for row in rows if row['group'] in (group, 'control')]
        if candidates:
            best = max(candidates, key=lambda row: (float(row['recall_mean']), float(row['ndcg_mean'])))
            lines.append(f"- Validation winner for **{LABELS[group]}**: `{best['variant']}` with Recall@100 {float(best['recall_mean']):.6%} ± {float(best['recall_std']):.6%}. This is an observed offline validation comparison, not a causal or online-effect claim.")
    lines.extend(['', '## Limits', '', '- All comparisons use the frozen non-random-exposure training and validation protocol; they do not establish online A/B-test impact.', '- The random-exposure evaluation and the sealed test are outside this model-selection report.', '- Single-seed screening determines finalists; only the selected key comparisons receive three-seed confirmation.', '- Exact full-catalog validation uses the fixed `upload_next_day` eligibility rule; uncovered targets remain misses.', ''])
    (args.output / 'ablation_report.md').write_text('\n'.join(lines))
    print(args.output / 'ablation_report.md')


if __name__ == '__main__':
    main()
