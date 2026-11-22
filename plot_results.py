"""Plot paired generation and, when available, judge results from JSONL files."""
import argparse
import math
from pathlib import Path
from statistics import mean

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from reproduce import read_jsonl


def index_rows(path):
    rows = read_jsonl(path)
    if not rows:
        raise ValueError(f'{path}: no rows')
    result = {}
    for row in rows:
        key = row['id']
        if key in result:
            raise ValueError(f'{path}: duplicate ID {key}')
        if not isinstance(row.get('response'), str):
            raise ValueError(f'{path}: missing response for {key}')
        result[key] = row
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline', required=True, help='No-watermark JSONL')
    parser.add_argument('--watermarked', required=True, help='Watermarked JSONL')
    parser.add_argument('--output-dir', default='outputs', help='Directory for PNG files')
    args = parser.parse_args()
    baseline, watermarked = index_rows(args.baseline), index_rows(args.watermarked)
    if baseline.keys() != watermarked.keys():
        raise ValueError('Input files must contain the same sample IDs')
    for key in baseline:
        for field in ('dataset', 'goal', 'prompt', 'setting', 'model'):
            if baseline[key].get(field) != watermarked[key].get(field):
                raise ValueError(f'{key}: mismatched {field}')
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    left = [len(baseline[key]['response']) for key in baseline]
    right = [len(watermarked[key]['response']) for key in baseline]
    label = next(iter(watermarked.values())).get('watermark', 'Watermarked')
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].bar(['No watermark', label], [mean(left), mean(right)], color=['#6b9ac4', '#ed9b40'])
    axes[0].set_ylabel('Mean response length (characters)')
    axes[0].set_title(f'Response length (n={len(left)})')
    axes[1].scatter(left, right, alpha=.7)
    upper = max(left + right + [1])
    axes[1].plot([0, upper], [0, upper], '--', color='gray')
    axes[1].set_xlabel('No watermark (characters)')
    axes[1].set_ylabel(f'{label} (characters)')
    axes[1].set_title('Paired responses')
    fig.tight_layout()
    path = output / 'response_lengths.png'
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f'Saved {path}')

    for field, title, filename in [('wildguard', 'WildGuard ASR', 'wildguard_asr.png'),
                                    ('strongreject', 'StrongREJECT score', 'strongreject_score.png')]:
        present = [field in row for row in list(baseline.values()) + list(watermarked.values())]
        if not any(present):
            print(f'Skipped {title}: no scores in either file')
            continue
        if not all(present):
            raise ValueError(f'{field}: scores are missing for some rows or one input')
        scores = [[float(row[field]) for row in group.values()] for group in (baseline, watermarked)]
        if any(not math.isfinite(score) or not 0 <= score <= 1 for group in scores for score in group):
            raise ValueError(f'{field}: scores must be finite and between 0 and 1')
        values = [mean(group) for group in scores]
        fig, ax = plt.subplots(figsize=(6, 4))
        bars = ax.bar(['No watermark', label], values, color=['#6b9ac4', '#ed9b40'])
        ax.set_ylim(0, 1)
        ax.set_ylabel(title)
        ax.set_title(f'{title}: delta = {values[1] - values[0]:+.3f} (n={len(left)})')
        ax.bar_label(bars, labels=[f'{value:.3f}' for value in values], padding=2)
        fig.tight_layout()
        path = output / filename
        fig.savefig(path, dpi=150)
        plt.close(fig)
        print(f'Saved {path}; baseline={values[0]:.4f}, watermark={values[1]:.4f}, delta={values[1] - values[0]:+.4f}')


if __name__ == '__main__':
    main()
