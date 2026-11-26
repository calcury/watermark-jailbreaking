"""Fit a cross-validated paired activation direction and export projections."""
import argparse
import json
from pathlib import Path

import numpy as np

from reproduce import read_jsonl


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input', required=True, help='boundary_experiment output')
    p.add_argument('--labels', required=True, help='JSONL with id, clean and watermarked binary labels')
    p.add_argument('--output', required=True)
    p.add_argument('--layer', type=int, default=-1)
    p.add_argument('--offset', type=int, default=1)
    p.add_argument('--folds', type=int, default=5)
    p.add_argument('--seed', type=int, default=123)
    args = p.parse_args()
    labels = {r['id']: r for r in read_jsonl(args.labels)}
    records = []
    for row in read_jsonl(args.input):
        if row['id'] not in labels:
            continue
        features = {b['trajectory']: b['features'][str(args.layer)] for b in row['branches']
                    if b['offset'] == args.offset and b['features'] is not None}
        if set(features) != {'clean', 'watermarked'}:
            continue
        label = labels[row['id']]
        if label['clean'] not in (0, 1) or label['watermarked'] not in (0, 1):
            raise ValueError('labels must be binary')
        records.append((row['id'], np.asarray(features['clean']),
                        np.asarray(features['watermarked']), label))
    if args.folds < 2 or len(records) < args.folds:
        raise ValueError('need at least one request per fold and at least two folds')
    rng = np.random.default_rng(args.seed)
    order = rng.permutation(len(records))
    folds = np.array_split(order, args.folds)
    output = Path(args.output)
    if output.exists():
        raise FileExistsError(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('x', encoding='utf-8') as f:
        for fold, test in enumerate(folds):
            train = np.concatenate([part for i, part in enumerate(folds) if i != fold])
            differences = [records[i][2] - records[i][1] for i in train
                           if records[i][3]['clean'] == 0 and records[i][3]['watermarked'] == 1]
            if not differences:
                raise ValueError(f'fold {fold} has no training safe-to-unsafe pairs')
            direction = np.mean(differences, axis=0)
            norm = np.linalg.norm(direction)
            if norm == 0:
                raise ValueError(f'fold {fold} has zero direction')
            direction /= norm
            for i in test:
                ident, clean, marked, label = records[i]
                item = {'id': ident, 'fold': fold, 'clean_label': label['clean'],
                        'watermarked_label': label['watermarked'],
                        'clean_projection': float(clean @ direction),
                        'watermarked_projection': float(marked @ direction),
                        'difference_projection': float((marked - clean) @ direction),
                        'training_pairs': len(differences)}
                f.write(json.dumps(item, allow_nan=False) + '\n')


if __name__ == '__main__':
    main()
