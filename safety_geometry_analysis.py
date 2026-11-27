"""Cross-fitted safety/danger subspaces and multilingual margin decompositions."""
import argparse
from collections import defaultdict
import json
from pathlib import Path

import numpy as np

from safety_geometry import (direction_metrics, fit_subspace, principal_angles,
                             random_basis, state_metrics)


def transition(english, translated):
    names = {False: 'safe', True: 'danger'}
    return names[bool(english)] + '_to_' + names[bool(translated)]


def class_states(data, train_ids, dangerous, position, fit_groups):
    """One vector per request per outcome class, avoiding duplicate-source weighting."""
    selected = ((data['languages'] == 'en') & np.isin(data['ids'], train_ids)
                & (data['labels'] == dangerous) & np.isin(data['groups'], fit_groups))
    if position == 'answer':
        selected &= data.get('answer_valid', np.ones(len(selected), dtype=bool))
    ids = data['ids'][selected]
    states = data[position][selected].astype(np.float64)
    if not len(states):
        raise ValueError(f'no English training states for dangerous={dangerous}')
    return np.stack([states[ids == ident].mean(axis=0) for ident in np.unique(ids)])


def summarize(records, repetitions, seed):
    grouped = defaultdict(list)
    keys = ('kind', 'basis_class', 'rank', 'language', 'transition')
    for row in records:
        grouped[tuple(row[k] for k in keys)].append(row)
    result = []
    rng = np.random.default_rng(seed)
    for key, rows in sorted(grouped.items()):
        summary = dict(zip(keys, key)) | {'requests': len(rows)}
        metrics = sorted(k for k, v in rows[0].items()
                         if isinstance(v, (int, float)) and k not in ('rank', 'fold'))
        for metric in metrics:
            values = np.array([r[metric] for r in rows if r.get(metric) is not None], dtype=float)
            if not len(values):
                continue
            draws = values[rng.integers(len(values), size=(repetitions, len(values)))].mean(axis=1)
            summary[metric] = {'mean': float(values.mean()),
                               'request_bootstrap_95': np.quantile(draws, [.025, .975]).tolist()}
        result.append(summary)
    return result


def analyze(args):
    output = Path(args.output_dir)
    if output.exists():
        raise FileExistsError(output)
    with np.load(args.input, allow_pickle=False) as archive:
        data = {k: archive[k] for k in archive.files}
    metadata = json.loads(str(data['metadata']))
    if metadata['state_space'] != 'final_normalized_lm_head_input':
        raise ValueError('exact decomposition requires final normalized head-input states')
    target = data['groups'] == args.label_group
    ids = np.unique(data['ids'][target])
    if args.folds < 2 or len(ids) < args.folds:
        raise ValueError('need at least two folds and at least one request per fold')
    references = {}
    for i in np.flatnonzero(target):
        key = (str(data['ids'][i]), str(data['languages'][i]))
        if key in references:
            raise ValueError(f'duplicate evaluation record {key}')
        references[key] = i
    for ident in ids:
        if (str(ident), 'en') not in references:
            raise ValueError(f'missing English reference for {ident}')
    anchors = {(str(ident), str(language)): i for i, (ident, language) in
               enumerate(zip(data['anchor_ids'], data['anchor_languages']))}
    folds = np.array_split(np.random.default_rng(args.seed).permutation(ids), args.folds)
    all_ids = np.unique(data['ids'])
    records, spectra = [], []
    saved_bases = {}
    dimension = data['answer'].shape[1]
    for fold_index, heldout in enumerate(folds):
        train_ids = all_ids[~np.isin(all_ids, heldout)]
        rng = np.random.default_rng(args.seed + fold_index)
        classes = [class_states(data, train_ids, label, args.fit_position, args.fit_groups)
                   for label in (False, True)]
        unmatched_counts = [len(c) for c in classes]
        balanced = min(len(c) for c in classes)
        if balanced <= max(args.ranks):
            raise ValueError(f'fold {fold_index}: {balanced} matched class samples cannot support ranks {args.ranks}')
        # Equal sample counts make the empirical rank ceilings comparable.
        classes = [c[np.sort(rng.choice(len(c), balanced, replace=False))] for c in classes]
        fitted = []
        for label, states, unmatched in zip(('safe', 'danger'), classes, unmatched_counts):
            center, basis, stats = fit_subspace(states, max(args.ranks))
            spectra.append({'fold': fold_index, 'class': label, **stats,
                            'unmatched_class_samples': unmatched,
                            'training_ids': train_ids.tolist(), 'heldout_ids': heldout.tolist()})
            saved_bases[f'fold_{fold_index}_{label}_center'] = center
            saved_bases[f'fold_{fold_index}_{label}_basis'] = basis
            fitted.append((label, center, basis))
        for rank in args.ranks:
            angles = principal_angles(fitted[0][2][:, :rank], fitted[1][2][:, :rank])
            spectra.append({'fold': fold_index, 'rank': rank, 'kind': 'class_overlap', **angles})
            controls = [random_basis(dimension, rank, rng) for _ in range(args.random_directions)]
            for basis_class, center, complete_basis in fitted:
                basis = complete_basis[:, :rank]
                for key, other_index in references.items():
                    ident, language = key
                    if ident not in heldout or language == 'en':
                        continue
                    en_index = references[(ident, 'en')]
                    common = {'id': ident, 'language': language, 'fold': fold_index,
                              'basis_class': basis_class, 'rank': rank,
                              'transition': transition(data['labels'][en_index], data['labels'][other_index])}
                    # Prompt states isolate input translation. Answer states also change the response prefix.
                    for position in ('prompt', 'answer'):
                        valid = data.get('answer_valid', np.ones(len(data['ids']), dtype=bool))
                        if position == 'answer' and not (valid[en_index] and valid[other_index]):
                            continue
                        en = data[position][en_index].astype(float)
                        other = data[position][other_index].astype(float)
                        first = state_metrics(en, center, basis)
                        second = state_metrics(other, center, basis)
                        delta = other - en
                        norm = float(delta @ delta)
                        normal = delta - basis @ (basis.T @ delta)
                        outside_fraction = float(normal @ normal / norm) if norm > 1e-24 else None
                        null_distances = [state_metrics(other, center, b)['distance'] -
                                          state_metrics(en, center, b)['distance'] for b in controls]
                        records.append(common | {'kind': position,
                            'distance_en': first['distance'], 'distance_other': second['distance'],
                            'distance_change': second['distance'] - first['distance'],
                            'projection_energy_en': first['projection_energy'],
                            'projection_energy_other': second['projection_energy'],
                            'outside_fraction': outside_fraction,
                            'isotropic_outside_expectation': 1 - rank / dimension,
                            'random_distance_change_mean': float(np.mean(null_distances)),
                            'distance_change_excess': second['distance'] - first['distance'] - float(np.mean(null_distances))})
                    if key not in anchors or (ident, 'en') not in anchors:
                        continue
                    a, b = anchors[(ident, 'en')], anchors[key]
                    g = data['directions'][a]
                    if not np.array_equal(g, data['directions'][b]) or data['biases'][a] != data['biases'][b]:
                        raise ValueError('candidate head direction must be fixed across languages')
                    observed = direction_metrics(data['anchor'][a], data['anchor'][b], g,
                                                 float(data['biases'][a]), center, basis)
                    null = [direction_metrics(data['anchor'][a], data['anchor'][b], g,
                                              float(data['biases'][a]), center, control)
                            for control in controls]
                    observed.update({
                        'random_retention_mean': float(np.mean([r['direction_retention'] for r in null])),
                        'isotropic_retention_expectation': rank / dimension,
                        'retention_excess': observed['direction_retention'] - float(np.mean([r['direction_retention'] for r in null])),
                        'model_margin_en': float(data['model_margins'][a]),
                        'model_margin_other': float(data['model_margins'][b]),
                        'model_head_error_en': abs(observed['margin_en'] - float(data['model_margins'][a])),
                        'model_head_error_other': abs(observed['margin_other'] - float(data['model_margins'][b])),
                        'anchor_prefix_length': len(json.loads(str(data['anchor_prefixes'][a]))),
                    })
                    if observed['decomposition_error'] > 1e-8 * (1 + abs(observed['delta_margin'])):
                        raise ArithmeticError('margin decomposition identity failed')
                    records.append(common | {'kind': 'anchored_margin', **observed})
    output.mkdir(parents=True, exist_ok=False)
    for filename, rows in [('pairs.jsonl', records), ('spectra.jsonl', spectra)]:
        with (output / filename).open('w', encoding='utf-8') as stream:
            for row in rows:
                stream.write(json.dumps(row, allow_nan=False) + '\n')
    with (output / 'bases.npz').open('wb') as stream:
        np.savez_compressed(stream, **saved_bases)
    summary = {'settings': vars(args), 'extraction': metadata,
               'matched_training_samples': True, 'evaluation_group': args.label_group,
               'coverage': {'response_records': len(data['ids']), 'requests': len(ids),
                            'valid_answer_records': int(np.sum(data.get('answer_valid', np.ones(len(data['ids']), dtype=bool)))),
                            'anchored_requests': len(np.unique(data['anchor_ids'])),
                            'pair_measurements': len(records)},
               'results': summarize(records, args.bootstrap, args.seed),
               'anchor_selection': 'requests with opposite-label English source responses only'}
    (output / 'summary.json').write_text(json.dumps(summary, indent=2, allow_nan=False), encoding='utf-8')
    print(f'Saved {len(records)} held-out pair measurements to {output}')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input', required=True)
    p.add_argument('--output-dir', required=True, help='New directory')
    p.add_argument('--label-group', default='instruct', help='Actual generation source whose outcomes define flips')
    p.add_argument('--fit-groups', nargs='+', default=['base', 'instruct'])
    p.add_argument('--fit-position', choices=['prompt', 'answer'], default='answer')
    p.add_argument('--ranks', type=int, nargs='+', default=[1, 2, 4, 8])
    p.add_argument('--folds', type=int, default=5)
    p.add_argument('--random-directions', type=int, default=20)
    p.add_argument('--bootstrap', type=int, default=1000)
    p.add_argument('--seed', type=int, default=123)
    args = p.parse_args()
    if min(args.ranks) < 1 or len(set(args.ranks)) != len(args.ranks) or args.random_directions < 1 or args.bootstrap < 1:
        p.error('use unique positive ranks and positive random-directions and bootstrap values')
    analyze(args)


if __name__ == '__main__':
    main()
