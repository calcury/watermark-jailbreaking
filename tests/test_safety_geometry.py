import argparse
from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from safety_geometry import (direction_metrics, fit_subspace, principal_angles,
                             project, random_basis, state_metrics)
from safety_geometry_analysis import analyze
from safety_geometry_experiment import english_anchors, prepare


class GeometryTests(unittest.TestCase):
    def test_exact_margin_decomposition_and_affine_projection(self):
        rng = np.random.default_rng(7)
        basis = random_basis(12, 3, rng)
        english, translated, g, center = rng.normal(size=(4, 12))
        result = direction_metrics(english, translated, g, .4, center, basis)
        self.assertLess(result['decomposition_error'], 1e-12)
        expected = g @ (center + project(translated - center, basis)) + .4
        self.assertAlmostEqual(result['projected_margin_other'], expected)

    def test_large_normal_displacement_can_preserve_margin(self):
        basis = np.array([[1.], [0.], [0.]])
        result = direction_metrics(np.array([1., 0., 0.]), np.array([1., 100., 0.]),
                                   np.array([2., 0., 0.]), 0., np.zeros(3), basis)
        self.assertEqual(result['outside_fraction'], 1.)
        self.assertEqual(result['direction_retention'], 1.)
        self.assertEqual(result['delta_margin'], 0.)

    def test_inside_displacement_can_cross_margin_boundary(self):
        basis = np.array([[1.], [0.]])
        result = direction_metrics(np.array([.1, 2.]), np.array([-.1, 2.]),
                                   np.array([1., 0.]), 0., np.zeros(2), basis)
        self.assertGreater(result['margin_en'], 0)
        self.assertLess(result['margin_other'], 0)
        self.assertEqual(result['outside_delta_margin'], 0.)

    def test_outside_direction_can_move_toward_affine_space(self):
        basis = np.array([[1.], [0.]])
        result = direction_metrics(np.array([1., 4.]), np.array([1., 2.]),
                                   np.array([0., 1.]), 0., np.zeros(2), basis)
        self.assertEqual(result['outside_displacement'], 2.)
        self.assertEqual(result['distance_change'], -2.)

    def test_svd_rank_and_rotation_invariance(self):
        rng = np.random.default_rng(19)
        x = rng.normal(size=(12, 6))
        center, basis, stats = fit_subspace(x, 2)
        rotated, _ = np.linalg.qr(rng.normal(size=(2, 2)))
        first = state_metrics(x[0], center, basis)
        second = state_metrics(x[0], center, basis @ rotated)
        self.assertAlmostEqual(first['distance'], second['distance'])
        self.assertAlmostEqual(principal_angles(basis, basis @ rotated)['overlap'], 1.)
        self.assertEqual(stats['numerical_rank'], 6)
        with self.assertRaises(ValueError):
            fit_subspace(x, 12)


class AnchorTests(unittest.TestCase):
    def test_shared_prefix_and_actual_first_divergence(self):
        class Tokenizer:
            def encode(self, text, add_special_tokens=False):
                return [int(v) for v in text.split()]
        rows = [{'id': 'a', 'language': 'en', 'group': 'instruct',
                 'dangerous': False, 'response': '1 2 3'},
                {'id': 'a', 'language': 'en', 'group': 'base',
                 'dangerous': True, 'response': '1 4 5'},
                {'id': 'b', 'language': 'en', 'group': 'base',
                 'dangerous': False, 'response': '6'}]
        anchors = english_anchors(rows, Tokenizer())
        self.assertEqual(anchors['a']['prefix'], [1])
        self.assertEqual(anchors['a']['safe_token'], 2)
        self.assertEqual(anchors['a']['danger_token'], 4)
        self.assertNotIn('b', anchors)


class PipelineTests(unittest.TestCase):
    def test_cross_fit_excludes_all_languages_of_test_request(self):
        rng = np.random.default_rng(21)
        ids, languages, groups, labels, answer, prompt = [], [], [], [], [], []
        anchor_ids, anchor_languages, anchor, directions, margins = [], [], [], [], []
        for ident in range(12):
            g = rng.normal(size=16)
            for language in ('en', 'ja'):
                state = rng.normal(size=16)
                anchor_ids.append(str(ident))
                anchor_languages.append(language)
                anchor.append(state)
                directions.append(g)
                margins.append(float(g @ state))
                for group in ('base', 'instruct'):
                    ids.append(str(ident))
                    languages.append(language)
                    groups.append(group)
                    label = (ident % 3 == 0) != (group == 'base')
                    labels.append(label)
                    prompt.append(state)
                    answer.append(rng.normal(size=16) + int(label))
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / 'features.npz'
            valid = np.ones(len(ids), dtype=bool)
            valid[3] = False  # Missing Japanese answer state for request zero, Instruct source.
            answer[3] = np.zeros(16)
            np.savez(source, ids=ids, languages=languages, groups=groups, labels=labels,
                     prompt=prompt, answer=answer, answer_valid=valid, anchor_ids=anchor_ids,
                     anchor_languages=anchor_languages, anchor=anchor, directions=directions,
                     biases=np.zeros(len(anchor)), model_margins=margins,
                     anchor_prefixes=np.array(['[]'] * len(anchor)),
                     metadata=np.array(json.dumps({'state_space': 'final_normalized_lm_head_input'})))
            args = argparse.Namespace(input=str(source), output_dir=str(root/'result'),
                label_group='instruct', fit_groups=['base', 'instruct'], fit_position='answer',
                ranks=[1, 2], folds=3, seed=123, random_directions=3, bootstrap=20)
            with redirect_stdout(io.StringIO()):
                analyze(args)
            rows = [json.loads(line) for line in (root/'result'/'pairs.jsonl').read_text().splitlines()]
            self.assertEqual(len(rows), 12 * 2 * 2 * 3 - 4)
            self.assertFalse(any(r['id'] == '0' and r['kind'] == 'answer' for r in rows))
            for row in rows:
                if row['kind'] == 'anchored_margin':
                    self.assertLess(row['decomposition_error'], 1e-10)
            stats = [json.loads(line) for line in (root/'result'/'spectra.jsonl').read_text().splitlines()]
            for row in stats:
                if 'training_ids' in row:
                    self.assertFalse(set(row['training_ids']) & set(row['heldout_ids']))
            with self.assertRaises(FileExistsError), redirect_stdout(io.StringIO()):
                analyze(args)

    def test_prepare_rejects_uncertain_labels(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            row = {'id': 'a', 'language': 'en', 'goal': 'goal', 'prompt': 'goal',
                   'response': 'text', 'model': 'model'}
            (root/'responses.jsonl').write_text(json.dumps(row)+'\n')
            (root/'labels.jsonl').write_text(json.dumps({'id': 'a', 'group': 'base',
                'language': 'en', 'success': None})+'\n')
            args = argparse.Namespace(responses=[f'base={root / "responses.jsonl"}'],
                judgments=str(root/'labels.jsonl'), languages=['en'], output=str(root/'out.jsonl'))
            with self.assertRaisesRegex(ValueError, 'uncertain'):
                prepare(args)

    def test_prepare_keeps_empty_responses_for_prompt_analysis(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            row = {'id': 'a', 'language': 'en', 'goal': 'goal', 'prompt': 'goal',
                   'response': '', 'model': 'model'}
            (root/'responses.jsonl').write_text(json.dumps(row)+'\n')
            (root/'labels.jsonl').write_text(json.dumps({'id': 'a', 'group': 'base',
                'language': 'en', 'success': False})+'\n')
            args = argparse.Namespace(responses=[f'base={root / "responses.jsonl"}'],
                judgments=str(root/'labels.jsonl'), languages=['en'], output=str(root/'out.jsonl'))
            with redirect_stdout(io.StringIO()):
                prepare(args)
            self.assertEqual(json.loads((root/'out.jsonl').read_text())['response'], '')


if __name__ == '__main__':
    unittest.main()
