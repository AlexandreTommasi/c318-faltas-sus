"""Verifica isolamento por paciente e seleção independente do limiar."""
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

from src.features import NUMERIC_FEATURES
from src.model_selection import patient_folds, select_threshold, run


class ModelSelectionTests(unittest.TestCase):
    def test_folds_keep_repeated_patients_together(self):
        groups = pd.Series(np.repeat(np.arange(40), 3))
        y = pd.Series(np.tile([0, 1, 0, 1], 30))
        folds = patient_folds(pd.DataFrame({'x': y}), y, groups)
        seen = []
        for train, valid in folds:
            self.assertFalse(set(groups.iloc[train]) & set(groups.iloc[valid]))
            self.assertEqual(set(y.iloc[valid]), {0, 1})
            seen.extend(valid)
        self.assertEqual(sorted(seen), list(range(len(y))))

    def test_threshold_maximizes_f1_and_handles_ties(self):
        y = [0, 1, 1, 0, 1, 0]
        scores = np.array([.1, .3, .3, .6, .8, .9])
        threshold, curve = select_threshold(y, scores)
        expected = max(f1_score(y, scores >= t) for t in np.unique(scores))
        self.assertAlmostEqual(f1_score(y, scores >= threshold), expected)
        self.assertAlmostEqual(curve.f1.max(), expected)
        with self.assertRaises(ValueError):
            select_threshold([0, 0], [.1, .2])

    def _make_df(self):
        rng = np.random.default_rng(42)
        n = 400
        df = pd.DataFrame({c: rng.integers(0, 2, n) for c in NUMERIC_FEATURES})
        df['age'] = rng.integers(0, 100, n)
        df['waiting_days'] = rng.integers(0, 40, n)
        return df.assign(patient_id=np.repeat(np.arange(n // 2), 2),
                         gender=rng.choice(['F', 'M'], n),
                         neighbourhood=rng.choice(['A', 'B'], n),
                         weekday=rng.integers(0, 5, n),
                         no_show=(rng.random(n) < .3).astype(int))

    def test_no_test_metrics_by_default(self):
        """Com evaluate_test=False (padrão), results não contém chaves test_*."""
        df = self._make_df()
        with tempfile.TemporaryDirectory() as directory:
            results = run(df, directory, n_splits=2)
            for name in ['logistic', 'decision_tree']:
                self.assertNotIn('test_default', results[name])
                self.assertNotIn('test_adjusted', results[name])

    def test_end_to_end_reports_and_probability_metrics(self):
        df = self._make_df()
        with tempfile.TemporaryDirectory() as directory:
            results = run(df, directory, n_splits=2)
            self.assertEqual(len(pd.read_csv(Path(directory) / 's4_cv.csv')), 4)
            self.assertEqual(json.loads((Path(directory) / 'results.json').read_text()), results)
            for name in ['logistic', 'decision_tree']:
                self.assertIn('best_params', results[name])
                self.assertIn('threshold', results[name])
                self.assertIn('validation_best_f1', results[name])


if __name__ == '__main__':
    unittest.main()
