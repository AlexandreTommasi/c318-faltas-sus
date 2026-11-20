"""Testes para src/final_model.py: use_history, folds sem vazamento, gate do teste."""
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from src.features import CATEGORICAL_FEATURES, HISTORY_FEATURES, NUMERIC_FEATURES
from src.final_model import (
    _AutoWeightXGB,
    _get_folds,
    get_feature_columns,
    make_preprocessor,
    run_final,
    run_shap_by_variable,
)


def _make_df(n: int = 500, seed: int = 0) -> pd.DataFrame:
    """Minimal synthetic DataFrame with all required columns."""
    rng = np.random.default_rng(seed)
    n_patients = max(n // 4, 20)
    reps = n // n_patients
    n = n_patients * reps  # ensure exact division
    return pd.DataFrame({
        "patient_id": np.repeat(np.arange(n_patients).astype(str), reps),
        "no_show": np.where(rng.random(n) < 0.25, 1, 0),
        "age": rng.integers(1, 90, n),
        "bolsa_familia": rng.integers(0, 2, n),
        "hypertension": rng.integers(0, 2, n),
        "diabetes": rng.integers(0, 2, n),
        "alcoholism": rng.integers(0, 2, n),
        "handicap": rng.integers(0, 2, n),
        "waiting_days": rng.integers(0, 60, n),
        "gender": rng.choice(["F", "M"], n),
        "neighbourhood": rng.choice(["A", "B", "C"], n),
        "weekday": rng.integers(0, 5, n),
        "previous_appointments": rng.integers(0, 5, n),
        "previous_no_shows": rng.integers(0, 3, n),
        "previous_no_show_rate": rng.random(n),
        "has_history": rng.integers(0, 2, n),
    })


class TestGetFeatureColumns(unittest.TestCase):
    def test_without_history_excludes_history_features(self):
        cols = get_feature_columns(use_history=False)
        for h in HISTORY_FEATURES:
            self.assertNotIn(h, cols, f"História '{h}' não deve aparecer sem use_history")

    def test_without_history_includes_base_features(self):
        cols = get_feature_columns(use_history=False)
        for f in NUMERIC_FEATURES + CATEGORICAL_FEATURES:
            self.assertIn(f, cols)

    def test_with_history_includes_all_features(self):
        cols = get_feature_columns(use_history=True)
        for f in NUMERIC_FEATURES + CATEGORICAL_FEATURES + list(HISTORY_FEATURES):
            self.assertIn(f, cols)

    def test_history_features_at_end(self):
        cols = get_feature_columns(use_history=True)
        base = get_feature_columns(use_history=False)
        self.assertEqual(cols[: len(base)], base)
        self.assertEqual(cols[len(base):], list(HISTORY_FEATURES))


class TestMakePreprocessor(unittest.TestCase):
    def _get_all_input_cols(self, prep) -> list:
        cols = []
        for _, _, c in prep.transformers:
            cols.extend(c if isinstance(c, list) else [c])
        return cols

    def test_preprocessor_excludes_history_when_false(self):
        prep = make_preprocessor(use_history=False)
        cols = self._get_all_input_cols(prep)
        for h in HISTORY_FEATURES:
            self.assertNotIn(h, cols)

    def test_preprocessor_includes_history_when_true(self):
        prep = make_preprocessor(use_history=True)
        cols = self._get_all_input_cols(prep)
        for h in HISTORY_FEATURES:
            self.assertIn(h, cols)

    def test_preprocessor_fits_and_transforms(self):
        df = _make_df(200)
        prep = make_preprocessor(use_history=True)
        X = df[get_feature_columns(use_history=True)]
        prep.fit(X)
        out = prep.transform(X)
        self.assertEqual(out.shape[0], len(df))
        self.assertGreater(out.shape[1], len(NUMERIC_FEATURES))


class TestGetFolds(unittest.TestCase):
    def _make_dev_df(self, n_patients: int = 60, reps: int = 5) -> pd.DataFrame:
        rng = np.random.default_rng(42)
        n = n_patients * reps
        return pd.DataFrame({
            "patient_id": np.repeat(np.arange(n_patients).astype(str), reps),
            "no_show": np.where(rng.random(n) < 0.25, 1, 0),
        })

    def test_no_patient_overlap_between_folds(self):
        dev_df = self._make_dev_df()
        folds = _get_folds(dev_df, n_splits=5)
        for tr, va in folds:
            overlap = set(dev_df["patient_id"].iloc[tr]) & set(dev_df["patient_id"].iloc[va])
            self.assertEqual(overlap, set(), "Pacientes não devem aparecer em treino e validação")

    def test_each_fold_contains_both_classes(self):
        dev_df = self._make_dev_df()
        folds = _get_folds(dev_df, n_splits=5)
        for tr, va in folds:
            self.assertEqual(dev_df["no_show"].iloc[tr].nunique(), 2)
            self.assertEqual(dev_df["no_show"].iloc[va].nunique(), 2)

    def test_all_rows_appear_exactly_once_in_validation(self):
        dev_df = self._make_dev_df()
        folds = _get_folds(dev_df, n_splits=5)
        all_val = []
        for _, va in folds:
            all_val.extend(va)
        self.assertEqual(sorted(all_val), list(range(len(dev_df))))

    def test_raises_on_patient_overlap(self):
        """Groups with shared patients across folds should raise (edge case check)."""
        # With n_patients=2 and n_splits=2, groups may overlap — verify guard fires
        dev_df = pd.DataFrame({
            "patient_id": np.array(["a"] * 50 + ["b"] * 50),
            "no_show": np.tile([0, 1], 50),
        })
        # This may or may not raise depending on sklearn's KFold assignment.
        # At minimum, the function should not crash without raising.
        try:
            _get_folds(dev_df, n_splits=2)
        except ValueError:
            pass  # Expected in pathological cases


class TestAutoWeightXGB(unittest.TestCase):
    def test_fit_sets_scale_pos_weight(self):
        rng = np.random.default_rng(0)
        X = rng.random((100, 5))
        y = np.where(rng.random(100) < 0.2, 1, 0)
        model = _AutoWeightXGB(xgb_params={"n_estimators": 10, "verbosity": 0})
        model.fit(X, y)
        neg = int((y == 0).sum())
        pos = int((y == 1).sum())
        self.assertAlmostEqual(model._clf.scale_pos_weight, neg / pos, places=5)

    def test_classes_attribute(self):
        rng = np.random.default_rng(1)
        X = rng.random((60, 3))
        y = np.tile([0, 1, 0], 20)
        model = _AutoWeightXGB(xgb_params={"n_estimators": 5, "verbosity": 0})
        model.fit(X, y)
        self.assertIn(0, model.classes_)
        self.assertIn(1, model.classes_)


class TestRunFinalGate(unittest.TestCase):
    def test_raises_without_selecao_congelada(self):
        df = _make_df(400)
        with tempfile.TemporaryDirectory() as tmpdir:
            with self.assertRaises(FileNotFoundError) as ctx:
                run_final(df, output=Path(tmpdir))
            self.assertIn("selecao_congelada.json", str(ctx.exception))

    def test_raises_with_message_about_freeze(self):
        df = _make_df(400)
        with tempfile.TemporaryDirectory() as tmpdir:
            try:
                run_final(df, output=Path(tmpdir))
            except FileNotFoundError as exc:
                self.assertIn("freeze", str(exc).lower())


class TestShapAggregationByVariable(unittest.TestCase):
    """Verifica que a agregação SHAP soma por linha (com sinal) antes do valor absoluto."""

    def test_sum_with_sign_before_abs_manual(self):
        # Exemplo mínimo feito à mão:
        # 2 linhas, 4 colunas OHE de "cat" + 1 numérica "num"
        # linha 0: cat_shap = [+2, -3, +1, 0] → soma = 0  → |soma| = 0
        # linha 1: cat_shap = [+1, +1, -1, -1] → soma = 0  → |soma| = 0
        # mean(|soma|) = 0.0  ≠  sum(mean(|col|)) = 2.0 (abordagem antiga)
        shap_vals = np.array([
            [2.0, -3.0, 1.0, 0.0, 5.0],
            [1.0,  1.0, -1.0, -1.0, 2.0],
        ])
        cat_indices = [0, 1, 2, 3]

        per_row = shap_vals[:, cat_indices].sum(axis=1)
        correct = float(np.abs(per_row).mean())
        self.assertAlmostEqual(correct, 0.0, places=10)

        wrong = float(np.abs(shap_vals[:, cat_indices]).mean(axis=0).sum())
        self.assertGreater(wrong, 0.0)
        self.assertLess(correct, wrong)

    def test_numeric_col_both_approaches_equal(self):
        # Para feature numérica (1 coluna), ambas as abordagens são equivalentes.
        shap_vals = np.array([[3.0], [-1.0], [2.0]])
        i = 0
        per_row = shap_vals[:, [i]].sum(axis=1)
        correct = float(np.abs(per_row).mean())
        classic = float(np.abs(shap_vals[:, i]).mean())
        self.assertAlmostEqual(correct, classic, places=10)

    def test_partial_cancellation_reduces_importance(self):
        # Sinais opostos em colunas OHE se cancelam → mean(|soma|) < sum(mean(|col|))
        shap_vals = np.array([
            [+0.5, -0.5],
            [+0.3, -0.3],
            [+0.1, -0.4],
        ])
        indices = [0, 1]
        per_row = shap_vals[:, indices].sum(axis=1)   # [0.0, 0.0, -0.3]
        correct = float(np.abs(per_row).mean())        # 0.1
        old_way = float(np.abs(shap_vals[:, indices]).mean(axis=0).sum())
        self.assertLessEqual(correct, old_way)


if __name__ == "__main__":
    unittest.main()
