"""Regressões das tarefas #11 (EDA no treino) e #15 (balanceamento)."""
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTENC
from imblearn.under_sampling import RandomUnderSampler
from sklearn.exceptions import NotFittedError

from src.eda import prepare_eda, rate_table
from src.features import (CATEGORICAL_FEATURES, HISTORY_FEATURES,
                          NUMERIC_FEATURES, split_features_target)
from src.imbalance import STRATEGIES, compare_strategies, make_strategy
from src.models import split_by_patient

DISCRETE = ["bolsa_familia", "hypertension", "diabetes", "alcoholism", "handicap"]
EXPECTED_SMOTENC_CATEGORICAL = [1, 2, 3, 4, 5, 7, 8, 9]


def feature_frame(n_per_class=12, seed=0):
    rng = np.random.RandomState(seed)
    n = 2 * n_per_class
    return pd.DataFrame({
        "age": rng.randint(0, 90, n),
        "bolsa_familia": rng.randint(0, 2, n),
        "hypertension": rng.randint(0, 2, n),
        "diabetes": rng.randint(0, 2, n),
        "alcoholism": rng.randint(0, 2, n),
        "handicap": rng.randint(0, 3, n),
        "waiting_days": rng.randint(0, 60, n),
        "gender": rng.choice(["F", "M"], n),
        "neighbourhood": rng.choice(["A", "B", "C"], n),
        "weekday": rng.randint(0, 6, n),
        "no_show": [0] * n_per_class + [1] * n_per_class,
    })


def synthetic_patients(n_patients=120, seed=1):
    rng = np.random.RandomState(seed)
    rows = []
    for i in range(n_patients):
        for label in (0, 1):
            rows.append({
                "patient_id": f"p{i:04d}",
                "appointment_id": i * 10 + label,
                "age": int(rng.randint(0, 90)),
                "bolsa_familia": int(rng.randint(0, 2)),
                "hypertension": int(rng.randint(0, 2)),
                "diabetes": int(rng.randint(0, 2)),
                "alcoholism": int(rng.randint(0, 2)),
                "handicap": int(rng.randint(0, 3)),
                "waiting_days": int(rng.randint(0, 60)),
                "gender": rng.choice(["F", "M"]),
                "neighbourhood": rng.choice(["A", "B", "C"]),
                "weekday": int(rng.randint(0, 6)),
                "sms_received": int(rng.randint(0, 2)),
                "no_show": label,
            })
    return pd.DataFrame(rows)


class StubClassifier:
    def __init__(self):
        self.fit_inputs = []
        self.predict_inputs = []

    def fit(self, X, y):
        self.fit_inputs.append(X.copy())
        self.classes_ = np.array([0, 1])
        return self

    def predict(self, X):
        self.predict_inputs.append(X.copy())
        return np.zeros(len(X), dtype=int)

    def predict_proba(self, X):
        return np.tile([0.5, 0.5], (len(X), 1))


class PrepareEdaTests(unittest.TestCase):
    def test_age_bins_edges(self):
        df = pd.DataFrame({
            "age": [0, 12, 13, 17, 18, 29, 30, 44, 45, 59, 60, 115],
            "waiting_days": 0,
        })
        groups = prepare_eda(df)["age_group"]
        self.assertFalse(groups.isna().any())
        self.assertEqual(groups.astype(str).tolist(), [
            "0–12", "0–12", "13–17", "13–17", "18–29", "18–29",
            "30–44", "30–44", "45–59", "45–59", "60+", "60+",
        ])

    def test_waiting_bins_edges(self):
        df = pd.DataFrame({
            "age": 30,
            "waiting_days": [0, 1, 2, 3, 7, 8, 14, 15, 30, 31, 60, 61, 179],
        })
        groups = prepare_eda(df)["waiting_group"]
        self.assertFalse(groups.isna().any())
        self.assertEqual(groups.astype(str).tolist(), [
            "0", "1–2", "1–2", "3–7", "3–7", "8–14", "8–14",
            "15–30", "15–30", "31–60", "31–60", "61+", "61+",
        ])


class RateTableTests(unittest.TestCase):
    def test_counts_rates_and_shares(self):
        df = pd.DataFrame({"gender": ["F", "F", "M"], "no_show": [1, 0, 0]})
        table = rate_table(df, ["gender"]).set_index("gender")
        self.assertEqual(table.loc["F", "n"], 2)
        self.assertEqual(table.loc["M", "n"], 1)
        self.assertEqual(table.loc["F", "no_shows"], 1)
        self.assertEqual(table.loc["M", "no_shows"], 0)
        self.assertAlmostEqual(table.loc["F", "no_show_rate"], 0.5)
        self.assertAlmostEqual(table.loc["M", "no_show_rate"], 0.0)
        self.assertAlmostEqual(table.loc["F", "share"], 2 / 3)
        self.assertAlmostEqual(table.loc["M", "share"], 1 / 3)

    def test_crossed_table_omits_unobserved_combinations(self):
        df = pd.DataFrame({
            "waiting_group": ["0", "0", "3–7", "3–7"],
            "sms_received": [0, 0, 0, 1],
            "no_show": [0, 0, 1, 0],
        })
        table = rate_table(df, ["waiting_group", "sms_received"])
        pairs = set(zip(table["waiting_group"].astype(str), table["sms_received"]))
        self.assertNotIn(("0", 1), pairs)
        self.assertEqual(len(table), 3)


class MakeStrategyTests(unittest.TestCase):
    def setUp(self):
        self.df = feature_frame()
        self.X, self.y = split_features_target(self.df)

    def test_all_strategies_unfitted(self):
        for name in STRATEGIES:
            pipeline = make_strategy(name)
            self.assertEqual([step for step, _ in pipeline.steps],
                             ["before", "sampler", "after", "model"])
            with self.assertRaises(NotFittedError):
                pipeline.predict(self.X)

    def test_unknown_strategy_raises(self):
        with self.assertRaises(ValueError):
            make_strategy("nao_existe")

    def test_only_class_weight_is_balanced(self):
        for name in STRATEGIES:
            weight = make_strategy(name).named_steps["model"].class_weight
            if name == "class_weight":
                self.assertEqual(weight, "balanced")
            else:
                self.assertIsNone(weight)

    def test_sampler_types(self):
        self.assertEqual(make_strategy("unbalanced").named_steps["sampler"],
                         "passthrough")
        self.assertEqual(make_strategy("class_weight").named_steps["sampler"],
                         "passthrough")
        self.assertIsInstance(
            make_strategy("undersampling").named_steps["sampler"],
            RandomUnderSampler)
        self.assertIsInstance(
            make_strategy("smote").named_steps["sampler"], SMOTENC)

    def test_smotenc_categorical_indices(self):
        sampler = make_strategy("smote").named_steps["sampler"]
        expected = [NUMERIC_FEATURES.index(c) for c in DISCRETE] + list(
            range(len(NUMERIC_FEATURES),
                  len(NUMERIC_FEATURES) + len(CATEGORICAL_FEATURES)))
        self.assertEqual(list(sampler.categorical_features), expected)
        self.assertEqual(list(sampler.categorical_features),
                         EXPECTED_SMOTENC_CATEGORICAL)

    def test_fit_and_unknown_category_transform(self):
        pipeline = make_strategy("smote")
        pipeline.fit(self.X, self.y)
        new = self.X.iloc[:2].copy()
        new["neighbourhood"] = "NEW"
        predictions = pipeline.predict(new)
        self.assertEqual(len(predictions), 2)

    def test_approved_inputs_exclude_identifiers_history_and_sms(self):
        self.assertEqual(list(self.X.columns),
                         NUMERIC_FEATURES + CATEGORICAL_FEATURES)
        for column in ("patient_id", "appointment_id", "sms_received",
                       *HISTORY_FEATURES, "no_show", "same_day"):
            self.assertNotIn(column, self.X.columns)

    def test_smotenc_keeps_discrete_values(self):
        pipeline = make_strategy("smote")
        before = pipeline.named_steps["before"]
        sampler = pipeline.named_steps["sampler"]
        imbalanced = pd.concat([
            self.df[self.df["no_show"] == 0],
            self.df[self.df["no_show"] == 1].head(8),
        ], ignore_index=True)
        X, y = split_features_target(imbalanced)
        transformed = before.fit_transform(X)
        resampled, _ = sampler.fit_resample(transformed, y)
        self.assertGreater(len(resampled), len(transformed))
        numeric = np.asarray(resampled)[:, :len(NUMERIC_FEATURES)]
        numeric = before.named_transformers_["numeric"].inverse_transform(numeric)
        numeric = pd.DataFrame(numeric, columns=NUMERIC_FEATURES)
        for name in DISCRETE:
            observed = np.sort(X[name].unique())
            distance = np.abs(numeric[name].to_numpy()[:, None] - observed[None, :])
            self.assertTrue((distance.min(axis=1) < 1e-6).all(), name)
        categorical = np.asarray(resampled)[:, len(NUMERIC_FEATURES):]
        categorical = before.named_transformers_["categorical"].inverse_transform(
            categorical)
        for index, name in enumerate(CATEGORICAL_FEATURES):
            self.assertLessEqual(set(np.asarray(categorical)[:, index]),
                                 set(X[name].unique()), name)


class CompareStrategiesTests(unittest.TestCase):
    def setUp(self):
        self.df = synthetic_patients()

    def run_comparison(self, df):
        made = {name: [] for name in STRATEGIES}

        def factory(strategy):
            stub = StubClassifier()
            made[strategy].append(stub)
            return stub

        with mock.patch("src.imbalance.make_strategy", side_effect=factory) as patched:
            results, summary, metadata = compare_strategies(df)
        return results, summary, metadata, made, patched

    def test_same_partitions_and_validation_inputs(self):
        results, summary, metadata, made, patched = self.run_comparison(self.df)
        self.assertEqual({call.args[0] for call in patched.call_args_list},
                         set(STRATEGIES))
        n_folds = metadata["n_splits"]
        for name in STRATEGIES:
            self.assertEqual(len(made[name]), n_folds)
        for fold in range(n_folds):
            fit_reference = made[STRATEGIES[0]][fold].fit_inputs[0]
            validation_reference = made[STRATEGIES[0]][fold].predict_inputs[0]
            for name in STRATEGIES[1:]:
                pd.testing.assert_frame_equal(
                    fit_reference, made[name][fold].fit_inputs[0])
                pd.testing.assert_frame_equal(
                    validation_reference, made[name][fold].predict_inputs[0])
            self.assertEqual(len(validation_reference),
                             metadata["folds"][fold]["validation_rows"])
            self.assertEqual(list(validation_reference.columns),
                             NUMERIC_FEATURES + CATEGORICAL_FEATURES)

    def test_no_shared_groups(self):
        _, _, metadata, _, _ = self.run_comparison(self.df)
        development = self.df.iloc[split_by_patient(self.df)[0]]
        patients = development["patient_id"].nunique()
        for fold in metadata["folds"]:
            self.assertEqual(fold["shared_patients"], 0)
            self.assertEqual(fold["train_patients"] + fold["validation_patients"],
                             patients)

    def test_heldout_patients_do_not_change_development(self):
        results, summary, _, _, _ = self.run_comparison(self.df)
        train_idx, test_idx = split_by_patient(self.df)
        heldout = set(self.df.iloc[test_idx]["patient_id"])
        changed = self.df.copy()
        mask = changed["patient_id"].isin(heldout)
        changed.loc[mask, "no_show"] = 1 - changed.loc[mask, "no_show"]
        results2, summary2, _, _, _ = self.run_comparison(changed)
        pd.testing.assert_frame_equal(results, results2)
        pd.testing.assert_frame_equal(summary, summary2)


class SprintPlotsTests(unittest.TestCase):
    def test_render_smoke(self):
        from src.sprint_plots import render
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            eda_dir = base / "eda"
            imbalance_dir = base / "imbalance"
            eda_dir.mkdir(parents=True)
            imbalance_dir.mkdir()

            def table(name, key, values):
                pd.DataFrame({
                    key: values,
                    "n": [10, 20] + [5] * (len(values) - 2),
                    "no_shows": [2, 4] + [1] * (len(values) - 2),
                    "no_show_rate": [0.2, 0.2] + [0.2] * (len(values) - 2),
                    "share": [0.4, 0.4] + [0.2] * (len(values) - 2),
                }).to_csv(eda_dir / f"{name}.csv", index=False)

            table("age_group", "age_group", ["0–12", "13–17", "60+"])
            table("gender", "gender", ["F", "M"])
            table("neighbourhood", "neighbourhood",
                  [f"BAIRRO{i}" for i in range(60)])
            table("bolsa_familia", "bolsa_familia", [0, 1])
            table("hypertension", "hypertension", [0, 1])
            table("diabetes", "diabetes", [0, 1])
            table("alcoholism", "alcoholism", [0, 1])
            table("handicap", "handicap", [0, 1, 2])
            table("waiting_group", "waiting_group",
                  ["0", "1–2", "3–7", "8–14", "15–30", "31–60", "61+"])
            table("weekday", "weekday", [0, 1, 2, 3, 4, 5])
            table("sms_received", "sms_received", [0, 1])
            table("age", "age", [0, 30, 60])
            table("waiting_days", "waiting_days", [0, 10, 60])
            pd.DataFrame({
                "waiting_group": ["0", "3–7", "3–7"],
                "sms_received": [0, 0, 1],
                "n": [10, 8, 9],
                "no_shows": [1, 2, 2],
                "no_show_rate": [0.1, 0.25, 0.22],
                "share": [0.4, 0.3, 0.3],
            }).to_csv(eda_dir / "sms_by_waiting.csv", index=False)
            metrics = ["pr_auc_ap", "roc_auc", "precision", "recall", "f1",
                       "precision_at_20", "recall_at_20"]
            comparison = pd.DataFrame(
                {f"{m}_{s}": [0.3, 0.31, 0.30, 0.29] if s == "mean"
                 else [0.005] * 4 for m in metrics for s in ("mean", "std")},
                index=["unbalanced", "class_weight", "undersampling", "smote"])
            comparison.index.name = "strategy"
            comparison.to_csv(imbalance_dir / "comparison.csv")

            written = render(base)
            self.assertTrue(all(path.exists() for path in written))
            names = {path.name for path in written}
            self.assertIn("eda_neighbourhood.png", names)
            self.assertIn("eda_sms_por_espera.png", names)
            self.assertIn("imbalance_comparacao.png", names)
            self.assertEqual(len(list((base / "figures").glob("*.png"))),
                             len(written))


if __name__ == "__main__":
    unittest.main()
