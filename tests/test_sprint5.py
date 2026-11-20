"""Regressoes da tarefa #19 (curvas ROC/PR e matrizes de confusao)."""
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import nbformat
import numpy as np
import pandas as pd

from src.features import CATEGORICAL_FEATURES, NUMERIC_FEATURES
from src.imbalance import STRATEGIES
from src.models import split_by_patient
from src.validation_curves import curve_tables, evaluate_development
from src.validation_plots import render
from test_sprint34 import StubClassifier, synthetic_patients

NOTEBOOK = Path(__file__).resolve().parent.parent / "notebooks" / "04_avaliacao.ipynb"
APPROVED = NUMERIC_FEATURES + CATEGORICAL_FEATURES


def run_with_stubs(df):
    created = {name: [] for name in STRATEGIES}

    def factory(name):
        stub = StubClassifier()
        created[name].append(stub)
        return stub

    with mock.patch("src.imbalance.make_strategy", side_effect=factory):
        artifacts = evaluate_development(df)
    return artifacts, created


class CurveTablesTests(unittest.TestCase):
    def test_exact_curves(self):
        roc, pr = curve_tables(np.array([0, 1, 0, 1]), np.array([.1, .4, .35, .8]))
        np.testing.assert_allclose(
            roc[["fpr", "tpr"]].to_numpy(), [[0, 0], [0, .5], [0, 1], [1, 1]])
        np.testing.assert_allclose(pr["recall"], [1, 1, 1, .5, 0])
        np.testing.assert_allclose(pr["precision"], [.5, 2 / 3, 1, 1, 1])

    def test_tied_scores(self):
        roc, pr = curve_tables(np.array([0, 1, 0, 1]), np.array([.5] * 4))
        np.testing.assert_allclose(roc[["fpr", "tpr"]].to_numpy(), [[0, 0], [1, 1]])
        self.assertEqual(list(pr.columns), ["recall", "precision"])
        np.testing.assert_allclose(pr.to_numpy(), [[1, .5], [0, 1]])

    def test_downsampling_keeps_extremes(self):
        y = np.arange(1000) % 2
        scores = np.linspace(0, 1, 1000)
        small = curve_tables(y, scores, max_points=17)
        full = curve_tables(y, scores, max_points=1001)
        for reduced, complete in zip(small, full):
            self.assertLessEqual(len(reduced), 17)
            self.assertGreater(len(complete), 17)
            np.testing.assert_allclose(reduced.iloc[0], complete.iloc[0])
            np.testing.assert_allclose(reduced.iloc[-1], complete.iloc[-1])

    def test_invalid_inputs(self):
        good_y, good_s = np.array([0, 1, 0, 1]), np.array([.1, .4, .35, .8])
        cases = [
            (np.zeros(4), good_s, {}),
            (good_y, np.array([.1, np.nan, .3, .4]), {}),
            (good_y, np.array([.1, 1.2, .3, .4]), {}),
            (good_y, np.array([-.1, .2, .3, .4]), {}),
            (good_y, good_s[:3], {}),
            (good_y, good_s, {"max_points": 1}),
        ]
        for y, scores, kwargs in cases:
            with self.assertRaises(ValueError):
                curve_tables(y, scores, **kwargs)


class EvaluateDevelopmentTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.df = synthetic_patients()
        cls.artifacts, cls.created = run_with_stubs(cls.df)
        train, test = split_by_patient(cls.df)
        cls.train, cls.test = train, test
        cls.development = cls.df.iloc[train].reset_index(drop=True)

    def test_counts(self):
        results, _, _, _, _, matrices = self.artifacts
        self.assertEqual(sum(len(models) for models in self.created.values()), 20)
        self.assertEqual(len(results), 20)
        for strategy in STRATEGIES:
            table = matrices[matrices["strategy"] == strategy]
            self.assertEqual(len(table), 4)
            self.assertEqual(table["count"].sum(), len(self.development))
            positives = table[table["actual"] == 1]["count"].sum()
            self.assertEqual(positives, self.development["no_show"].sum())

    def test_each_row_validated_once_and_folds_are_disjoint(self):
        test_patients = set(self.df.iloc[self.test]["patient_id"])
        patient = self.development["patient_id"]
        for strategy, models in self.created.items():
            self.assertEqual(len(models), 5)
            validated = []
            for stub in models:
                fit_index = stub.fit_inputs[0].index
                predict_index = stub.predict_inputs[-1].index
                self.assertEqual(len(stub.fit_inputs), 1)
                self.assertTrue(fit_index.intersection(predict_index).empty)
                fit_patients = set(patient.iloc[fit_index])
                valid_patients = set(patient.iloc[predict_index])
                self.assertFalse(fit_patients & valid_patients)
                self.assertFalse((fit_patients | valid_patients) & test_patients)
                validated.extend(predict_index.tolist())
            self.assertEqual(sorted(validated), list(range(len(self.development))))

    def test_only_approved_features(self):
        for models in self.created.values():
            for stub in models:
                for frame in stub.fit_inputs + stub.predict_inputs:
                    self.assertEqual(list(frame.columns), APPROVED)
                    for column in ("patient_id", "appointment_id", "sms_received", "no_show"):
                        self.assertNotIn(column, frame.columns)

    def test_reserved_test_labels_do_not_change_outputs(self):
        altered = self.df.copy()
        column = altered.columns.get_loc("no_show")
        altered.iloc[self.test, column] = 1 - altered.iloc[self.test, column]
        other, _ = run_with_stubs(altered)
        base = self.artifacts
        self.assertEqual(base[0].to_json(), other[0].to_json())
        for left, right in zip(base[1:], other[1:]):
            if isinstance(left, dict):
                self.assertEqual(left, right)
            else:
                pd.testing.assert_frame_equal(left, right)


class RenderTests(unittest.TestCase):
    def test_render_writes_three_images_from_aggregates_only(self):
        (_, summary, metadata, roc, pr, matrices), _ = run_with_stubs(synthetic_patients())
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / "evaluation"
            source.mkdir()
            summary.to_csv(source / "comparison.csv")
            roc.to_csv(source / "roc_curves.csv", index=False)
            pr.to_csv(source / "precision_recall_curves.csv", index=False)
            matrices.to_csv(source / "confusion_matrices.csv", index=False)
            (source / "metadata.json").write_text(json.dumps(metadata))
            with mock.patch("src.validation_curves.evaluate_development",
                            side_effect=AssertionError("treino proibido")), \
                    mock.patch("src.data.load_raw",
                               side_effect=AssertionError("dado bruto proibido")):
                written = render(tmp)
            images = sorted(Path(tmp, "figures").glob("*.png"))
            self.assertEqual(len(written), 3)
            self.assertEqual([p.name for p in images],
                             ["confusion_matrices.png", "precision_recall.png", "roc.png"])
            for image in images:
                self.assertGreater(image.stat().st_size, 0)


class NotebookTests(unittest.TestCase):
    def test_notebook_valid_and_without_outputs(self):
        notebook = nbformat.read(NOTEBOOK, as_version=4)
        nbformat.validate(notebook)
        for cell in notebook.cells:
            if cell.cell_type == "code":
                self.assertEqual(cell.outputs, [])
                self.assertIsNone(cell.execution_count)


if __name__ == "__main__":
    unittest.main()
