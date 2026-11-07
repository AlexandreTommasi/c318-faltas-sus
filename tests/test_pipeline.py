"""Regressões metodológicas: vazamento, partições e ranking com empates."""
import unittest

import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier

from src.features import split_features_target, encode_categoricals
from src.models import split_by_patient, split_temporal
from src.evaluate import compute_metrics


class PipelineTests(unittest.TestCase):
    def fixture(self):
        return pd.DataFrame({
            'patient_id': ['a', 'a', 'a', 'b'],
            'scheduled_day': pd.to_datetime(['2016-05-01', '2016-05-02', '2016-05-03', '2016-05-03'], utc=True),
            'appointment_day': pd.to_datetime(['2016-05-02', '2016-05-02', '2016-05-04', '2016-05-04'], utc=True),
            'no_show': [1, 0, 1, 1],
        })

    def test_cleaning_preserves_target_and_removes_invalid_rows(self):
        from src.data import clean
        df = pd.DataFrame({
            'PatientId': [1, 1, 2, 3, 4],
            'AppointmentID': [10, 11, 12, 13, 14],
            'ScheduledDay': ['2016-05-02'] * 5,
            'AppointmentDay': ['2016-05-02'] * 4 + ['2016-05-01'],
            'Age': [20, 20, -1, 115, 30],
            'No-show': ['Yes', 'No', 'Yes', 'No', 'Yes'],
        })
        result = clean(df)
        self.assertEqual(result.patient_id.tolist(), ['1', '3'])
        self.assertEqual(result.no_show.tolist(), [1, 0])
        pd.testing.assert_frame_equal(clean(result), result)

    def test_group_partition(self):
        df = self.fixture()
        train, test = split_by_patient(df, .5)
        self.assertFalse(set(df.iloc[train].patient_id) & set(df.iloc[test].patient_id))

    def test_temporal_excludes_pending_outcomes(self):
        df = self.fixture()
        train, test = split_temporal(df, '2016-05-03')
        self.assertTrue((df.iloc[train].appointment_day < pd.Timestamp('2016-05-03', tz='UTC')).all())
        self.assertTrue((df.iloc[test].scheduled_day >= pd.Timestamp('2016-05-03', tz='UTC')).all())
        self.assertFalse(set(train) & set(test))

    def test_dummy_ranking_uses_expected_random_ties(self):
        X = np.zeros((10, 1))
        y = np.array([1, 1, 0, 0, 0, 0, 0, 0, 0, 0])
        model = DummyClassifier(strategy='prior').fit(X, y)
        metrics = compute_metrics(model, X, y)
        self.assertAlmostEqual(metrics['precision_at_20'], .2)
        self.assertAlmostEqual(metrics['recall_at_20'], .2)
        self.assertEqual(metrics['confusion_matrix'], [[8, 0], [2, 0]])

    def test_preprocessor_does_not_learn_test_categories(self):
        from src.features import NUMERIC_FEATURES, CATEGORICAL_FEATURES
        df = pd.DataFrame({c: [0, 1] for c in NUMERIC_FEATURES})
        df = df.assign(gender=['F', 'M'], neighbourhood=['A', 'NEW'], weekday=[0, 1], no_show=[0, 1], sms_received=[0, 1])
        X, _ = split_features_target(df)
        self.assertNotIn('sms_received', X)
        transformer = encode_categoricals(X).fit(X.iloc[:1])
        transformer.transform(X.iloc[1:])
        self.assertNotIn('NEW', transformer.named_transformers_['categorical'].categories_[1])


if __name__ == '__main__':
    unittest.main()
