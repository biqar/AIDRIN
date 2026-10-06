"""Unit tests for shared ID-like column assessment."""

import unittest
import uuid

import numpy as np
import pandas as pd

from web.readiness.id_like import (
    ID_LIKE_THRESHOLD,
    ID_UNIQUE_RATIO_FLOOR,
    assess_id_likeness,
    is_integer_like_numeric,
)
from web.routes.metrics import _build_feature_profiles, _feature_readiness_status


class TestIsIntegerLikeNumeric(unittest.TestCase):
    def test_integer_dtype(self):
        self.assertTrue(is_integer_like_numeric(pd.Series([1, 2, 3])))

    def test_whole_number_floats(self):
        self.assertTrue(is_integer_like_numeric(pd.Series([1.0, 2.0, 3.0])))

    def test_continuous_floats(self):
        self.assertFalse(is_integer_like_numeric(pd.Series([1.1, 2.2, 3.3])))


class TestAssessIdLikeness(unittest.TestCase):
    def test_below_floor_is_not_id_like(self):
        s = pd.Series(["a", "b", "a", "b", "a"] * 4)
        result = assess_id_likeness(s, col_name="label", n_rows=len(s))
        self.assertLess(result.unique_ratio, ID_UNIQUE_RATIO_FLOOR)
        self.assertEqual(result.score, 0.0)
        self.assertFalse(result.is_id_like)
        self.assertFalse(result.is_possible_id)

    def test_near_unique_continuous_float_not_id_like(self):
        n = 100
        rng = np.random.default_rng(0)
        s = pd.Series(rng.normal(size=n))
        result = assess_id_likeness(s, col_name="velocity", n_rows=n)
        self.assertGreaterEqual(result.unique_ratio, ID_UNIQUE_RATIO_FLOOR)
        self.assertFalse(result.is_id_like)
        self.assertLess(result.score, ID_LIKE_THRESHOLD)

    def test_near_unique_integer_id_is_id_like(self):
        n = 100
        s = pd.Series(np.arange(n))
        result = assess_id_likeness(s, col_name="row_id", n_rows=n)
        self.assertTrue(result.is_id_like)
        self.assertEqual(result.reason, "ID-like (near-unique values)")
        self.assertGreaterEqual(result.score, ID_LIKE_THRESHOLD)

    def test_uuid_categorical_is_id_like(self):
        n = 50
        s = pd.Series([str(uuid.uuid4()) for _ in range(n)])
        result = assess_id_likeness(s, col_name="token", n_rows=n)
        self.assertTrue(result.is_id_like)
        self.assertTrue(result.signals["identifier_shaped"])

    def test_name_hint_boosts_integer_sequence(self):
        n = 100
        s = pd.Series(np.arange(n))
        with_hint = assess_id_likeness(s, col_name="patient_id", n_rows=n)
        without = assess_id_likeness(s, col_name="amount", n_rows=n)
        self.assertGreaterEqual(with_hint.score, without.score)
        self.assertTrue(with_hint.is_id_like)


class TestFeatureReadinessStatusIdLike(unittest.TestCase):
    def test_continuous_near_unique_not_poor_from_id_rule(self):
        n = 80
        rng = np.random.default_rng(1)
        s = pd.Series(rng.normal(size=n))
        status = _feature_readiness_status(
            pct_missing=0.0,
            n_unique=int(s.nunique()),
            n_rows=n,
            feat_type="numerical",
            pct_dominant=1.0 / n,
            series=s,
            col_name="accel_x",
        )
        self.assertNotEqual(status, "poor")

    def test_integer_id_is_poor(self):
        n = 80
        s = pd.Series(np.arange(n))
        status = _feature_readiness_status(
            pct_missing=0.0,
            n_unique=n,
            n_rows=n,
            feat_type="numerical",
            pct_dominant=1.0 / n,
            series=s,
            col_name="user_id",
        )
        self.assertEqual(status, "poor")

    def test_parkfield_style_profiles_not_all_poor(self):
        n = 100
        rng = np.random.default_rng(2)
        df = pd.DataFrame({
            "time": rng.normal(size=n),
            "chan1": rng.normal(size=n),
            "chan2": rng.normal(size=n),
            "row_id": np.arange(n),
        })
        profiles, _ = _build_feature_profiles(df)
        by_name = {p["feature"]: p["status"] for p in profiles}
        self.assertNotEqual(by_name["time"], "poor")
        self.assertNotEqual(by_name["chan1"], "poor")
        self.assertNotEqual(by_name["chan2"], "poor")
        self.assertEqual(by_name["row_id"], "poor")


if __name__ == "__main__":
    unittest.main()
