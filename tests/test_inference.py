"""Tests for the inference pipeline (end-to-end prediction)."""
import numpy as np
import pytest
from unittest.mock import patch, MagicMock

from src.feature_engineering import compute_all_features, select_features_mi


ASPIRIN = "CC(=O)Oc1ccccc1C(=O)O"
HIV_INHIBITOR = "CC(C)CN(CC(O)C(Cc1ccccc1)NC(=O)OC1CCOC1)S(=O)(=O)c1ccc(N)cc1"  # Amprenavir-like


class TestFeatureEngineering:
    def test_feature_extraction_deterministic(self):
        f1 = compute_all_features(ASPIRIN)
        f2 = compute_all_features(ASPIRIN)
        assert f1 is not None and f2 is not None
        for k in f1:
            assert f1[k] == f2[k], f"Feature {k} not deterministic"

    def test_different_molecules_different_features(self):
        f1 = compute_all_features(ASPIRIN)
        f2 = compute_all_features(HIV_INHIBITOR)
        assert f1['mol_weight'] != f2['mol_weight']

    def test_hiv_inhibitor_properties(self):
        """Amprenavir-like molecule should have known properties."""
        feats = compute_all_features(HIV_INHIBITOR)
        assert feats is not None
        assert feats['mol_weight'] > 300
        assert feats['ring_count'] >= 1


class TestMISelection:
    def test_selects_k_features(self):
        np.random.seed(42)
        X = np.random.randn(200, 50).astype(np.float32)
        y = (X[:, 0] + X[:, 1] > 0).astype(int)
        idx = select_features_mi(X, y, k=10, seed=42)
        assert len(idx) == 10
        assert idx.max() < 50

    def test_deterministic(self):
        np.random.seed(42)
        X = np.random.randn(200, 50).astype(np.float32)
        y = (X[:, 0] + X[:, 1] > 0).astype(int)
        idx1 = select_features_mi(X, y, k=10, seed=42)
        idx2 = select_features_mi(X, y, k=10, seed=42)
        np.testing.assert_array_equal(sorted(idx1), sorted(idx2))


class TestPredictionFormat:
    def test_feature_vector_completeness(self):
        """Ensure feature vector has all expected categories."""
        feats = compute_all_features(ASPIRIN)
        domain_keys = ['mol_weight', 'logp', 'hbd', 'hba', 'tpsa',
                       'rotatable_bonds', 'aromatic_rings', 'ring_count',
                       'heavy_atom_count', 'fraction_csp3']
        for k in domain_keys:
            assert k in feats, f"Missing domain feature: {k}"

        morgan_keys = [f'mfp_{i}' for i in range(1024)]
        for k in morgan_keys:
            assert k in feats, f"Missing Morgan bit: {k}"

        maccs_keys = [f'maccs_{i}' for i in range(167)]
        for k in maccs_keys:
            assert k in feats, f"Missing MACCS key: {k}"

    def test_fingerprint_binary(self):
        feats = compute_all_features(ASPIRIN)
        for i in range(1024):
            assert feats[f'mfp_{i}'] in (0, 1)
        for i in range(167):
            assert feats[f'maccs_{i}'] in (0, 1)
