"""Tests for data pipeline and feature engineering."""
import numpy as np
import pytest
from src.data_pipeline import compute_lipinski_features, compute_morgan_fingerprints
from src.feature_engineering import compute_all_features, compute_features_batch


ASPIRIN = "CC(=O)Oc1ccccc1C(=O)O"
CAFFEINE = "Cn1c(=O)c2c(ncn2C)n(C)c1=O"
INVALID = "NOT_A_SMILES"


class TestLipinskiFeatures:
    def test_valid_smiles(self):
        feats = compute_lipinski_features(ASPIRIN)
        assert feats is not None
        assert 'mol_weight' in feats
        assert 'logp' in feats
        assert 'hbd' in feats
        assert 'hba' in feats
        assert feats['mol_weight'] > 0

    def test_invalid_smiles(self):
        assert compute_lipinski_features(INVALID) is None

    def test_lipinski_violations(self):
        feats = compute_lipinski_features(ASPIRIN)
        assert feats['lipinski_violations'] >= 0
        assert feats['passes_lipinski'] in (0, 1)

    def test_aspirin_known_values(self):
        feats = compute_lipinski_features(ASPIRIN)
        assert 170 < feats['mol_weight'] < 190  # MW ~ 180.16
        assert feats['hbd'] <= 5
        assert feats['hba'] <= 10
        assert feats['passes_lipinski'] == 1


class TestMorganFingerprints:
    def test_valid_smiles(self):
        fp = compute_morgan_fingerprints(ASPIRIN)
        assert fp is not None
        assert len(fp) == 1024
        assert fp.dtype == np.int64 or fp.dtype == np.int32 or fp.min() >= 0

    def test_invalid_smiles(self):
        assert compute_morgan_fingerprints(INVALID) is None

    def test_custom_bits(self):
        fp = compute_morgan_fingerprints(ASPIRIN, n_bits=512)
        assert len(fp) == 512

    def test_binary_values(self):
        fp = compute_morgan_fingerprints(ASPIRIN)
        assert set(np.unique(fp)).issubset({0, 1})


class TestAllFeatures:
    def test_feature_count(self):
        feats = compute_all_features(ASPIRIN)
        assert feats is not None
        assert len(feats) > 1200  # 14 domain + 1024 Morgan + 167 MACCS + fragments

    def test_invalid_smiles(self):
        assert compute_all_features(INVALID) is None

    def test_morgan_keys(self):
        feats = compute_all_features(ASPIRIN)
        assert 'mfp_0' in feats
        assert 'mfp_1023' in feats

    def test_maccs_keys(self):
        feats = compute_all_features(ASPIRIN)
        assert 'maccs_0' in feats
        assert 'maccs_166' in feats


class TestBatchFeatures:
    def test_batch_processing(self):
        df = compute_features_batch([ASPIRIN, CAFFEINE])
        assert len(df) == 2
        assert 'mol_weight' in df.columns

    def test_invalid_filtered(self):
        df = compute_features_batch([ASPIRIN, INVALID, CAFFEINE])
        assert len(df) == 2

    def test_empty_input(self):
        df = compute_features_batch([])
        assert len(df) == 0
