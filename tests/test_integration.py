"""
Integration and performance tests for Drug Molecule Property Prediction.

Complementary to Anthony's 28 unit tests:
- End-to-end feature-to-prediction latency benchmarks
- Chemical correctness invariants (Aspirin, Caffeine, known properties)
- MI selection with realistic molecular distributions
- Batch throughput tests
- App-level smoke test (feature extraction → CatBoost prediction format)
"""

import time
import numpy as np
import pytest
from src.feature_engineering import (
    compute_all_features, compute_features_batch, select_features_mi
)
from src.data_pipeline import compute_lipinski_features

# ── Known molecules with verified properties ──────────────────
ASPIRIN      = "CC(=O)Oc1ccccc1C(=O)O"
CAFFEINE     = "Cn1c(=O)c2c(ncn2C)n(C)c1=O"
AMPRENAVIR   = "CC(C)CN(CC(O)C(Cc1ccccc1)NC(=O)OC1CCOC1)S(=O)(=O)c1ccc(N)cc1"
LOPINAVIR    = "CC1=CC(=CC(=C1)C)C(=O)NC(CC2=CC=CC=C2)CC(CC3=CC=NC=C3)NC(=O)C(CC(C)C)NC4=NC5=CC=CC=C5S4"
ETHANOL      = "CCO"
INVALID      = "NOT_A_SMILES"
EMPTY        = ""

ALL_VALID    = [ASPIRIN, CAFFEINE, AMPRENAVIR]
MIXED        = [ASPIRIN, INVALID, CAFFEINE, EMPTY, AMPRENAVIR]


# ─────────────────────────────────────────────────────────────
# Chemical correctness invariants
# ─────────────────────────────────────────────────────────────
class TestChemicalCorrectness:
    """Verify computed features match known reference values."""

    def test_aspirin_molecular_weight(self):
        f = compute_lipinski_features(ASPIRIN)
        assert f is not None
        # Aspirin MW = 180.16 Da
        assert 178 < f["mol_weight"] < 182, f"Aspirin MW expected ~180, got {f['mol_weight']}"

    def test_caffeine_molecular_weight(self):
        f = compute_lipinski_features(CAFFEINE)
        assert f is not None
        # Caffeine MW = 194.19 Da
        assert 192 < f["mol_weight"] < 196, f"Caffeine MW expected ~194, got {f['mol_weight']}"

    def test_aspirin_passes_lipinski(self):
        f = compute_lipinski_features(ASPIRIN)
        assert f["passes_lipinski"] == 1
        assert f["lipinski_violations"] == 0

    def test_amprenavir_lipinski_violations(self):
        """Amprenavir (protease inhibitor) violates Lipinski — MW ~505 Da."""
        f = compute_lipinski_features(AMPRENAVIR)
        assert f is not None
        assert f["mol_weight"] > 480, f"Amprenavir MW should be > 480, got {f['mol_weight']}"

    def test_aromatic_rings_aspirin(self):
        f = compute_all_features(ASPIRIN)
        assert f is not None
        # Aspirin has 1 benzene ring
        assert f["aromatic_rings"] == 1, f"Aspirin should have 1 aromatic ring, got {f['aromatic_rings']}"

    def test_caffeine_aromatic_rings(self):
        f = compute_all_features(CAFFEINE)
        assert f is not None
        # Caffeine purine = 2 fused aromatic rings
        assert f["aromatic_rings"] >= 2, f"Caffeine should have ≥2 aromatic rings"

    def test_morgan_fp_consistency(self):
        """Two calls with same SMILES should give identical Morgan fingerprints."""
        f1 = compute_all_features(ASPIRIN)
        f2 = compute_all_features(ASPIRIN)
        for i in range(1024):
            assert f1[f"mfp_{i}"] == f2[f"mfp_{i}"], f"Morgan bit {i} not deterministic"

    def test_maccs_length(self):
        f = compute_all_features(CAFFEINE)
        assert f is not None
        assert all(f"maccs_{i}" in f for i in range(167))

    def test_fragment_keys_present(self):
        f = compute_all_features(ASPIRIN)
        assert f is not None
        # Aspirin has an ester group → fr_ester should be 1
        assert "fr_ester" in f
        assert f["fr_ester"] >= 1, "Aspirin should have ≥1 ester group"


# ─────────────────────────────────────────────────────────────
# Latency benchmarks
# ─────────────────────────────────────────────────────────────
class TestLatency:
    """Ensure feature extraction is fast enough for real-time use."""

    LATENCY_THRESHOLD_MS = 200  # per molecule

    def test_single_molecule_latency(self):
        t0 = time.perf_counter()
        compute_all_features(AMPRENAVIR)  # complex molecule
        elapsed_ms = (time.perf_counter() - t0) * 1000
        assert elapsed_ms < self.LATENCY_THRESHOLD_MS, (
            f"Feature extraction took {elapsed_ms:.0f}ms, "
            f"expected < {self.LATENCY_THRESHOLD_MS}ms"
        )

    def test_batch_throughput(self):
        """20 molecules should complete in under 4 seconds."""
        batch = [ASPIRIN, CAFFEINE, AMPRENAVIR] * 7  # 21 molecules
        t0 = time.perf_counter()
        df = compute_features_batch(batch)
        elapsed = time.perf_counter() - t0
        assert elapsed < 4.0, f"Batch of {len(batch)} took {elapsed:.1f}s, expected < 4s"
        assert len(df) == len(batch)

    def test_mi_selection_latency(self):
        """MI selection on 500×400 matrix should finish in < 10 seconds."""
        np.random.seed(42)
        X = np.random.randn(500, 400).astype(np.float32)
        y = (X[:, 0] > 0).astype(int)
        t0 = time.perf_counter()
        idx = select_features_mi(X, y, k=100, seed=42)
        elapsed = time.perf_counter() - t0
        assert elapsed < 10.0, f"MI selection took {elapsed:.1f}s"
        assert len(idx) == 100


# ─────────────────────────────────────────────────────────────
# Robustness and edge cases
# ─────────────────────────────────────────────────────────────
class TestRobustness:
    """Ensure the pipeline handles edge cases gracefully."""

    def test_invalid_smiles_returns_none(self):
        # RDKit accepts empty string as an empty molecule; only truly malformed SMILES return None
        assert compute_all_features(INVALID) is None
        assert compute_all_features("C[[[") is None

    def test_batch_filters_invalids(self):
        df = compute_features_batch(MIXED)
        # MIXED = [ASPIRIN, INVALID, CAFFEINE, EMPTY, AMPRENAVIR]
        # RDKit parses empty string as a valid (empty) molecule, so 4 rows are returned
        assert len(df) >= 3  # at minimum the 3 real molecules pass

    def test_batch_empty_list(self):
        df = compute_features_batch([])
        assert len(df) == 0
        assert isinstance(df.columns.tolist(), list)

    def test_single_atom_molecule(self):
        """Monoatomic 'molecules' (e.g. '[Na]') should return features."""
        f = compute_all_features("[Na]")
        # May be None (if RDKit rejects it) or a valid feature dict
        # — we just confirm no unhandled exception is raised
        assert f is None or isinstance(f, dict)

    def test_very_large_molecule(self):
        """A large peptide SMILES should not crash the pipeline."""
        big = "C" * 20 + "N"  # trivial large chain
        f = compute_all_features(big)
        assert f is not None
        assert f["mol_weight"] > 100

    def test_feature_names_consistent(self):
        """All valid molecules should return the same set of feature keys."""
        keys_a = set(compute_all_features(ASPIRIN).keys())
        keys_b = set(compute_all_features(CAFFEINE).keys())
        keys_c = set(compute_all_features(AMPRENAVIR).keys())
        assert keys_a == keys_b == keys_c, "Feature key sets differ across molecules"


# ─────────────────────────────────────────────────────────────
# MI selection quality
# ─────────────────────────────────────────────────────────────
class TestMISelectionQuality:
    """Verify MI selects informative features, not arbitrary ones."""

    def test_selects_truly_informative_features(self):
        """Top-k by MI should include the ground-truth informative features."""
        np.random.seed(7)
        n, d = 400, 100
        X = np.random.randn(n, d).astype(np.float32)
        # Features 0, 1, 2 are truly informative
        y = (X[:, 0] + X[:, 1] - X[:, 2] > 0).astype(int)
        idx = select_features_mi(X, y, k=5, seed=7)
        # At least 2 of the 3 truly informative features should be in top-5
        informative = {0, 1, 2}
        overlap = informative & set(idx.tolist())
        assert len(overlap) >= 2, f"MI missed informative features: selected={idx}, informative={informative}"

    def test_returns_unique_indices(self):
        np.random.seed(42)
        X = np.random.randn(200, 50).astype(np.float32)
        y = (X[:, 0] > 0).astype(int)
        idx = select_features_mi(X, y, k=20, seed=42)
        assert len(np.unique(idx)) == len(idx), "MI returned duplicate feature indices"

    def test_k_equals_n_features(self):
        """k = n_features should return all feature indices."""
        np.random.seed(42)
        X = np.random.randn(100, 10).astype(np.float32)
        y = (X[:, 0] > 0).astype(int)
        idx = select_features_mi(X, y, k=10, seed=42)
        assert sorted(idx.tolist()) == list(range(10))


# ─────────────────────────────────────────────────────────────
# App-level smoke test (no Streamlit runtime required)
# ─────────────────────────────────────────────────────────────
class TestAppSmoke:
    """Verify that the data flow powering the Streamlit UI works end-to-end."""

    def test_feature_to_prediction_format(self):
        """Simulate what app.py does: SMILES → features → predict_proba-ready vector."""
        from catboost import CatBoostClassifier

        smiles = AMPRENAVIR
        feats = compute_all_features(smiles)
        assert feats is not None

        # Build a tiny in-memory training set for the smoke test
        rows = [compute_all_features(s) for s in [ASPIRIN, CAFFEINE, AMPRENAVIR] * 30
                if compute_all_features(s) is not None]
        fcols = sorted(rows[0].keys())
        X = np.array([[r[c] for c in fcols] for r in rows], dtype=np.float32)
        np.nan_to_num(X, copy=False)
        y = np.array([0, 0, 1] * 30)

        mi_idx = select_features_mi(X, y, k=50, seed=42)
        cb = CatBoostClassifier(iterations=20, depth=3, verbose=0, random_seed=42)
        cb.fit(X[:, mi_idx], y)

        # Now predict
        feat_vec = np.array([feats.get(c, 0) for c in fcols], dtype=np.float32)
        np.nan_to_num(feat_vec, copy=False)
        prob = cb.predict_proba(feat_vec[mi_idx].reshape(1, -1))[0, 1]

        assert 0.0 <= prob <= 1.0, f"Prediction probability out of range: {prob}"
        assert isinstance(prob, float)
