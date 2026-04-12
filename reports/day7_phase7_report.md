# Phase 7: Testing + Production Pipeline + Polish -- Drug Molecule Property Prediction
**Date:** 2026-04-12
**Session:** 7 of 7
**Researcher:** Anthony Rodrigues

## Objective
Consolidate 6 phases of research into a production-ready codebase: clean training pipeline, inference API, comprehensive tests, and polished documentation.

## What Was Built

### Production Pipeline
1. **`src/train.py`** -- Full training pipeline that trains both GIN+Edge and CatBoost MI-400, saves model artifacts and feature selector. Configurable via `config/config.yaml`.
2. **`src/predict.py`** -- Loads the ensemble and predicts HIV activity for new SMILES strings. Returns probability, Lipinski compliance, and per-model scores. Supports single molecule and batch file input.
3. **`src/evaluate.py`** -- Runs the full evaluation suite on the OGB test set: ROC-AUC, AUPRC, F1, per-model and ensemble metrics, Lipinski subgroup analysis, and latency benchmarking.
4. **`src/feature_engineering.py`** -- Refactored feature extraction into a clean module: `compute_all_features()` for single SMILES, `compute_features_batch()` for batch, `select_features_mi()` for MI-based selection.
5. **`config/config.yaml`** -- All hyperparameters (GIN architecture, CatBoost tuning, ensemble weights, feature config) in one file.

### Test Suite (28 tests, all passing)
| Test File | Tests | Coverage |
|-----------|-------|----------|
| test_data_pipeline.py | 15 | Lipinski features, Morgan FP, MACCS keys, batch processing, invalid SMILES |
| test_model.py | 7 | GIN forward pass, output range, batching, pooling types, gradient flow, param count |
| test_inference.py | 6 | Feature determinism, MI selection, feature completeness, fingerprint binary checks |

### Documentation
- **README.md** -- Added Mermaid architecture diagram, setup instructions, project structure, limitations section, references
- **models/model_card.md** -- Model capabilities, performance, known limitations (Lipinski blind spot), intended use

## Consolidated Project Metrics

| Model | ROC-AUC | Phase | Notes |
|-------|---------|-------|-------|
| GIN+CatBoost Ensemble (0.3/0.7) | **0.8114** | 4 | Champion -- Jaccard error overlap = 0.161 |
| CatBoost MI-400 (Mark best run) | 0.8105 | 3 | MI feature selection was the key CatBoost unlock |
| GIN+Edge (tuned, 64d/3L) | 0.7982 | 4 | BondEncoder was the key GNN unlock (+0.081) |
| CatBoost (tuned, depth=8) | 0.7939 | 4 | Val-test gap widened with tuning |
| GIN+Edge (default) | 0.7860 | 3 | Bond features > virtual node |
| CatBoost (Phase 1 baseline) | 0.7782 | 1 | Auto class weights + combined features |
| RF (Lip+FP combined) | 0.7707 | 1 | First baseline |
| MLP-Domain9 | 0.7670 | 2 | 5K params beats all raw-feature GNNs |
| GIN (raw features) | 0.7053 | 2 | Graph topology alone insufficient |
| OGB SOTA (reference) | 0.8476 | -- | Requires deeper architectures |

## Key Insights Across All 7 Phases

1. **Feature curation > model architecture:** MI-selected 400 features from a 1,302-dim pool (+0.043 AUC) beats any single model architecture change
2. **Bond encoding is the GNN unlock:** +0.081 AUC from BondEncoder alone -- the largest single improvement across all phases
3. **MACCS keys are 2.4x more information-dense per bit than Morgan FP** -- hand-curated substructure keys outperform hashed fingerprints
4. **GIN and CatBoost fail on structurally different molecules** -- CB fails on large polar (MW 491), GIN on small (MW 374). Ensemble rescues 542, hurts zero.
5. **Lipinski-compliant actives are nearly invisible** -- AUC=0.6707 vs 0.8450 for violators. The model learned "bigger = more likely HIV inhibitor" which is historically accurate but a systematic blind spot.
6. **SHAP and LIME disagree 96% of the time (Jaccard=0.042)** -- the model uses different feature pathways per molecule, not capturable by any single global explanation

## Files Created/Modified
- `config/config.yaml` (new)
- `src/feature_engineering.py` (new)
- `src/train.py` (new)
- `src/predict.py` (new)
- `src/evaluate.py` (new)
- `models/model_card.md` (new)
- `tests/__init__.py` (new)
- `tests/test_data_pipeline.py` (new)
- `tests/test_model.py` (new)
- `tests/test_inference.py` (new)
- `README.md` (updated -- architecture diagram, setup, structure, limitations, references)
- `requirements.txt` (updated -- added catboost, pyyaml, optuna, pytest)

## References Used Today
- [1] Google Model Cards -- model card format and ethical considerations structure
- [2] Hu et al., 2020 (OGB) -- OGB AtomEncoder/BondEncoder vocab sizes for test fixture design
- [3] Hugging Face Model Card Guide -- limitations and intended use best practices
