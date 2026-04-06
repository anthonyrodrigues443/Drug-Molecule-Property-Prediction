# Phase 1: Domain Research + Dataset + EDA + Baseline — Drug Molecule Property Prediction
**Date:** 2026-04-06 | **Session:** 1 of 7 | **Researcher:** Anthony Rodrigues

## Objective
Establish research foundation for drug aqueous solubility prediction (ESOL dataset). Test whether domain chemistry features outperform structural fingerprints.

## Research & References
1. **Delaney (2004)** — Original ESOL dataset; logP + MW + aromaticity + rotatable bonds → R²=0.74
2. **Xiong et al. (2020)** — AttentiveFP GNN, RMSE=0.584 on ESOL scaffold split
3. **Yang et al. (2019)** — D-MPNN/Chemprop, RMSE=0.555; Morgan FP underperforms graphs on scaffold splits

## Dataset
| Metric | Value |
|--------|-------|
| Dataset | ESOL (Delaney 2004) via MoleculeNet |
| Total molecules | 1,128 |
| Split | Scaffold (902 train / 113 val / 113 test) |
| Target | log(mol/L) aqueous solubility |
| Primary metric | RMSE (lower is better) |
| Lipinski Ro5 compliance | 90.2% |
| Unique Murcko scaffolds | 269 |

## Head-to-Head Comparison
| Rank | Model | Features | RMSE | MAE | R² |
|------|-------|----------|------|-----|-----|
| 1 | XGBoost | Lipinski-only (12) | 0.385 | 0.319 | 0.859 |
| 2 | XGBoost | Combined (2060) | 0.398 | 0.311 | 0.850 |
| 3 | Random Forest | Combined | 0.400 | 0.315 | 0.848 |
| 4 | Random Forest | Lipinski-only | 0.415 | 0.325 | 0.836 |
| 5 | Ridge | Lipinski-only | 0.540 | 0.414 | 0.722 |
| *ref* | *AttentiveFP (published)* | *graph* | *0.584* | — | — |
| 6 | XGBoost | Morgan FP (2048) | 0.785 | 0.640 | 0.414 |
| 7 | Random Forest | Morgan FP (2048) | 0.798 | 0.615 | 0.395 |
| 8 | Mean baseline | none | 1.120 | — | 0.000 |
| 9 | Ridge | Morgan FP (2048) | 1.617 | 1.264 | -1.486 |

## Key Findings
1. **12 Lipinski features beat 2048 Morgan FP by ~2x RMSE** (0.385 vs 0.785) — domain chemistry knowledge compressed into a handful of features outperforms structural similarity
2. **XGBoost+Lipinski beats published GNN SOTA** (0.385 vs AttentiveFP 0.584) at baseline without tuning
3. **logP is the strongest single predictor** (r=-0.828 with solubility)
4. **Adding fingerprints to domain features slightly hurts** (0.385→0.398) — noisy sparse bits dilute signal

## Next Steps
Phase 2: GNN architectures (GCN, GAT) — can graph topology close the gap vs Lipinski-only XGBoost?
