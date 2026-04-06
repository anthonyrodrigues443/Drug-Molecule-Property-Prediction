# Drug Molecule Property Prediction — Experiment Log

**Project:** DL-1 Drug Molecule Property Prediction
**Dataset:** ESOL (MoleculeNet) — 1128 molecules
**Primary Metric:** RMSE (lower is better)
**Published SOTA:** AttentiveFP RMSE=0.584

---

## Phase 1: EDA + Baseline Models (2026-04-06)

| Rank | Model | Feature Set | RMSE | MAE | R² |
|------|-------|-------------|------|-----|----|
| 1 | XGBoost | Lipinski-only | 0.3846 | 0.3189 | 0.8594 |
| 2 | XGBoost | Combined | 0.3977 | 0.3108 | 0.8496 |
| 3 | Random Forest | Combined | 0.4000 | 0.3153 | 0.8479 |
| 4 | Random Forest | Lipinski-only | 0.4152 | 0.3254 | 0.8361 |
| 5 | Ridge Regression | Lipinski-only | 0.5403 | 0.4141 | 0.7224 |
| 6 | Linear Regression | Lipinski-only | 0.5442 | 0.4158 | 0.7185 |
| 7 | XGBoost | Morgan-FP-only | 0.7852 | 0.6404 | 0.4139 |
| 8 | Random Forest | Morgan-FP-only | 0.7976 | 0.6147 | 0.3952 |
| 9 | Mean baseline | none | 1.1201 | nan | 0.0000 |
| 10 | Ridge Regression | Morgan-FP-only | 1.6170 | 1.2644 | -1.4858 |

**Published benchmarks for reference:**
| Paper | Model | RMSE |
|-------|-------|------|
| Duvenaud et al., 2015 | Graph CNN | 0.73 |
| Gilmer et al., 2017 | MPNN | 0.72 |
| Yang et al., 2019 | D-MPNN (Chemprop) | 0.555 |
| Xiong et al., 2020 | AttentiveFP | 0.584 |

