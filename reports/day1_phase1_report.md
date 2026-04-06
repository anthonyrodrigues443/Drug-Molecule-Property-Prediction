# Phase 1: Domain Research + Dataset + EDA + Baseline — Drug Molecule Property Prediction
**Date:** 2026-04-06 | **Session:** 1 of 7 | **Researcher:** Anthony Rodrigues

## Objective
Research the best dataset and evaluation metric for molecular property prediction, then establish baselines for HIV drug activity classification.

## Research & References
1. [Wu et al. (2018)](https://pmc.ncbi.nlm.nih.gov/articles/PMC5868307/) — MoleculeNet benchmark (1800+ citations). Compared ESOL, FreeSolv, Lipophilicity, BBBP — all too small (<5K).
2. [Hu et al. (2020)](https://arxiv.org/abs/2005.00687) — Open Graph Benchmark. ogbg-molhiv: 41K molecules, scaffold split, public leaderboard.
3. [OGB Leaderboard](https://ogb.stanford.edu/docs/leader_graphprop/) — SOTA: 0.8476 ROC-AUC (Multi-RF Fusion + Multi-GNN).
4. [Saito & Rehmsmeier (2024)](https://www.sciencedirect.com/science/article/pii/S2666389924001090) — ROC-AUC is appropriate for imbalanced data when imbalance doesn't distort score distributions.

## Dataset Selection
| Dataset | Size | Leaderboard | Decision |
|---------|------|-------------|----------|
| ESOL | 1,128 | None | Rejected: too small |
| Lipophilicity | 4,200 | TDC | Rejected: small |
| AqSolDB | 9,982 | TDC | Considered |
| **ogbg-molhiv** | **41,127** | **OGB (20+ entries)** | **Chosen** |

## Metric Selection
| Metric | Source | Decision |
|--------|--------|----------|
| ROC-AUC | OGB leaderboard official metric | **Primary** — measures ranking ability, standard in drug screening |
| AUPRC | Common for imbalanced data | **Secondary** — tracked but not primary (3.5% imbalance not extreme enough per Saito 2024) |
| F1/Precision/Recall | Standard classification | **Secondary** — threshold-dependent, useful for error analysis |

## Dataset
| Metric | Value |
|--------|-------|
| Total molecules | 41,127 (41,120 valid after RDKit) |
| Positive rate | 3.5% (1,443 active / 39,684 inactive) |
| Split | Scaffold (32,901 train / 4,113 val / 4,113 test) |
| Primary metric | ROC-AUC |
| Published SOTA | 0.8476 |

## Head-to-Head Comparison (ranked by ROC-AUC)
| Rank | Model | Features | ROC-AUC | AUPRC | F1 |
|------|-------|----------|---------|-------|-----|
| 1 | Random Forest | Combined (1036) | 0.7707 | 0.3722 | 0.406 |
| 2 | XGBoost | Combined | 0.7613 | — | 0.400 |
| 3 | LogReg | Lipinski (12) | 0.7463 | — | 0.136 |
| 4 | XGBoost | Morgan FP (1024) | 0.7293 | — | 0.350 |
| 5 | RF | Morgan FP | 0.7166 | — | 0.383 |
| — | *OGB SOTA* | *Multi-GNN* | *0.8476* | — | — |

## Key Findings
1. **Combined features beat either alone** — RF: 0.7707 (combined) vs 0.6937 (Lipinski) vs 0.7166 (FP)
2. **Gap to SOTA is 0.077 AUC** — the missing signal is graph topology (bond connectivity, ring structure)
3. **3.5% positive rate creates a precision challenge** — LogReg gets 55% recall but only 7.7% precision
4. **LogReg on 12 Lipinski features hits 0.7463 AUC** — strong for just molecular descriptors

## Next Steps
Phase 2: GNN architectures (GCN, GAT, GIN) to close the 0.077 AUC gap.
