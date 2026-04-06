# Phase 1: Domain Research + Dataset + EDA + Baseline — Drug Molecule Property Prediction
**Date:** 2026-04-06 | **Session:** 1 of 7 | **Researcher:** Anthony Rodrigues

## Objective
Research the best dataset for molecular property prediction, establish baselines, and determine whether domain chemistry features (Lipinski) or structural fingerprints (Morgan) better predict HIV drug activity.

## Research & References
1. [Wu et al. (2018)](https://pmc.ncbi.nlm.nih.gov/articles/PMC5868307/) — MoleculeNet benchmark (1800+ citations). Found: ESOL=1,128, Lipophilicity=4,200 — too small for credible ML.
2. [Hu et al. (2020)](https://arxiv.org/abs/2005.00687) — Open Graph Benchmark. ogbg-molhiv: 41K molecules with public leaderboard, scaffold split.
3. [OGB Leaderboard](https://ogb.stanford.edu/docs/leader_graphprop/) — SOTA: Multi-RF Fusion + Multi-GNN = 0.8476 ROC-AUC. GIN baseline: 0.8273.
4. [Nature Comms (2023)](https://www.nature.com/articles/s41467-023-41948-6) — Systematic study showing scaffold splits expose real generalization gaps.

**Dataset chosen: ogbg-molhiv** over ESOL (1,128), Lipophilicity (4,200), BBBP (2,050) because: 41K molecules, OGB leaderboard, impactful HIV drug prediction task.

## Dataset
| Metric | Value |
|--------|-------|
| Dataset | ogbg-molhiv (OGB) |
| Total molecules | 41,127 (41,120 valid) |
| Positive rate | 3.5% (1,443 active / 39,684 inactive) |
| Split | Scaffold (32,901 train / 4,113 val / 4,113 test) |
| Primary metric | ROC-AUC |
| Published SOTA | 0.8476 (Multi-RF Fusion + Multi-GNN) |

## Head-to-Head Comparison
| Rank | Model | Features | ROC-AUC | F1 | Precision | Recall | Train (s) |
|------|-------|----------|---------|-----|-----------|--------|-----------|
| 1 | Random Forest | Combined (1036) | 0.7707 | 0.406 | 0.597 | 0.308 | 1.1 |
| 2 | XGBoost | Combined | 0.7613 | 0.400 | 0.373 | 0.431 | 1.4 |
| 3 | LogReg | Lipinski (12) | 0.7463 | 0.136 | 0.077 | 0.554 | 0.0 |
| 4 | XGBoost | Morgan FP (1024) | 0.7293 | 0.350 | — | — | — |
| 5 | RF | Morgan FP | 0.7166 | 0.383 | — | — | — |
| 6 | LogReg | Morgan FP | 0.7114 | 0.148 | — | — | — |
| 7 | XGBoost | Lipinski | 0.6957 | 0.212 | — | — | — |
| 8 | RF | Lipinski | 0.6937 | 0.297 | — | — | — |
| — | Majority class | none | 0.500 | 0.000 | — | — | — |
| *ref* | *OGB SOTA* | *Multi-GNN* | *0.8476* | — | — | — | — |

## Key Findings
1. **Combined features (Lipinski + FP) consistently beat either alone** — RF: 0.7707 (combined) vs 0.6937 (Lipinski) vs 0.7166 (FP). Unlike ESOL where Lipinski won, HIV activity prediction needs BOTH domain descriptors and structural patterns.
2. **Best baseline is 0.077 AUC below SOTA** — RF Combined at 0.7707 vs OGB SOTA at 0.8476. The gap is in graph topology — our models can't see bonds/rings directly.
3. **3.5% positive rate creates a severe class imbalance** — LogReg with class_weight='balanced' gets 55% recall but only 7.7% precision. The challenge is precision, not recall.
4. **LogReg on 12 Lipinski features alone hits 0.7463 AUC** — surprisingly strong for just molecular descriptors, showing HIV activity correlates with basic drug-likeness properties.

## What Didn't Work
Lipinski-only features with tree models (RF: 0.6937, XGB: 0.6957) — 12 features aren't enough to discriminate among 41K diverse molecules. The fingerprints add structural specificity that domain descriptors miss.

## Next Steps
Phase 2: GNN architectures (GCN, GAT, GIN) that process molecular graphs directly — they should close the ~0.08 AUC gap by capturing bond connectivity and ring topology.

## References Used Today
- [1] Wu et al. (2018). MoleculeNet. https://pmc.ncbi.nlm.nih.gov/articles/PMC5868307/
- [2] Hu et al. (2020). Open Graph Benchmark. https://arxiv.org/abs/2005.00687
- [3] OGB Leaderboard. https://ogb.stanford.edu/docs/leader_graphprop/
- [4] Fang et al. (2023). Nature Comms. https://www.nature.com/articles/s41467-023-41948-6
