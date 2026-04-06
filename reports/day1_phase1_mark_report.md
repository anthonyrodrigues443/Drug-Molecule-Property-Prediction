# Phase 1: Class-Imbalance Strategies + Graph Features — Drug Molecule Property Prediction
**Date:** 2026-04-06 | **Session:** 1 of 7 | **Researcher:** Mark Rodrigues

## Objective
At 3.5% positive rate on ogbg-molhiv, is the bottleneck the model or the class-imbalance handling?

## Building on Anthony's Work
**Anthony found:** RF (Combined 1036) = ROC-AUC 0.7707, Combined > individual, 0.077 gap to SOTA.
**My approach:** Class-weighting strategies, CatBoost/LightGBM (new models), graph topology features, threshold tuning, feature importance.
**Combined insight:** CatBoost auto-weight is the new champion (0.7782). Class-weighting is a trade-off. Threshold tuning matters more than model choice. 12 domain features hold 12/15 top importance slots.

## Research & References
1. Hu et al. (2020) — OGB benchmark, ogbg-molhiv SOTA = 0.8476
2. Prokhorenkova et al. (2018) — CatBoost ordered boosting handles imbalance gracefully
3. He & Garcia (2009) — At 3.5% moderate imbalance, weighting helps recall without AUC collapse

## Dataset
| Metric | Value |
|--------|-------|
| Dataset | ogbg-molhiv (OGB) |
| Total | 41,127 molecules |
| Positive rate | 3.51% |
| Split | OGB scaffold (32,901/4,113/4,113) |
| Primary metric | ROC-AUC |

## Head-to-Head Comparison
| Rank | Model | ROC-AUC | AUPRC | Recall |
|------|-------|---------|-------|--------|
| 1 | CatBoost (Combined, auto_weight) | 0.7782 | 0.3708 | 0.523 |
| 2 | CatBoost (Combined, no weight) | 0.7746 | 0.3821 | 0.162 |
| 3 | LightGBM (Combined, no weight) | 0.7732 | 0.3826 | 0.162 |
| 4 | XGBoost (Combined, no weight) | 0.7703 | 0.3956 | 0.169 |
| ref | Anthony RF (Combined) | 0.7707 | 0.3722 | 0.308 |

## Key Findings (after iteration)
1. **CatBoost auto-weight is the new champion** — ROC-AUC 0.7782 (+0.0075 vs Anthony RF)
2. **Threshold tuning boosts F1 by +27%** — optimal threshold is 0.59, not 0.50 (F1: 0.269 -> 0.342)
3. **Domain features dominate top importance** — 12/15 top features are domain, but FP bits collectively hold 72.5% of total importance
4. **Class-weighting is a trade-off** — lower AUC but 3x higher recall
5. **Graph topology alone reaches 0.70 AUC** — real signal but insufficient alone

## Code Changes
- `notebooks/phase1_mark_imbalance_baselines.ipynb` — Full executed notebook with 7 experiments + iteration
- `results/phase1_mark_imbalance_comparison.png` — ROC-AUC + AUPRC comparison
- `results/phase1_mark_threshold_tuning.png` — Threshold sweep plot
- `results/phase1_mark_feature_importance.png` — CatBoost top-15 features
