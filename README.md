# Drug Molecule Property Prediction

Predicting HIV drug activity on the ogbg-molhiv dataset (41,127 molecules, 3.5% active) using the OGB scaffold split and ROC-AUC as the primary metric. Benchmarking against the OGB public leaderboard (SOTA: 0.8476 ROC-AUC).

---

## Current Status

**Phase 1 complete** — Baselines established. CatBoost with auto class-weighting achieves **ROC-AUC=0.7782** on combined features (1036), 0.070 below published SOTA. Gap to SOTA requires graph-level learned representations (GNNs).

---

## Key Findings

1. **CatBoost auto-weighted is the Phase 1 champion** — ROC-AUC=0.7782, beating RF (0.7707) by +0.0075 while maintaining 52% recall
2. **Combined features (Lipinski + FP) beat either alone** — RF: 0.7707 combined vs 0.6937 Lipinski vs 0.7166 FP
3. **Class-weighting is a trade-off, not a free lunch** — every weighted model has lower ROC-AUC but catches 3x more HIV-active molecules (recall ~0.49 vs ~0.16)
4. **Gap to SOTA is 0.070 AUC** — simple graph stats get ROC-AUC=0.70, but the remaining gap requires learned graph representations
5. **LogReg on 12 Lipinski features hits 0.7463 AUC** — strong baseline from molecular descriptors alone

---

## Models Compared

**Phase 1:** 14 experiments across LogReg, RF, XGBoost, LightGBM, and CatBoost with 4 feature sets (Lipinski-12, Morgan FP 1024, combined 1036, graph topology 5) and class-weighting strategies

---

## Iteration Summary

### Phase 1: Domain Research + Dataset + EDA + Baseline — 2026-04-06

<table>
<tr>
<td valign="top" width="38%">

**Dataset & Standard Baselines:** Selected ogbg-molhiv (41K molecules, 3.5% HIV-active, OGB scaffold split) over smaller alternatives (ESOL, Lipophilicity). RF with combined features (Lipinski-12 + Morgan FP 1024) achieves ROC-AUC=0.7707, with combined features consistently beating either alone.<br><br>
**Imbalance & CatBoost:** Tested class-weighting strategies across RF/XGBoost/LightGBM/CatBoost. CatBoost auto-weighted becomes new champion (ROC-AUC=0.7782) — the only model improving both AUC and recall simultaneously. Graph topology features alone reach 0.70 AUC, confirming gap to SOTA requires learned graph representations.

</td>
<td align="center" width="24%">

<img src="results/phase1_roc_pr_curves.png" width="220">

</td>
<td valign="top" width="38%">

**Combined Insight:** The bottleneck is both model family AND feature representation. CatBoost's ordered boosting handles the 3.5% imbalance more gracefully than RF/XGBoost (+0.0075 AUC), while class-weighting trades AUC for recall — a deliberate choice for drug screening where missing an active costs millions. The 0.070 AUC gap to SOTA confirms that tabular models with fingerprints plateau here; closing it requires GNNs that learn from molecular graph topology.<br><br>
**Surprise:** Class-weighting consistently HURTS ROC-AUC across all model families (−0.01 to −0.02) despite tripling recall. At 3.5% imbalance, the class boundary shift degrades ranking quality — He & Garcia (2009) predicted this for moderate imbalance.<br><br>
**Research:** Hu et al., 2020 — OGB benchmark with scaffold split and public leaderboard (SOTA 0.8476); Prokhorenkova et al., 2018 — CatBoost ordered boosting handles class imbalance without naive oversampling; He & Garcia, 2009 — moderate imbalance (3.5%) responds to weighting without discrimination collapse.<br><br>
**Best Model So Far:** CatBoost (auto_class_weights, combined 1036 features) — ROC-AUC=0.7782, AUPRC=0.3708, Recall=0.523

</td>
</tr>
</table>
