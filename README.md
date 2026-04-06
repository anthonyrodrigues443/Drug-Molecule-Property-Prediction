# Drug Molecule Property Prediction

Predicting HIV drug activity on the ogbg-molhiv dataset (41,127 molecules, 3.5% active) using the OGB scaffold split and ROC-AUC as the primary metric. Benchmarking against the OGB public leaderboard (SOTA: 0.8476 ROC-AUC).

---

## Current Status

**Phase 1 complete** — Baselines established. CatBoost with auto class-weighting achieves **ROC-AUC=0.7782** on combined features (1036), 0.070 below published SOTA. Threshold tuning at 0.59 boosts F1 by +27%. Gap to SOTA requires learned graph representations (GNNs).

---

## Key Findings

1. **CatBoost auto-weighted is the Phase 1 champion** — ROC-AUC=0.7782, beating RF (0.7707) by +0.0075 while maintaining 52% recall
2. **Combined features (Lipinski + FP) beat either alone** — RF: 0.7707 combined vs 0.6937 Lipinski vs 0.7166 FP
3. **Threshold tuning matters more than model choice** — optimal threshold 0.59 (not 0.50) boosts F1 from 0.269 to 0.342 (+27%)
4. **Domain features dominate top importance but FP carries collective weight** — 12/15 top features are domain, but FP bits hold 72.5% of total importance
5. **Class-weighting is a trade-off, not a free lunch** — lower AUC but 3x higher recall; for drug screening, catching actives may justify the AUC cost

---

## Models Compared

**Phase 1:** 15+ experiments across LogReg, RF, XGBoost, LightGBM, and CatBoost with 4 feature sets (Lipinski-12, Morgan FP 1024, combined 1036, graph topology 5), class-weighting strategies, and threshold tuning

---

## Iteration Summary

### Phase 1: Domain Research + Dataset + EDA + Baseline — 2026-04-06

<table>
<tr>
<td valign="top" width="38%">

**Dataset & Standard Baselines:** Selected ogbg-molhiv (41K molecules, 3.5% HIV-active, OGB scaffold split) over smaller alternatives (ESOL, Lipophilicity, AqSolDB). RF with combined features (Lipinski-12 + Morgan FP 1024) achieves ROC-AUC=0.7707. Combined features consistently beat either alone — unlike solubility tasks, HIV activity needs both domain descriptors and structural patterns.<br><br>
**Imbalance & Threshold Tuning:** CatBoost auto-weighted becomes new champion (ROC-AUC=0.7782, recall=0.523). Threshold tuning at 0.59 boosts F1 by +27%. Feature importance reveals 12/15 top features are domain descriptors, but FP bits collectively hold 72.5% of total importance. Graph topology alone reaches 0.70 AUC.

</td>
<td align="center" width="24%">

<img src="results/phase1_roc_pr_curves.png" width="220">

</td>
<td valign="top" width="38%">

**Combined Insight:** The bottleneck is both model family and decision calibration. CatBoost's ordered boosting handles 3.5% imbalance more gracefully than RF/XGBoost (+0.0075 AUC), while threshold tuning at 0.59 extracts +27% F1 without changing the model. Feature importance shows domain features and fingerprints are complementary: domain features rank individually highest, but fingerprints carry 72.5% of collective signal. The 0.070 AUC gap to SOTA confirms tabular models plateau here — closing it requires GNNs.<br><br>
**Surprise:** Threshold tuning (0.50→0.59) boosts F1 more than switching model families. At 3.5% imbalance, the default 0.50 threshold wastes discrimination — the model already ranks well, it just cuts at the wrong point.<br><br>
**Research:** Hu et al., 2020 — OGB benchmark, SOTA 0.8476 requires graph-level representations; Prokhorenkova et al., 2018 — CatBoost ordered boosting handles class imbalance without naive oversampling; He & Garcia, 2009 — moderate imbalance responds to weighting without discrimination collapse.<br><br>
**Best Model So Far:** CatBoost (auto_class_weights, combined 1036 features) — ROC-AUC=0.7782, AUPRC=0.3708, Recall=0.523

</td>
</tr>
</table>
