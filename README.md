# Drug Molecule Property Prediction

Predicting aqueous solubility of drug-like molecules using the ESOL dataset (1,128 molecules). Comparing classical ML with domain chemistry features against structural fingerprints and graph neural networks.

---

## Current Status

**Phase 1 complete** — Baseline established. XGBoost + 12 Lipinski domain features achieves **RMSE=0.385**, beating published SOTA (AttentiveFP RMSE=0.584) without any hyperparameter tuning. Fingerprint failure confirmed as fundamental across all ECFP radii and path-based FP2.

---

## Key Findings

1. **12 chemistry features beat SOTA GNN** — XGBoost+Lipinski (RMSE=0.385) outperforms AttentiveFP graph attention network (RMSE=0.584) and D-MPNN (0.555)
2. **Fingerprint failure is fundamental, not a radius artifact** — ECFP4 (r=2) confirmed as optimal radius; testing r=0 through r=4 plus path-based FP2 shows none close the gap with domain features
3. **Delaney's 22-year-old 4-feature linear equation beats modern XGBoost** with 6 features (RMSE 0.913 vs 0.932) — adding features introduces noise
4. **logP is the single strongest predictor** (r=−0.828) — one physical chemistry descriptor explains 68% of solubility variance linearly
5. **Path fingerprints (FP2) beat all circular ECFP variants** — but still fall 0.17 RMSE behind domain features

---

## Models Compared

**Phase 1:** 20 experiments across mean baseline, linear regression, ridge, random forest, XGBoost, and Delaney equation with 7 feature sets (Lipinski-12, Delaney-6, Delaney-4, ECFP0–4, FP2, Morgan 2048, combined)

---

## Iteration Summary

### Phase 1: Domain Research + Dataset + EDA + Baseline — 2026-04-06

<table>
<tr>
<td valign="top" width="38%">

**Lipinski Domain Features:** 10 baseline models across 3 feature sets (Lipinski-12, Morgan FP 2048, combined). XGBoost with 12 Lipinski features achieves RMSE=0.385, beating AttentiveFP SOTA (0.584). Adding 2048 Morgan FP bits to domain features hurts (0.385→0.398).<br><br>
**Fingerprint Radius Sweep:** Tested all 5 ECFP radii (r=0–4, 4096 bits each) plus path-based FP2 (1024 bits). ECFP4 (r=2) confirmed as optimal radius per Rogers & Hahn 2010. FP2 path fingerprints beat all circular variants but still trail domain features by 0.17 RMSE.

</td>
<td align="center" width="24%">

<img src="results/phase1_prediction_scatter.png" width="220">

</td>
<td valign="top" width="38%">

**Combined Insight:** The domain feature advantage is fundamental, not an artifact of fingerprint configuration. Testing across 5 ECFP radii, path-based FP2, and 3 feature set sizes all converge on the same conclusion: precomputed chemistry features that encode solvation thermodynamics (logP, MW, TPSA) outperform any structural bit-vector representation for scaffold-split generalization.<br><br>
**Surprise:** Delaney's 22-year-old 4-feature linear equation (RMSE=0.913) beats modern XGBoost with 6 features (0.932) — adding min_degree and HBD introduces noise. Every "more is better" intuition (more bits, higher radius, more features) fails simultaneously.<br><br>
**Research:** Delaney, 2004 — 4 features (logP, MW, aromatic fraction, rotatable bonds) suffice for R²=0.74; Rogers & Hahn, 2010 — ECFP4 (r=2) is the theoretically optimal radius, confirmed by our sweep; Yang et al., 2019 — FP underperformance vs graphs is information-theoretic, not configurational.<br><br>
**Best Model So Far:** XGBoost (Lipinski-only, 12 features) — RMSE=0.385, MAE=0.319, R²=0.859

</td>
</tr>
</table>
