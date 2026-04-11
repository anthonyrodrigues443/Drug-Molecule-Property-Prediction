# Phase 6 (Mark): LIME + Subgroup SHAP + Feature Group Attribution
**Date:** 2026-04-11
**Session:** 6 of 7
**Researcher:** Mark Rodrigues

## Objective
Anthony (Phase 6) revealed global SHAP + GIN atom saliency: sulfur 2.8x more important than carbon, MACCS substructure keys dominate bulk properties 2.4x. My question was sharper: **Does the global story hold at the individual molecule level?** Three experiments: LIME local explanations, subgroup SHAP by Lipinski compliance, and feature group attribution.

## Building on Anthony's Work
**Anthony found:** SHAP TreeExplainer on CatBoost shows MACCS keys dominate (1.35 total vs 0.56 domain). GIN gradient saliency identifies sulfur as 2.8x more predictive than carbon, consistent with HIV protease pharmacology. Both-wrong molecules cluster near drug-like space boundary (MW 502, TPSA 136).

**My approach:** Anthony asked "what does the model look at globally?" I asked "is that true for each molecule individually?" Used LIME (local perturbation-based) to compare per-molecule explanations against SHAP's global summary. Then split test set into Lipinski-compliant vs. violating to understand Phase 4's recall asymmetry mechanistically. Finally measured AUC for each feature group trained alone to detect SHAP's collinearity inflation.

**Combined insight:** Anthony's SHAP global says MACCS (43% importance for compliant, 38% for violating). My LIME says Morgan (64% of local weight). When you zoom into individual molecules, the feature story is almost entirely different from the population average. The model has learned something more complex than any global summary captures.

## Research References
1. Ribeiro et al. (2016) — "Why Should I Trust You?" LIME: perturb inputs, fit local linear model. https://arxiv.org/abs/1602.04938
2. Lundberg & Lee (2017) — SHAP: unified game-theoretic attribution. https://arxiv.org/abs/1705.07874
3. Lapuschkin et al. (2019) — Unmasking Clever Hans predictors — global XAI can mislead if features are correlated. https://www.nature.com/articles/s41467-019-08987-4
4. Lipinski et al. (1997) — Rule of Five. Many HIV protease inhibitors (saquinavir, ritonavir) violate it (MW 670-721, HBA 11+).

## Dataset
| Metric | Value |
|--------|-------|
| Total molecules | 41,127 |
| Features (full pool) | 1,302 (Lipinski 14 + MACCS 167 + Morgan 1024 + Fr_ 85 + Advanced 12) |
| MI-selected | Top 400 by mutual information |
| Train/Val/Test split | OGB official scaffold split |
| Positive rate (test) | ~3.5% HIV-active |

## Experiments

### Experiment 6.1: LIME Local Explanations
**Hypothesis:** If Anthony's global SHAP (MACCS dominates) is reliable, LIME should rank similar features as top predictors for individual molecules.

**Method:** LimeTabularExplainer with 5,000 perturbation samples on 8 representative test molecules (2 TP, 2 TN, 2 FP, 2 FN). Compare top-5 feature sets against global SHAP top-5.

**Result:** Mean LIME-localSHAP Jaccard = **0.042** (4.2% feature overlap). This is near-zero agreement.

**LIME category breakdown:**
| Category | LIME fraction | SHAP global fraction | Δ |
|----------|--------------|---------------------|---|
| Morgan   | 63.7%        | ~29-30%              | +34% |
| MACCS    | 15.2%        | ~38-44%              | -23% |
| RDKit Fr_| 17.0%        | ~10-11%              | +7% |
| Domain   | 4.0%         | ~16-20%              | -14% |

**Interpretation:** SHAP (global) says MACCS dominates. LIME (local) says Morgan dominates. The disagreement is not measurement noise — it is the same model, same features, different molecules. SHAP collinearity inflates MACCS importance at the population level. But for any individual molecule, Morgan fingerprints often provide the decisive local perturbation signal.

### Experiment 6.2: Subgroup SHAP — Lipinski Violators vs. Compliant
**Hypothesis (from Phase 4):** Violating actives had 2x recall (0.828 vs 0.400). SHAP should show a different mechanism — the model uses different feature pathways for large complex HIV inhibitors.

**Result:**
| Subgroup | AUC | Recall (actives) | Notes |
|---------|-----|-----------------|-------|
| Compliant (0 violations) | 0.6707 | 3.3% | Model nearly fails completely |
| Violating (>=1) | 0.8450 | 54.3% | Model's real strength zone |
| Overall | 0.7713 | — | Masked average |

**SHAP category breakdown by subgroup:**
| Category | Compliant | Violating | Delta |
|----------|-----------|-----------|-------|
| MACCS | 43.8% | 38.5% | -5.3% |
| Morgan | 29.8% | 30.0% | +0.2% |
| Domain | 16.2% | 20.0% | +3.8% |
| RDKit Fr_ | 10.3% | 11.5% | +1.2% |

**Top-10 SHAP overlap:** 8 of 10 features shared between subgroups — the same features matter, but the model fails completely on compliant actives (0.033 recall) despite recognizing which features are important.

**Interpretation:** The model did not learn a separate "small molecule" pathway. It learned one set of rules that works brilliantly for large complex actives (Lipinski violators) and mostly fails for small simple ones. This is a fundamental data distribution issue: the training set has proportionally more Lipinski-violating actives, so the model overfits to those structural patterns.

### Experiment 6.3: Feature Group Attribution — Independent Predictive Power
**Hypothesis:** SHAP shows MACCS dominates (38-44%). But SHAP is affected by feature collinearity — correlated features share credit. True attribution requires training each group alone.

**Result:**
| Group | N features | Test AUC | 
|-------|-----------|----------|
| Domain (14) | 14 | **0.7581** |
| Morgan ECFP4 | 1,024 | 0.7550 |
| MACCS | 167 | 0.7401 |
| RDKit Fr_ | 85 | 0.6979 |
| MI-400 (combined) | 400 | 0.7713 |

**Counterintuitive finding:** 14 domain features (Lipinski properties: MW, logP, HBD, HBA, TPSA, etc.) achieve nearly the same AUC (0.7581) as 1,024 Morgan fingerprint bits (0.7550). But SHAP credits Morgan at only 29% vs domain at 16-20% when combined. When trained alone, domain features are *more* informative per-feature than any other group by a large margin.

**Interpretation:** SHAP's collinearity handling compresses domain importance relative to the high-dimensional fingerprint families. The 14 domain features are largely orthogonal to each other; each carries unique information. The 1,024 Morgan bits are highly redundant; SHAP spreads their credit across many positions.

### Experiment 6.4: LIME vs. SHAP Divergence (Agreement Scores)
**Result:**
- Mean LIME-localSHAP Jaccard: **0.042** (near-zero agreement per molecule)
- Mean localSHAP-globalSHAP Jaccard: **0.137** (low but higher than LIME-SHAP)

**Pattern by prediction type:** Agreement was lowest for FN (false negatives) — the molecules the model gets most wrong also have the most inconsistent explanations across methods.

**Interpretation:** For correct high-confidence predictions (TP, TN), LIME and SHAP agree somewhat more (though still low). For wrong predictions (FP, FN), the explanations scatter completely. This suggests that when the model is wrong, it's using fundamentally different feature pathways than when correct — not just threshold-crossing, but a different decision basis.

## Head-to-Head Results Table
| Rank | Experiment | Metric | Score | Key insight |
|------|-----------|--------|-------|-------------|
| — | Overall CatBoost MI-400 | Test AUC | 0.7713 | Combined features best |
| — | Subgroup: Violating | Test AUC | 0.8450 | Model's true strength |
| — | Subgroup: Compliant | Test AUC | 0.6707 | Model nearly fails |
| — | Domain alone (14 feats) | Test AUC | 0.7581 | 14 beats 1024 Morgan per-feature |
| — | Morgan alone (1024) | Test AUC | 0.7550 | Massive feature count, thin margin |
| — | MACCS alone (167) | Test AUC | 0.7401 | SHAP's "top" group is 3rd place alone |
| — | RDKit Fr_ alone (85) | Test AUC | 0.6979 | Fragment features weakest in isolation |

## Key Findings

1. **The model has 3.3% recall on Lipinski-compliant HIV actives.** Overall 0.6707 AUC for compliant vs 0.8450 for violators. The headline "AUC=0.77" hides a near-complete failure on small drug-like HIV compounds. The model learned to find large, complex, Lipinski-violating inhibitors (like HIV protease inhibitors) and almost nothing about small-molecule activity.

2. **LIME and SHAP agree on only 4% of top features per molecule.** Anthony's SHAP says MACCS dominates (43%). LIME says Morgan dominates (64%). Both are correct summaries of different views of the same model. SHAP captures collinearity-adjusted global credit; LIME captures local perturbation sensitivity. The divergence (Jaccard=0.042) is a warning sign: explainability methods give incompatible answers, and neither should be fully trusted alone.

3. **14 domain features nearly match 1,024 Morgan bits (0.7581 vs 0.7550 AUC).** SHAP credits domain features at only 16-20% of total importance when combined, but when trained alone they outperform Morgan. This is SHAP collinearity compression — the 14 unique domain signals get diluted by the 1,024 redundant fingerprint bits when all are present together.

## What Didn't Work
- LIME interpretation was noisy for low-probability molecules (P~0.5). The local linear approximation in that region fits poorly, which likely inflates the LIME-SHAP disagreement.
- Subgroup SHAP top-10 overlap (8/10 features shared) suggests the model isn't using qualitatively different signals for the two groups — it's the same signals failing differently.

## Next Steps (Phase 7)
- Production pipeline: wrap CatBoost MI-400 in a src/predict.py with SMILES input → HIV probability output
- Inference benchmark: time per prediction (should be <1ms)
- Model card: document the Lipinski-compliance blind spot as a known limitation
- Streamlit app: molecular input + prediction + SHAP explanation + Lipinski compliance check

## References Used Today
- [1] Ribeiro et al. (2016) "Why Should I Trust You?" LIME — https://arxiv.org/abs/1602.04938
- [2] Lundberg & Lee (2017) SHAP — https://arxiv.org/abs/1705.07874
- [3] Lapuschkin et al. (2019) Unmasking Clever Hans — https://www.nature.com/articles/s41467-019-08987-4
- [4] Lipinski et al. (1997) Rule of Five — J. Pharmacol. Biopharm. 23(1):3-25

## Code Changes
- `src/build_phase6_mark_notebook.py` — notebook builder script
- `notebooks/phase6_mark_explainability.ipynb` — executed research notebook
- `results/phase6_mark_results.json` — all Phase 6 Mark metrics
- `results/phase6_mark_lime_explanations.png` — LIME per-molecule 8-panel
- `results/phase6_mark_subgroup_shap.png` — Subgroup SHAP comparison
- `results/phase6_mark_group_attribution.png` — Feature group AUC
- `results/phase6_mark_lime_shap_divergence.png` — Agreement scores heatmap
- `results/phase6_mark_summary.png` — Combined 6-panel summary
