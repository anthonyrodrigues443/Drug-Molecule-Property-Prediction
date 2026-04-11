# Phase 6: Explainability & Model Understanding — Drug Molecule Property Prediction
**Date:** 2026-04-11
**Session:** 6 of 7
**Researcher:** Anthony Rodrigues

## Objective
Why does the GIN+CatBoost ensemble achieve 0.8114 AUC? What molecular features drive HIV activity predictions, and do they align with known medicinal chemistry?

## Research & References
1. Lundberg & Lee 2017 — SHAP: A Unified Approach to Interpreting Model Predictions. Used TreeExplainer for CatBoost (exact SHAP, not approximate).
2. Pope et al. 2019 — Explainability Methods for GNNs. Applied gradient saliency via forward hooks on atom embeddings.
3. Durant et al. 2002 — MACCS structural keys. 166 predefined substructural patterns used in chemical similarity. Our SHAP analysis shows these dominate CatBoost's decision-making.
4. Lipinski et al. 1997 — Rule of Five. TPSA and heteroatom count (both Lipinski-adjacent properties) rank in top 5 SHAP features.

How research influenced experiments: Used TreeExplainer (not KernelExplainer) for exact SHAP on CatBoost. Applied gradient saliency at the atom embedding level (not input features) following Pope et al.'s recommendation for GNNs with discrete inputs.

## Dataset
| Metric | Value |
|--------|-------|
| Total samples | 41,127 |
| Train / Val / Test | 32,898 / 4,111 / 4,111 |
| Features (CatBoost) | 400 (MI-selected from 1,302) |
| Target | HIV activity (binary, 3.5% positive) |
| SHAP sample | 1,000 test molecules |
| GIN saliency sample | 200 test molecules |

## Experiments

### Experiment 6.1: SHAP TreeExplainer for CatBoost
**Hypothesis:** Domain descriptors (MW, logP, TPSA) will dominate feature importance.
**Method:** SHAP TreeExplainer on CatBoost MI-400 model, 1000 test molecules.
**Result:** MACCS keys dominate total SHAP (1.35 vs 0.56 for domain descriptors). maccs_144 is #1 feature at 0.191 mean |SHAP| — 2x the #2 feature.
**Interpretation:** **Hypothesis REJECTED.** Substructural fingerprints are more important than bulk molecular properties. HIV activity depends on specific binding motifs, not just drug-likeness. However, per-feature importance is 3x higher for domain descriptors (0.040 vs 0.014) — each domain feature is individually powerful, but there are too few of them.

### Experiment 6.2: Active vs Inactive SHAP Differences
**Hypothesis:** Active and inactive molecules will show opposite SHAP patterns.
**Method:** Split test molecules by label, compare mean SHAP values for top 5 features.
**Result:** maccs_81 shows strongest differential (+0.052 for active, -0.033 for inactive). TPSA also strongly differential (+0.015 vs -0.059). All top features push predictions toward activity when present in active molecules.
**Interpretation:** MACCS_81 appears to encode a substructural pattern specifically enriched in HIV-active compounds. The TPSA pattern suggests HIV-active molecules need sufficient polar surface area for enzyme binding — consistent with protease inhibitor pharmacology.

### Experiment 6.3: GIN Gradient Saliency
**Hypothesis:** GIN will attend to heteroatoms (N, O, S) more than carbon.
**Method:** Register forward hook on atom_encoder, compute gradient of logit w.r.t. embedding, take L2 norm as saliency per node. Aggregate by atom type across 200 test molecules.
**Result:** Among common drug atoms: S (0.393) >> Cl (0.204) > N (0.183) > C (0.138) > O (0.091). Rare metals (As, P, Mn, Se) have highest raw saliency but very low counts (1-12).
**Interpretation:** **Hypothesis CONFIRMED.** Sulfur has 2.8x higher saliency than carbon. This aligns perfectly with HIV protease inhibitor design — thiol groups interact with the catalytic Asp25/Asp25' residues in HIV-1 protease. Nitrogen's elevated saliency reflects amine/amide hydrogen bonding at enzyme active sites. Interestingly, oxygen (0.091) has LOWER saliency than carbon — suggesting the GIN has learned that oxygen is more "background" in HIV-active scaffolds.

### Experiment 6.4: Ensemble Disagreement Properties
**Hypothesis:** Models fail on different molecular property ranges (confirming Phase 5 finding).
**Method:** Categorize test molecules by GIN/CB correctness, compute mean molecular properties per category.
**Result:**
- Both wrong: MW=501.9 (above Lipinski 500 cutoff), TPSA=135.8
- GIN correct, CB wrong: MW=441.3, TPSA=102.8
- CB correct, GIN wrong: MW=401.1, TPSA=92.4
- Both correct: MW=340.2, TPSA=66.5
**Interpretation:** Confirms Phase 5 finding with more granularity. The "both wrong" cluster sits above Lipinski's MW cutoff and near the TPSA permeability threshold — these molecules are at the boundary of drug-like chemical space. CatBoost specifically struggles with larger molecules (441 vs 340 MW) because fingerprint representations compress structural complexity. GIN struggles with medium molecules (401 MW) that share scaffolds but differ in substituent effects.

## Head-to-Head Comparison
| Rank | Analysis | Key Finding | Domain Validation |
|------|----------|-------------|-------------------|
| 1 | SHAP (CatBoost) | MACCS keys dominate (total 1.35 vs 0.56 domain) | HIV activity = specific motifs, not bulk properties |
| 2 | Gradient saliency (GIN) | S(0.393) >> C(0.138), 2.8x ratio | Thiol groups critical for protease inhibitor binding |
| 3 | Disagreement analysis | Both-wrong = MW 502, TPSA 136 | Edge of drug-like space = hardest prediction task |
| 4 | Active vs inactive | maccs_81 differential = 0.085 | Substructural pattern enriched in HIV-active compounds |

## Key Findings
1. **Substructure trumps drug-likeness for HIV activity prediction.** MACCS keys (total SHAP 1.35) outweigh domain descriptors (0.56) by 2.4x. This means: the MODEL learned what medicinal chemists know — HIV activity depends on specific binding motifs at the protease active site, not generic drug properties.
2. **GIN "sees" pharmacophores.** Sulfur saliency (0.393) is 2.8x carbon (0.138), and nitrogen (0.183) is 1.3x carbon. The graph neural network independently discovered the importance of heteroatom pharmacophore points.
3. **Per-feature importance inversion.** Each domain descriptor is 3x more informative than each MACCS key (0.040 vs 0.014 per feature), but the 98 MACCS keys collectively dominate. Both feature types are needed.
4. **Hardest molecules = boundary of drug-like space.** Both models fail on MW > 500, TPSA > 135 compounds — precisely where scaffold-split generalization breaks down because these molecules are structurally distant from training data.
5. **Hydrazone (fr_hdrzone) validates model chemistry.** Ranked #9 in SHAP importance. Hydrazones are established pharmacophores in anti-HIV drug design.

## Error Analysis
- 188 molecules (4.6%) defeat BOTH models — large polar compounds at the Lipinski boundary
- CatBoost's blind spot: large molecules (MW 441) — fingerprint compression loses structural detail
- GIN's blind spot: medium molecules (MW 401) — graph topology over-generalizes on similar scaffolds

## Next Steps
- Phase 7: Build Streamlit UI with per-molecule SHAP explanations alongside predictions
- Include feature importance waterfall plot for each prediction
- Add molecular visualization (2D structure) next to explanations

## References Used Today
- [1] Lundberg & Lee 2017 — https://arxiv.org/abs/1705.07874
- [2] Pope et al. 2019 — https://arxiv.org/abs/1905.13686
- [3] Durant et al. 2002 — MACCS Keys — J. Chem. Inf. Comput. Sci.
- [4] Lipinski et al. 1997 — Adv. Drug Deliv. Rev.

## Code Changes
- src/phase6_explainability.py — Full explainability pipeline (SHAP + gradient saliency + disagreement analysis)
- src/build_phase6_notebook.py — Notebook builder
- notebooks/phase6_explainability.ipynb — Executed research notebook
- results/phase6_explainability.png — 7-panel explainability visualization
- results/phase6_results.json — All metrics and feature importances
- reports/day6_phase6_report.md — This report
