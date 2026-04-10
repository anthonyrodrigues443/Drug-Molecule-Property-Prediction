# Phase 5: Advanced Techniques + Ablation Study — Drug Molecule Property Prediction
**Date:** 2026-04-10
**Session:** 5 of 7
**Researcher:** Anthony Rodrigues

## Objective
Three questions: (1) Which ensemble components matter? (2) Can learned stacking beat simple averaging? (3) Does focal loss help on hard molecules?

## Research & References
1. **Lin et al. 2017** — Focal loss for dense object detection. Adapted for imbalanced molecular classification: down-weights easy negatives, focuses on hard positives
2. **Wolpert 1992** — Stacked generalization. Meta-learner trained on base model predictions as features
3. **Hu et al. 2020** — OGB leaderboard: GIN-VN achieves 0.7707 +/- 0.0149 on ogbg-molhiv

Research influenced experiments: focal loss was chosen specifically because Phase 4 identified 188 molecules that BOTH models get wrong. If these are genuinely hard examples (not noise), focal loss should improve predictions on them by down-weighting the easy 77% majority.

## Dataset
| Metric | Value |
|--------|-------|
| Total molecules | 41,127 |
| Valid (RDKit parse) | 41,120 |
| Test set | 4,111 |
| Positive rate | 3.5% |
| Scaffold split | OGB standard |

## Experiments

### Experiment 5.1: Ablation Study
**Hypothesis:** Each component contributes meaningfully; removing any drops AUC.
**Method:** Remove one component at a time from GIN+CB ensemble (0.3/0.7).
**Result:**

| Component Removed | Ensemble AUC | Delta | Verdict |
|---|---|---|---|
| (Full ensemble baseline) | 0.7832 | -- | BASELINE |
| Remove GIN | 0.7663 | -0.017 | CRITICAL |
| Remove CatBoost | 0.7465 | -0.037 | CRITICAL |
| Remove MI selection | 0.7863 | +0.003 | NOT NEEDED |
| Remove class weights | 0.7606 | -0.023 | CRITICAL |

**Interpretation:** CatBoost carries the most signal (-0.037 when removed). Class weighting is critical (-0.023) — without it, the model defaults to majority-class prediction. MI feature selection is NOT helping in this run (+0.003 without it), suggesting CatBoost's built-in feature handling may already capture the selection benefit. GIN adds +0.017 of complementary topology signal.

### Experiment 5.2: Stacking Meta-Learner
**Hypothesis:** LogReg on [GIN_prob, CB_prob] learns better weights than hand-tuned 0.3/0.7.
**Method:** Train LogReg on validation set predictions, evaluate on test.
**Result:**
- Weighted Average (0.3/0.7): AUC = 0.7832
- LogReg Stacking: AUC = 0.7837 (+0.0005)
- LogReg learned coefficients: GIN=3.781, CB=3.735 (nearly equal!)

**Interpretation:** Stacking is essentially tied with simple averaging. The learned coefficients are nearly equal (3.78 vs 3.74), suggesting the meta-learner sees both models as equally valuable once probabilities are on the logit scale. The 0.3/0.7 weighting works because CatBoost probabilities are better calibrated, not because CatBoost is 2.3x more important.

### Experiment 5.3: Focal Loss GIN
**Hypothesis:** Focal loss (gamma=2, alpha=0.75) helps GIN learn from the 151 "hard" molecules.
**Method:** Replace BCE with focal loss in GIN training, keep everything else the same.
**Result:**
- BCE GIN: 0.7465 → Focal GIN: 0.7629 (+0.016)
- BCE Ensemble: 0.7832 → Focal Ensemble: 0.7742 (-0.009)

**Interpretation:** Focal loss meaningfully improves GIN standalone (+0.016). But it hurts the ensemble (-0.009) because it changes the probability distribution — focal loss pushes predictions away from 0.5, which miscalibrates the probabilities that CatBoost's predictions were tuned to combine with.

### Experiment 5.4: Error Analysis
**Method:** Profile the 151 molecules that both models get wrong.
**Result:**
| Feature | Hard (mean) | Easy (mean) | Ratio |
|---|---|---|---|
| mol_weight | 539 | 332 | 1.6x |
| H-bond donors | 3.2 | 1.1 | 2.8x |
| TPSA | 139 | 65 | 2.1x |
| rotatable_bonds | 5.9 | 3.2 | 1.9x |
| Positive rate | 24.3% | 1.5% | 16x |

**Interpretation:** Hard molecules are large, polar, flexible peptide-like compounds with 16x higher HIV-activity enrichment. These molecules have complex 3D conformational behavior that 2D graph representations cannot capture.

## Head-to-Head Comparison
| Rank | Model | AUC | Notes |
|------|-------|-----|-------|
| 1 | GIN+CB Ensemble (P4) | 0.8114 | Project champion |
| 2 | CatBoost MI-400 (Mark P3) | 0.8105 | Best single model |
| 3 | GIN+Edge tuned (P4) | 0.7982 | Best GNN |
| 4 | LogReg Stacking (P5) | 0.7837 | ~tied with weighted avg |
| 5 | Weighted Avg (P5 this run) | 0.7832 | GNN variance lower |
| 6 | Focal GIN+CB (P5) | 0.7742 | Focal hurts ensemble |

## Key Findings
1. **Class weighting matters more than architecture** — removing auto_class_weights costs -0.023 AUC, more than removing GIN (-0.017)
2. **MI selection may not be necessary** — CatBoost performs equally well on all 1302 features in this run (+0.003 without MI)
3. **Focal loss: individual gain != ensemble gain** — +0.016 for GIN alone, -0.009 for ensemble. Probability calibration matters for combinations
4. **Hard molecules are peptide-like** — large (MW 539), polar (TPSA 139), with 2.8x more H-bond donors. 2D methods hit a ceiling

## Error Analysis
- Both correct: 3,365 (81.9%)
- Both wrong: 151 (3.7%) — large, polar, HIV-active enriched
- GIN only wrong: 482 (11.7%)
- CB only wrong: 113 (2.7%)
- Jaccard overlap: 0.202

## Next Steps
- Phase 6: Explainability (SHAP values for CatBoost, attention maps for GIN)
- Investigate whether 3D conformer features (from RDKit/DimeNet) could capture the hard molecules
- Consider calibrating focal-loss GIN probabilities (Platt scaling) before ensembling

## Code Changes
- src/phase5_advanced_techniques.py (full experiment script)
- src/build_phase5_notebook.py (notebook builder)
- notebooks/phase5_advanced_techniques.ipynb (executed R&D notebook)
- results/phase5_anthony_results.json
- results/phase5_advanced_techniques.png
- results/phase5_ablation.png, phase5_hard_molecules.png
- reports/day5_phase5_report.md
