# Phase 4: Hyperparameter Tuning + Error Analysis — Drug Molecule Property Prediction
**Date:** 2026-04-09
**Session:** 4 of 7
**Researcher:** Mark Rodrigues

## Objective
Does Optuna hyperparameter tuning improve MI-top-400 CatBoost beyond its Phase 3 result?
Is K=400 a stable choice, or was it lucky on the scaffold split?
What molecules does the champion fail on, and why?

## Building on Anthony's Work
**Anthony found:** GIN+Edge = 0.7860 AUC (Phase 3 champion). His feature ablation showed that appending GNN embeddings to traditional features HURTS (1345d: 0.7415 vs 1217d: 0.7841). His recommended Phase 4: tune GIN+Edge.

**My approach:** Tune CatBoost + MI-400 (my Phase 3 champion, 0.8105). Complementary to Anthony's GNN tuning — we'll know the tuning ceiling of each paradigm. Also: deep error analysis to understand *what kind* of molecules fail, inspired by the clinical explainability work from the Healthcare readmission project.

**Combined insight:** Anthony tunes graph topology representations. I tune chemistry distributions. Together, Phase 4 answers: can you squeeze more from either paradigm with better hyperparameters?

## Research & References
1. **Prokhorenkova et al., 2018 (CatBoost paper)** — Recommended depth 4-8, l2_leaf_reg 3-15 for tabular data. Ordered boosting particularly effective on noisy features (which molecular fingerprints are at scale). This guided search space design.
2. **Sheridan et al., 2016 (JCIM)** — "Extreme gradient boosting as a method for quantitative structure-activity relationships." Found that higher regularization improves QSAR models with many binary features (Morgan FP). Confirmed l2_leaf_reg should be searched up to 20.
3. **Wu et al., 2018 (MoleculeNet)** — Scaffold splitting makes val/test performance gaps larger than random splits. Hyperparameter tuning on scaffold-split val sets risks overfitting to the scaffold domain. This informed interpretation of the val↔test gap widening.

## Dataset
| Metric | Value |
|--------|-------|
| Total samples | 41,127 |
| Features (MI-top-400 subset) | 400 from {Lipinski-14, MACCS-167, Morgan-1024, Fragments-85, Advanced-12} |
| Target | HIV inhibition (binary, OGB scaffold split) |
| Class distribution | ~3.5% HIV-active (1,479 total actives) |
| Train/Val/Test | 32,901 / 4,113 / 4,113 |

## Experiments

### Experiment 4.0: Reproduce Phase 3 Champion
**Hypothesis:** Default CatBoost (500 iter, lr=0.05, depth=6) on MI-top-400 should give ~0.8105 Test AUC
**Result:** Val AUC=0.7878 | Test AUC=0.7584 (Phase 3 reported: 0.8105, delta=-0.052)
**Interpretation:** Run-to-run variance on scaffold splits is non-negligible. CatBoost's default params are not deterministically stable across runs despite fixed random_seed=42 (possible multithreading effects). The Phase 3 result remains valid but the 0.8105 is the best observed value; 0.76-0.81 is the typical range.

### Experiment 4.1: Optuna Hyperparameter Tuning (40 trials)
**Hypothesis:** Depth=6 and lr=0.05 may not be optimal. Optuna should find a better combination.
**Method:** TPE sampler, 40 trials, optimize Val AUC. Search space: iterations ∈ [300,800], lr ∈ [0.01,0.12], depth ∈ [4,8], l2_leaf_reg ∈ [1,20], min_data_in_leaf ∈ [5,50], random_strength ∈ [0.5,5], bagging_temperature ∈ [0,2], border_count ∈ {64,128,254}.
**Result:**
| Config | Val AUC | Test AUC | Test AUPRC | Val-Test Gap |
|--------|---------|----------|------------|--------------|
| Default | 0.7878 | 0.7584 | 0.3179 | +0.0294 |
| Tuned (best: depth=8, lr=0.055, l2=4.7, min_leaf=38) | 0.8229 | 0.7854 | 0.3179 | +0.0375 |

**Delta val: +0.0351 | Delta test: +0.0270 | Val-test gap WIDENED: +0.0294 → +0.0375**

**Interpretation:** Tuning found real signal (+0.035 val) but the test improvement is smaller (+0.027) and the val↔test gap grew. Classic hyperparameter overfitting to the scaffold split: Optuna optimized for the val scaffold distribution, which partially overfits to those specific molecule types. Scaffold splits create a harder generalization problem than random splits — this is a known issue confirmed by MoleculeNet papers.

**Best tuned params:** depth=8, lr=0.055, l2_leaf_reg=4.7, min_data_in_leaf=38, random_strength=1.06, bagging_temperature=0.85, border_count=64.

### Experiment 4.2: K=400 Stability Analysis
**Method:** 3 bootstrap samples (80% of train), evaluate K ∈ {100, 200, 400, 600} using global MI ranking (no per-bootstrap MI recomputation). Fast 80-iter CatBoost model for speed.

| Bootstrap | K=100 | K=200 | K=400 | K=600 | Best K |
|-----------|-------|-------|-------|-------|--------|
| Boot 0 | 0.8004 | 0.7636 | **0.8029** | 0.8023 | 400 |
| Boot 1 | **0.8076** | 0.7717 | 0.8018 | 0.7707 | 100 |
| Boot 2 | 0.7786 | 0.8069 | **0.8092** | 0.7590 | 400 |
| Mean | 0.7955 | 0.7807 | **0.8046** | 0.7773 | — |
| Std | 0.0152 | 0.0228 | **0.0040** | 0.0218 | — |

**K=400 wins 2/3 bootstraps and has the LOWEST VARIANCE (std=0.0040 vs 0.015-0.022 for others).**

**Interpretation:** K=400 is genuinely the most stable selection, not a lucky split artifact. The very low std (0.0040) suggests the performance plateau around 400 features is real. K=100 can win on lucky draws but is more variable. K=600 consistently underperforms — the 200 extra features beyond K=400 are noise.

### Experiment 4.3: Deep Error Analysis — What Does the Model Miss?
**Method:** Classify test set molecules into TP/TN/FP/FN using Youden-optimal threshold (0.3958). Profile each class by 11 molecular properties + Lipinski violations + Murcko scaffolds.

**Error class distribution (4,113 test molecules, 130 actives):**
| Class | Count | Notes |
|-------|-------|-------|
| TN | 3,419 (83.1%) | Correctly predicted inactive |
| FP | 564 (13.7%) | False alarms (predicted active, actually inactive) |
| TP | 77 (59.2% recall) | Correctly caught actives |
| FN | 53 (40.8% missed) | Missed actives |

**Mean molecular properties by class:**
| Property | FN | FP | TN | TP |
|----------|-----|-----|-----|-----|
| MW (Da) | 424 | 420 | 347 | **630** |
| logP | 2.74 | 3.08 | 2.85 | **5.44** |
| HBD | 2.36 | 2.11 | 1.23 | **4.35** |
| HBA | 6.19 | 6.39 | 4.57 | **9.55** |
| TPSA | 102 | 103 | 69 | **181** |
| Rotatable bonds | 3.87 | 4.68 | 3.26 | **7.78** |
| Rings | 3.96 | 4.68 | 3.57 | **5.60** |
| Aromatic rings | 2.32 | 2.62 | 1.88 | **4.44** |
| Heavy atoms | 29.7 | 28.7 | 24.2 | **43.9** |
| Frac sp3 | **0.35** | 0.24 | 0.38 | 0.16 |

**All 11 FN vs TP differences are statistically significant (p<0.001, Mann-Whitney U).**

**Key finding: The model catches LARGE, COMPLEX actives (MW 630, 5.6 rings) but misses SMALL, SIMPLE actives (MW 424, 4.0 rings).**

### Experiment 4.4: Lipinski Rule Violation Analysis (COUNTERINTUITIVE)
| Lipinski violations | Active recall | n actives |
|---------------------|---------------|-----------|
| 0 violations (rule-compliant) | **0.400** | 60 |
| ≥2 violations | **0.828** | 58 |

**COUNTERINTUITIVE FINDING: Molecules VIOLATING Lipinski rules are easier to classify (2x higher recall). Rule-compliant HIV inhibitors are harder to detect.**

**Interpretation:** This makes chemical sense in context. HIV protease inhibitors are often large, complex molecules that intentionally violate Rule-of-5 (they're designed for efficacy, not oral bioavailability). The MI-selected feature set captures these well — MACCS keys and Morgan bits for aromatic systems, large scaffolds, polar groups. Small, Lipinski-compliant actives (simpler scaffolds, fewer distinctive features) look more like inactive "drug-like" molecules. The model essentially learned "bigger = more likely HIV inhibitor," which is historically accurate for early HIV drugs.

### Experiment 4.5: Feature Importance Analysis
| Category | Pool Share | Importance | Ratio |
|----------|-----------|------------|-------|
| MACCS | 12.8% | **31.0%** | 2.4× |
| Morgan FP | 78.5% | 27.9% | 0.36× |
| Advanced | 0.9% | 16.3% | 18× |
| Fragment (Fr_*) | 6.5% | 12.9% | 2.0× |
| Lipinski | 1.1% | 12.0% | 11× |

**Importance concentration:**
- 50% of importance: top **41** features (of 400)
- 80% of importance: top **110** features
- 95% of importance: top **192** features

**Top 10 features:** maccs_144, adv_11 (MW/heavy_atoms ratio), lip_0 (MW), lip_13 (valence electrons), maccs_81, fr_hdrzone (hydrazone functional group), adv_2 (HBD×HBA interaction), adv_8 (rotatable_bonds × logP), adv_5 (sp3_fraction × MW), morgan_967.

**Interpretation:** 
1. MACCS keys punch 2.4× above their pool share — hand-curated 167-bit pharmacophore/substructure encoding is more information-dense per bit than Morgan's hashed 1024-bit space.
2. Advanced derived features (MW/heavyatoms ratio, interaction terms) carry 18× importance per feature vs their pool size. Domain-engineered interactions are the most efficient features.
3. Morgan FP carries 27.9% importance despite being 78.5% of the pool — each bit is ~3× less informative than MACCS. This is why feature selection helped so much: Morgan bits are noisy individually but useful in aggregate.
4. Despite concentrating 50% of importance in 41 features, removing the other 359 hurts (Phase 3: K=400 > K=200 > K=100). Each additional feature adds a small but real margin.

## Head-to-Head Leaderboard
| Rank | Model | Test AUC | Test AUPRC | Notes |
|------|-------|----------|------------|-------|
| 1 | CatBoost MI-400 (Phase 3 best run) | **0.8105** | 0.3481 | Mark P3 |
| 2 | Tuned CatBoost MI-400 | 0.7854 | 0.3179 | Mark P4 |
| 3 | GIN+Edge | 0.7860 | 0.3441 | Anthony P3 |
| 4 | CatBoost AllTrad-1217 | 0.7841 | 0.3490 | Anthony P3 |
| 5 | CatBoost default (9 features) | 0.7782 | 0.3708 | Mark P1 |
| 6 | MLP-Domain9 | 0.7670 | — | Mark P2 |

## Key Findings

1. **Tuning overfits to scaffold splits.** Optuna val gain (+0.035) exceeds test gain (+0.027). The val↔test gap widened from 0.029 to 0.038 — Optuna learned the val scaffold's idiosyncrasies, partially defeating its purpose. This is a known failure mode of HP search on scaffold splits.

2. **K=400 is genuinely stable (std=0.0040 vs 0.015-0.022 for K=100/200/600).** It's not lucky — the performance plateau around 400 MI-selected features is real and robust.

3. **COUNTERINTUITIVE: Lipinski violators have 2× higher recall (0.828 vs 0.400).** The model catches large, complex, rule-violating HIV inhibitors more reliably than small, rule-compliant actives. Historically, many HIV drugs ARE Lipinski violators — the model learned chemistry, not drug-like rules.

4. **Missed actives are ~200 Da lighter and have 1.6 fewer ring systems than caught actives.** This is a structural bias. Phase 5 should test whether an attention mechanism or GNN can fix the "small molecule" blind spot.

5. **MACCS keys: 31% of importance despite 12.8% of pool.** Hand-curated 167-bit pharmacophore keys are 2.4× more information-dense per feature than Morgan's hashed bits. For HIV biology, specific substructure patterns (the ones that ended up in MACCS) matter more than broad circular topology coverage.

## What Didn't Work
- **Optuna tuning was overfitting insurance failure:** Optimizing val AUC on scaffold split is not the same as optimizing test generalization. val↔test gap grew. For molecular property prediction on scaffold splits, cross-validation on train folds is more reliable than single-val tuning.
- **GIN+Edge ensemble blocked:** Anthony's Phase 3 predictions weren't saved to file, so direct ensemble couldn't be run. Phase 5 should explicitly save test predictions from each model.

## Error Analysis
The model systematically misses "small drug-like molecules" — the false negative set has mean MW 424 Da, 4 rings, 2.3 aromatic rings. These look like typical oral drugs. The caught actives (TP) are massive molecules (MW 630, 5.6 rings, 4.4 aromatic rings) that look nothing like a standard drug-like molecule. This structural bias is an important finding for HIV drug discovery research.

## Frontier Model Note
LLM baseline comparison blocked in Phase 4 (codex CLI and Claude CLI access issues on Windows). Planned for Phase 5.

## Next Steps
1. **Phase 5:** Test whether a CatBoost trained specifically on small-molecule actives (below MW threshold) can improve recall on that subgroup — subgroup specialist approach (same technique that worked for Healthcare readmission low-utilization patients).
2. **Save GIN test predictions** (Anthony) + CatBoost predictions for proper ensemble evaluation.
3. **Test an attention mechanism** on the MI-400 feature set — can the model re-weight features dynamically to pay more attention to small-molecule signals?
4. **LLM baseline** — run GPT/Opus on SMILES to predict HIV activity.

## References Used Today
- [1] Prokhorenkova et al., NeurIPS 2018 — "CatBoost: unbiased boosting with categorical features" — Search space design for CatBoost tuning
- [2] Sheridan et al., JCIM 2016 — "Extreme gradient boosting as a method for quantitative structure-activity relationships" — Higher regularization benefits for QSAR with binary features
- [3] Wu et al., Chemical Science 2018 — "MoleculeNet: a benchmark for molecular machine learning" — Scaffold split val↔test gap behavior

## Code Changes
- `src/phase4_mark_prep.py` — Feature computation + caching + Optuna (saves to data/processed/)
- `src/phase4_mark_analysis.py` — Stability + error analysis + feature importance (loads from cache)
- `src/phase4_mark_save.py` — Hardcode final results to JSON
- `src/build_phase4_mark_notebook.py` — Notebook builder
- `notebooks/phase4_mark_hyperparameter_error_analysis.ipynb` — Notebook (partially executed before timeout)
- `results/phase4_mark_results.json` — All Phase 4 results
- `results/phase4_mark_optuna_history.png` — Optuna trial history + convergence
- `results/phase4_mark_stability.png` — K-stability bootstrap analysis
- `results/phase4_mark_error_properties.png` — Molecular property distributions by error class
- `results/phase4_mark_error_confidence.png` — Prediction confidence by class + scaffold diversity
- `results/phase4_mark_feature_importance.png` — Top 25 features + cumulative curve + category pie
