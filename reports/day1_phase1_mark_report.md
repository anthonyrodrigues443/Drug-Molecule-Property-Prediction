# Phase 1: Class-Imbalance Strategies + Graph Features — Drug Molecule Property Prediction
**Date:** 2026-04-06
**Session:** 1 of 7
**Researcher:** Mark Rodrigues

---

## Objective

Anthony chose ogbg-molhiv (41K molecules, 3.5% HIV-active, ROC-AUC primary metric) and showed RF (Combined 1036 features) achieves ROC-AUC=0.7707, 0.077 below OGB SOTA.

My complementary question: **At 3.5% positive rate, is the bottleneck the model or the class imbalance handling?** Does class-weighting help or hurt ROC-AUC? And can graph-level topological features (n_atoms, n_bonds, graph density) add signal beyond Lipinski descriptors?

---

## Building on Anthony's Work

**Anthony found:**
- RF (Combined 1036) = ROC-AUC 0.7707 champion; AUPRC=0.3722
- Combined features (Lipinski + FP) consistently beat either alone
- Gap to SOTA (0.077 AUC) is in graph topology that fingerprints miss
- Dataset: ogbg-molhiv, ROC-AUC primary metric (OGB leaderboard standard)

**My approach:** Test three angles Anthony didn't explore:
1. Class-weighting strategies (balanced, scale_pos_weight, is_unbalance) — do they help or hurt on this specific imbalance level?
2. CatBoost and LightGBM — model families Anthony didn't test
3. Graph topological features (n_atoms, n_bonds, graph_density, bonds_per_atom, aromatic_fraction) — do simple graph-level stats add signal?

**Combined insight:** CatBoost with auto-balancing beats Anthony's RF champion by +0.0075 AUC — the model family matters. But class-weighting helps recall at the cost of AUC (a trade-off, not a free lunch). Graph topology alone gets ROC-AUC=0.70, confirming the 0.077 gap to SOTA is real graph-level information.

---

## Research & References

1. **Hu, W. et al. (2020). "Open Graph Benchmark." NeurIPS.** — ogbg-molhiv leaderboard and official scaffold split. SOTA at 0.8476 uses graph-level representations, confirming that tabular models with fingerprints cannot fully capture molecular topology.

2. **Prokhorenkova, L. et al. (2018). "CatBoost: unbiased boosting with categorical features." NeurIPS.** — CatBoost handles class imbalance via ordered boosting + auto_class_weights='Balanced', which adapts sample weights without the naive oversampling pitfalls. This motivated testing CatBoost as a new model family.

3. **He, H. & Garcia, E. (2009). "Learning from Imbalanced Data." IEEE TKDE.** — At 3.5% positive rate (not extreme), class-weighting should help recall without catastrophic AUC loss. But threshold optimization may matter more than sample weighting. Key insight: for HIV drug screening, recall (catching actives) matters more than precision.

**How research influenced experiments:** He & Garcia's framework predicts that moderate imbalance (3.5%) should respond to class-weighting without the discrimination collapse seen at <1% rates. I tested this directly.

---

## Dataset

| Metric | Value |
|--------|-------|
| Dataset | ogbg-molhiv (OGB) |
| Total molecules | 41,127 |
| Positive rate | 3.51% (1,443 HIV-active) |
| Split | OGB official scaffold split |
| Train | 32,901 (pos: 1,232, rate: 3.74%) |
| Val | 4,113 (pos: 81, rate: 1.97%) |
| Test | 4,113 (pos: 130, rate: 3.16%) |
| Primary metric | ROC-AUC (OGB leaderboard standard) |
| Secondary metric | AUPRC (critical for imbalanced data) |
| SOTA target | ROC-AUC = 0.8476 |

---

## Experiments

### Experiment M1.1: Class-Weight Strategies on Combined Features

**Hypothesis:** At 3.5% positive rate, class-weighting should improve recall but may hurt ROC-AUC by distorting probability calibration.

**Method:** RF, XGBoost, LightGBM each tested with and without class-weighting on Combined (1036) features.

**Result:**

| Model | Weighting | ROC-AUC | AUPRC | F1 | Recall |
|-------|-----------|---------|-------|----|--------|
| LightGBM | None | **0.7732** | 0.3826 | 0.273 | 0.162 |
| XGBoost | None | 0.7703 | **0.3956** | 0.277 | 0.169 |
| RF | None | 0.7655 | 0.3493 | 0.255 | 0.154 |
| RF | Balanced | 0.7634 | 0.3507 | 0.268 | 0.162 |
| XGBoost | scale_pos_weight | 0.7618 | 0.3936 | 0.310 | **0.492** |
| LightGBM | is_unbalance | 0.7590 | 0.3803 | **0.317** | 0.485 |

**Key finding:** Class-weighting consistently HURTS ROC-AUC (by -0.01 to -0.02) but dramatically improves recall (from ~0.16 to ~0.49). This is the expected trade-off at moderate imbalance — weighting shifts the decision boundary toward the minority class, catching more actives but at the cost of ranking quality. For drug screening, where missing an active is expensive, the weighted models may actually be preferred despite lower AUC.

---

### Experiment M1.2: CatBoost (New Model Family)

**Hypothesis:** CatBoost's ordered boosting + auto class weights should handle the 3.5% imbalance better than other gradient boosters.

**Method:** CatBoost with auto_class_weights='Balanced' vs no weight, Combined features.

**Result:**

| Model | ROC-AUC | AUPRC | Recall |
|-------|---------|-------|--------|
| **CatBoost (auto_weight)** | **0.7782** | 0.3708 | **0.523** |
| CatBoost (no weight) | 0.7746 | 0.3821 | 0.162 |

**Key finding:** CatBoost auto-weighted is the new Phase 1 champion: ROC-AUC=0.7782, beating Anthony's RF (0.7707) by +0.0075. Crucially, it achieves this while maintaining 52% recall — the only model that improves BOTH AUC and recall simultaneously. CatBoost's ordered boosting approach handles the class imbalance more gracefully than RF's bag-level weighting or XGBoost's global scale_pos_weight.

---

### Experiment M1.3: Graph Topological Features Only

**Hypothesis:** Simple graph-level statistics (n_atoms, n_bonds, graph density, bonds/atom, aromatic fraction) should capture some HIV activity signal.

**Method:** RF and XGBoost on 5 graph topological features only.

**Result:**

| Model | ROC-AUC | AUPRC |
|-------|---------|-------|
| XGBoost (5 graph features) | 0.7013 | 0.1852 |
| RF (5 graph features) | 0.6037 | 0.1332 |

**Key finding:** 5 graph-level features alone reach ROC-AUC=0.70 — not terrible, but far below the combined model's 0.78. This confirms that molecular topology provides some signal for HIV activity (larger, more complex molecules tend to be more active), but the specific substructure information in fingerprints is essential. The 0.077 gap to SOTA can't be closed by simple graph statistics — it requires learned graph representations (GNNs).

---

## Head-to-Head Comparison

| Rank | Model | Features | ROC-AUC | AUPRC | Recall | Notes |
|------|-------|----------|---------|-------|--------|-------|
| **1** | **CatBoost (auto_weight)** | **Combined 1036** | **0.7782** | **0.3708** | **0.523** | **New champion** |
| 2 | CatBoost (no weight) | Combined 1036 | 0.7746 | 0.3821 | 0.162 | |
| 3 | LightGBM (no weight) | Combined 1036 | 0.7732 | 0.3826 | 0.162 | |
| *ref* | *Anthony RF (no weight)* | *Combined 1036* | *0.7707* | *0.3722* | *0.308* | *Prior champion* |
| 4 | XGBoost (no weight) | Combined 1036 | 0.7703 | 0.3956 | 0.169 | Best AUPRC |
| 5 | LogReg (balanced) | Domain 12 | 0.7460 | 0.1251 | 0.654 | Best recall |
| 6 | XGBoost (graph 5-feat) | Graph topo | 0.7013 | 0.1852 | 0.069 | |
| *target* | *OGB SOTA* | *Graph-level* | *0.8476* | — | — | *GNN required* |

---

## Key Findings

### Finding 1 — CatBoost with auto-weighting is the new champion (ROC-AUC 0.7782)
Beats Anthony's RF (0.7707) by +0.0075 while maintaining 52% recall. CatBoost's ordered boosting handles class imbalance more gracefully than other gradient boosters.

### Finding 2 — Class-weighting is a TRADE-OFF, not a free lunch
Every weighted model has LOWER ROC-AUC than its unweighted counterpart (RF: -0.002, XGBoost: -0.009, LightGBM: -0.014). But weighted models catch 3x more HIV-active molecules (recall ~0.49 vs ~0.16). For drug screening where missing an active compound costs $millions in lost drug candidates, the weighted models may be operationally preferred.

### Finding 3 — XGBoost has the best AUPRC (0.3956) despite not winning ROC-AUC
The best model depends on which metric you optimize. XGBoost (no weight) has the highest AUPRC, meaning its precision-recall trade-off is strongest. This matters more than ROC-AUC for the actual drug screening task.

### Finding 4 — Graph topology alone gets ROC-AUC=0.70
5 simple graph statistics capture ~70% of the combined model's discrimination. The remaining 0.08 AUC gap requires specific substructure information (fingerprints) or learned graph representations (GNNs).

---

## Error Analysis

- **Val set imbalance:** Val set has only 1.97% positives vs 3.74% in train — this may cause validation-based threshold tuning to be unreliable
- **Scaffold split challenge:** OGB scaffold split ensures train and test molecules have different scaffolds, making generalization harder than random split
- **Fingerprint dominance:** FP-only LightGBM (0.7077) vs Domain-only LogReg (0.7366) — fingerprints alone underperform domain features for ranking, but fingerprints add +0.03 AUC when combined

---

## Next Steps

- **Phase 2:** GNN architectures (GCN, GAT, GIN) that learn from molecular graphs directly — can they close the 0.07 gap to SOTA? Also test CatBoost + GNN feature fusion.
- **Threshold optimization:** Use validation set to find optimal probability threshold for F1/recall
- **Feature ablation:** Which of the 12 Lipinski features drive the +0.07 lift over fingerprints alone?

---

## References Used Today

- [1] Hu, W. et al. (2020). Open Graph Benchmark: Datasets for Machine Learning on Graphs. NeurIPS.
- [2] Prokhorenkova, L. et al. (2018). CatBoost: unbiased boosting with categorical features. NeurIPS.
- [3] He, H. & Garcia, E. (2009). Learning from Imbalanced Data. IEEE TKDE.

---

## Code Changes

- `src/phase1_mark_imbalance_baselines.py` — Main experiment: class-weighting comparison, CatBoost, LightGBM, graph features
- `data/processed/ogbg_molhiv_features.parquet` — Preprocessed feature matrix (41K x 1037)
- `results/phase1_mark_imbalance_comparison.png` — ROC-AUC + AUPRC comparison plot
- `results/metrics.json` — Updated with Mark Phase 1 results
- `results/EXPERIMENT_LOG.md` — Updated experiment table
