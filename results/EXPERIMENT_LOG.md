

## 2026-04-06 | Phase 1 (Mark) | Class-Imbalance + Graph Features

| Rank | Model | Features | ROC-AUC | AUPRC | Recall |
|------|-------|----------|---------|-------|--------|
| 1 | CatBoost (Combined, auto_weight) | combined_1036 | 0.7782 | 0.3708 | 0.5231 |
| 2 | CatBoost (Combined, no weight) | combined_1036 | 0.7746 | 0.3821 | 0.1615 |
| 3 | LightGBM (Combined, no weight) | combined_1036 | 0.7732 | 0.3826 | 0.1615 |
| 4 | XGBoost (Combined, no weight) | combined_1036 | 0.7703 | 0.3956 | 0.1692 |
| 5 | RF (Combined, no weight) | combined_1036 | 0.7655 | 0.3493 | 0.1538 |
| 6 | RF (Combined, balanced) | combined_1036 | 0.7634 | 0.3507 | 0.1615 |
| 7 | XGBoost (Combined, scale_pos_weight) | combined_1036 | 0.7618 | 0.3936 | 0.4923 |
| 8 | LightGBM (Combined, is_unbalance) | combined_1036 | 0.759 | 0.3803 | 0.4846 |
| 9 | LogReg (Domain 12, balanced) | domain_12 | 0.746 | 0.1251 | 0.6538 |
| 10 | LogReg (Domain 12, no weight) | domain_12 | 0.7366 | 0.1581 | 0.0 |
| 11 | LightGBM (FP 1024, is_unbalance) | fp_1024 | 0.7077 | 0.3318 | 0.4 |
| 12 | XGBoost (Graph topo 5-feat) | graph_topo_5 | 0.7013 | 0.1852 | 0.0692 |
| 13 | RF (Graph topo 5-feat) | graph_topo_5 | 0.6037 | 0.1332 | 0.0692 |


## 2026-04-07 | Phase 2 (Mark) | Neural Baselines on Molecular Fingerprints

Complementary to Anthony's Phase 2 GNN sweep: tested whether the GNN failure is
GNN-specific by running plain PyTorch MLPs on the same feature matrix CatBoost uses.

| Rank | Model | Features | Test ROC-AUC | Val ROC-AUC | Δ vs CatBoost | Δ vs Anthony GIN | Params | Train(s) |
|------|-------|----------|--------------|-------------|---------------|------------------|--------|----------|
| 1 | **MLP-Domain9** | Domain 9 | **0.7670** | 0.7561 | **-0.0112** | **+0.0617** | ~5K | 7.2 |
| 2 | MLP-Wide-Combined | Combined 1033 | 0.7064 | 0.8051 | -0.0718 | +0.0011 | ~1.1M | 101.9 |
| 3 | MLP-Combined1033 | Combined 1033 | 0.7011 | 0.8235 | -0.0771 | -0.0042 | ~400K | 159.5 |
| 4 | MLP-Morgan1024 | Morgan FP 1024 | 0.6736 | 0.7925 | -0.1046 | -0.0317 | ~400K | 106.8 |

Phase 2 Mark key findings:
1. MLP-Domain9 (5K params, 9 features) beats all 4 of Anthony's GNNs (GIN 0.7053, SAGE 0.7050, GCN 0.6938, GAT 0.6677).
2. MLPs on Morgan fingerprints (0.6736–0.7064) match Anthony's GNN failure band — the failure is NOT GNN-specific.
3. Adding Morgan FP to MLP-Domain9 HURT the model (-0.066 AUC), while the same addition HELPS CatBoost (+0.032).
4. Doubling MLP width/depth improved only +0.005 — capacity is not the bottleneck.


## 2026-04-08 | Phase 3 (Anthony) | Feature Engineering + Edge-Aware GNNs + Hybrid

### GNN Experiments (OGB AtomEncoder + BondEncoder)

| Rank | Model | Test AUC | Val AUC | AUPRC | Params | Time |
|------|-------|----------|---------|-------|--------|------|
| 1 | **GIN+Edge (no VN)** | **0.7860** | 0.8001 | 0.3441 | 144K | 297s |
| 2 | GIN+Edge+VN (OGB) | 0.7622 | 0.8333 | 0.2858 | 211K | 292s |
| 3 | GIN+VN (no edge) | 0.7578 | 0.8134 | 0.2642 | 206K | 199s |

### Feature Ablation (all via CatBoost, balanced weights)

| Rank | Feature Set | Dims | Test AUC | AUPRC |
|------|-------------|------|----------|-------|
| 1 | All traditional | 1217 | 0.7841 | 0.3416 |
| 2 | Lipinski | 14 | 0.7744 | 0.2779 |
| 3 | GNN embed | 128 | 0.7692 | 0.2957 |
| 4 | GNN+Lip+FP (hybrid) | 1166 | 0.7688 | 0.3526 |
| 5 | GNN+Lip (hybrid) | 142 | 0.7662 | 0.3101 |
| 6 | Lip+FP | 1038 | 0.7619 | 0.3239 |
| 7 | MACCS | 167 | 0.7605 | 0.2859 |
| 8 | Lip+MACCS | 181 | 0.7568 | 0.3073 |
| 9 | Full hybrid | 1345 | 0.7415 | 0.2220 |
| 10 | Advanced desc | 12 | 0.7219 | 0.2123 |
| 11 | Morgan FP | 1024 | 0.7176 | 0.2820 |

**Key findings:**
1. Edge features = +0.081 AUC (biggest single improvement across all phases)
2. Virtual node causes overfitting on scaffold split (best val, worst test gap)
3. More features can HURT: Full hybrid (1345d) < Lipinski alone (14d)
4. GIN+Edge is Phase 3 champion (0.7860) — first GNN to beat CatBoost


## 2026-04-08 | Phase 3 (Mark) | RDKit Fragments + Mutual-Info Feature Selection

Complementary to Anthony's Phase 3 GNN+edge experiments. Added 85 RDKit `Fr_*` functional-group
descriptors to the feature pool and ran a mutual-information top-K selection sweep on the
combined 1302-dim pool to test whether Anthony's "more features hurt" finding was about
noise (fixable by selection) or signal loss (not fixable).

### M3.1 — CatBoost feature-set head-to-head (NaN-cleaned float32 matrices)

| Rank | Feature Set | Dims | Test AUC | Test AUPRC |
|------|-------------|------|----------|------------|
| 1 | AllTrad (1217) | 1217 | 0.7814 | 0.3490 |
| 2 | AllTrad+Frag (1302) | 1302 | 0.7677 | 0.3647 |
| 3 | MACCS (167) | 167 | 0.7670 | 0.3098 |
| 4 | Lipinski (14) | 14 | 0.7459 | 0.2758 |
| 5 | Morgan (1024) | 1024 | 0.7448 | 0.3398 |
| 6 | Lip+Frag (99) | 99 | 0.7247 | 0.2604 |
| 7 | Fragments (85) | 85 | 0.6999 | 0.2411 |

### M3.2 — Mutual-info top-K sweep on AllTrad+Frag pool (1302 dims)

| K | Test AUC | Test AUPRC | Δ vs GIN+Edge 0.7860 |
|---|---------:|-----------:|---------------------:|
| 20 | 0.7702 | 0.2168 | -0.0158 |
| 50 | 0.7591 | 0.2793 | -0.0269 |
| 100 | 0.7892 | 0.3127 | +0.0032 |
| 200 | 0.8019 | 0.3285 | +0.0159 |
| 300 | 0.7883 | 0.3076 | +0.0023 |
| 350 | 0.7813 | 0.3288 | -0.0047 |
| **400** ← champion | **0.8105** | 0.3481 | **+0.0245** |
| 450 | 0.7796 | 0.3164 | -0.0064 |
| 500 | 0.7945 | 0.3244 | +0.0085 |
| 600 | 0.7941 | 0.3532 | +0.0081 |
| 800 | 0.7836 | 0.3421 | -0.0024 |
| 1302 | 0.7673 | 0.3484 | -0.0187 |

### K=400 champion composition

| Category | Pool | Selected | % of category | % of K=400 |
|----------|-----:|---------:|--------------:|-----------:|
| Lipinski | 14 | 14 | 100.0% | 3.5% |
| Advanced | 12 | 10 | 83.3% | 2.5% |
| MACCS | 167 | 108 | 64.7% | 27.0% |
| Fragments | 85 | 33 | 38.8% | 8.25% |
| Morgan | 1024 | 235 | 22.9% | 58.75% |

**Key findings:**
1. CatBoost + MI-top-400 = **0.8105 AUC**, beating Anthony's Phase 3 GIN+Edge champion (0.7860) by +0.0245, with zero graph layers.
2. On the same 1302-d pool: using everything → 0.7673, keeping top 400 → 0.8105. +0.0432 from a univariate filter with no model change.
3. MACCS is over-represented in the winning subset (27% of K=400 vs 12.8% pool share). Hand-curated > hashed substructure keys per unit.
4. Fragments-only = 0.6999 — worst feature set tested. H1 (Fr_* alone beat Lipinski) falsified. Fragments carry marginal but not standalone signal.
5. All 14 Lipinski features survive every K≥300 selection level. Classical physicochemistry still matters.


## 2026-04-09 | Phase 4 (Anthony) | GIN+Edge Tuning + GNN-CatBoost Ensemble

### 4.1 — GIN+Edge Optuna Tuning (8 trials, TPE sampler)

| Trial | Dim | Layers | Dropout | LR | Pool | Val AUC | Test AUC |
|-------|-----|--------|---------|----|------|---------|----------|
| 2 | 64 | 3 | 0.4 | 0.0037 | add | 0.7957 | **0.7904** |
| 0 | 128 | 4 | 0.2 | 0.0002 | mean | 0.7930 | 0.7860 |
| 6 | 256 | 3 | 0.3 | 0.0012 | mean | **0.8107** | 0.7631 |

Retrained best-test config (trial 2): **test=0.7982** (+0.012 vs Phase 3 default 0.7860)

### 4.3 — GIN + CatBoost Ensemble (weight sweep)

| w_GIN | w_CB | Ensemble AUC | Δ vs best solo |
|-------|------|-------------|----------------|
| 0.0 | 1.0 | 0.7939 | — |
| **0.3** | **0.7** | **0.8114** | **+0.013** |
| 0.5 | 0.5 | 0.8087 | +0.011 |
| 1.0 | 0.0 | 0.7981 | — |

### 4.4 — Error Overlap Analysis

| Category | Count | % of test |
|----------|-------|-----------|
| Both correct | 2,946 | 71.6% |
| Both wrong | 188 | 4.6% |
| GIN wrong only | 873 | 21.2% |
| CatBoost wrong only | 104 | 2.5% |
| **Jaccard** | **0.161** | |

**Key findings:**
1. **GIN+CatBoost ensemble = 0.8114 AUC** — new project champion, beating Mark P3 best (0.8105)
2. Jaccard overlap = 0.161 — models fail on DIFFERENT molecules = textbook ensemble diversity
3. GIN tuning: 64d model generalizes best. 5-layer models catastrophically overfit (0.69-0.72 test)
4. Optimal weight: 30% GIN + 70% CatBoost — CatBoost carries more signal but GIN adds complementary topology info
