

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
