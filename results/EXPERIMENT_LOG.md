

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
