# Model Card: HIV Drug Activity Predictor

## Model Details

- **Model type:** GIN+Edge GNN + CatBoost ensemble (0.3/0.7 weighted average)
- **Task:** Binary classification — HIV activity prediction (active/inactive)
- **Dataset:** ogbg-molhiv (41,127 molecules, 3.5% HIV-active)
- **Split:** OGB scaffold split (32,901 train / 4,113 val / 4,113 test)
- **Primary metric:** ROC-AUC (OGB leaderboard standard)
- **Champion AUC:** 0.8114

## Architecture

### GIN+Edge (Graph Isomorphism Network with Bond Encoding)
- 3 GIN layers, 64-dim hidden, add-pooling, dropout=0.4
- AtomEncoder + BondEncoder from OGB
- Bond features aggregated via scatter-reduce before each GIN layer
- Trained with BCE + pos_weight for class imbalance

### CatBoost (Gradient Boosted Trees)
- MI-selected top-400 features from 1,302-dim pool
- Feature pool: 14 Lipinski/domain + 1,024 Morgan FP + 167 MACCS keys + 85 RDKit fragments
- depth=8, lr=0.055, l2_reg=4.7, balanced class weights

### Ensemble
- Simple weighted average: 0.3 * GIN + 0.7 * CatBoost
- Error Jaccard overlap = 0.161 (models fail on different molecules)
- Ensemble rescued 542 test molecules, hurt zero

## Performance

| Model | ROC-AUC | AUPRC |
|-------|---------|-------|
| Ensemble (0.3/0.7) | **0.8114** | 0.368 |
| GIN+Edge (tuned) | 0.7982 | — |
| CatBoost MI-400 | 0.7939 | 0.333 |
| OGB SOTA (reference) | 0.8476 | — |

## Known Limitations

1. **Lipinski-compliant actives:** AUC=0.6707, recall=3.3%. The model excels at identifying large, complex HIV protease inhibitors (violators: AUC=0.845) but nearly fails on small drug-like actives.
2. **Scaffold split sensitivity:** Performance varies with scaffold alignment. Fragment feature impact ranged from -0.004 to +0.026 across different runs.
3. **No 3D information:** Uses 2D graph topology only. Conformer-aware models may capture binding pocket geometry.

## Intended Use

Research tool for virtual screening of HIV drug candidates. Should be used as a prioritization filter, not a replacement for experimental validation. Best suited for identifying large, structurally complex molecules with HIV protease inhibitor characteristics.

## Ethical Considerations

- Model trained on publicly available OGB benchmark data
- Not validated for clinical decision-making
- Performance gap on Lipinski-compliant molecules means the model may systematically miss orally bioavailable drug candidates
