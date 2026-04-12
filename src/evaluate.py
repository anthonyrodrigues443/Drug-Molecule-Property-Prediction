"""
Evaluation suite for Drug Molecule Property Prediction.

Runs the full evaluation pipeline on the OGB scaffold-split test set:
  - ROC-AUC, AUPRC, F1, Precision, Recall
  - Per-model and ensemble metrics
  - Subgroup analysis (Lipinski compliant vs violators)
  - Inference latency benchmarking

Usage:
    python -m src.evaluate
"""
import json
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (roc_auc_score, average_precision_score,
                              f1_score, precision_score, recall_score,
                              confusion_matrix)

warnings.filterwarnings('ignore')

from src.predict import load_ensemble, predict_single
from src.feature_engineering import compute_all_features

BASE_DIR = Path(__file__).parent.parent
RESULTS_DIR = BASE_DIR / 'results'


def evaluate_on_test_set():
    """Run full evaluation on the OGB test split."""
    import torch
    from torch_geometric.loader import DataLoader as PyGDataLoader
    from ogb.graphproppred import GraphPropPredDataset
    from src.train import ogb_to_pyg, predict_gin

    gin, cb, selector, config = load_ensemble()

    print('Loading test data...')
    dataset = GraphPropPredDataset(name='ogbg-molhiv',
                                    root=str(BASE_DIR / 'data' / 'raw'))
    split_idx = dataset.get_idx_split()
    smiles_df = pd.read_csv(BASE_DIR / 'data' / 'raw' / 'ogbg_molhiv' / 'mapping' / 'mol.csv.gz')
    labels = dataset.labels.flatten()
    test_idx = split_idx['test'].tolist()

    # GIN predictions
    print('Running GIN predictions...')
    test_data = ogb_to_pyg(dataset, test_idx)
    test_loader = PyGDataLoader(test_data, batch_size=512)
    gin_preds, gin_labels = predict_gin(gin, test_loader)

    # CatBoost predictions
    print('Running CatBoost predictions...')
    fcols = selector['feature_columns']
    mi_idx = selector['mi_indices']
    cb_preds_list = []
    cb_valid_mask = []
    for i in test_idx:
        f = compute_all_features(smiles_df['smiles'].iloc[i])
        if f is not None:
            vec = np.array([f.get(c, 0) for c in fcols], dtype=np.float32)
            np.nan_to_num(vec, copy=False)
            cb_preds_list.append(cb.predict_proba(vec[mi_idx].reshape(1, -1))[0, 1])
            cb_valid_mask.append(True)
        else:
            cb_valid_mask.append(False)

    # Align (only molecules valid for both)
    gin_by_idx = dict(zip(test_idx, gin_preds))
    lab_by_idx = dict(zip(test_idx, gin_labels))
    cb_by_idx = {}
    j = 0
    for i, valid in zip(test_idx, cb_valid_mask):
        if valid:
            cb_by_idx[i] = cb_preds_list[j]
            j += 1

    common = sorted(set(gin_by_idx) & set(cb_by_idx))
    g = np.array([gin_by_idx[i] for i in common])
    c = np.array([cb_by_idx[i] for i in common])
    y = np.array([lab_by_idx[i] for i in common])

    ec = config['ensemble']
    ens = ec['gin_weight'] * g + ec['cb_weight'] * c

    print(f'\nTest set: {len(common)} molecules')

    def metrics(preds, labels, name):
        auc = roc_auc_score(labels, preds)
        auprc = average_precision_score(labels, preds)
        binary = (preds >= 0.5).astype(int)
        f1 = f1_score(labels, binary, zero_division=0)
        prec = precision_score(labels, binary, zero_division=0)
        rec = recall_score(labels, binary, zero_division=0)
        return {'model': name, 'roc_auc': round(auc, 4), 'auprc': round(auprc, 4),
                'f1': round(f1, 4), 'precision': round(prec, 4), 'recall': round(rec, 4)}

    results = [
        metrics(g, y, 'GIN+Edge'),
        metrics(c, y, 'CatBoost MI-400'),
        metrics(ens, y, 'Ensemble (0.3/0.7)'),
    ]

    print('\n' + '=' * 70)
    print('EVALUATION RESULTS')
    print('=' * 70)
    rdf = pd.DataFrame(results)
    print(rdf.to_string(index=False))

    # Subgroup: Lipinski
    print('\nSubgroup analysis (Lipinski compliance)...')
    lip_feats = {}
    for i in common:
        f = compute_all_features(smiles_df['smiles'].iloc[i])
        if f:
            lip_feats[i] = f.get('lipinski_violations', 0)

    compliant = [i for i in common if lip_feats.get(i, 0) == 0]
    violators = [i for i in common if lip_feats.get(i, 0) > 0]

    if compliant and violators:
        ens_c = np.array([ens[common.index(i)] for i in compliant])
        y_c = np.array([y[common.index(i)] for i in compliant])
        ens_v = np.array([ens[common.index(i)] for i in violators])
        y_v = np.array([y[common.index(i)] for i in violators])

        if len(np.unique(y_c)) > 1 and len(np.unique(y_v)) > 1:
            print(f'  Compliant: n={len(compliant)}, AUC={roc_auc_score(y_c, ens_c):.4f}, '
                  f'Active rate={y_c.mean():.3f}')
            print(f'  Violators: n={len(violators)}, AUC={roc_auc_score(y_v, ens_v):.4f}, '
                  f'Active rate={y_v.mean():.3f}')

    # Latency benchmark
    print('\nLatency benchmark (10 molecules)...')
    sample_smiles = [smiles_df['smiles'].iloc[i] for i in common[:10]]
    latencies = []
    for smi in sample_smiles:
        t0 = time.perf_counter()
        predict_single(smi, gin, cb, selector, config)
        latencies.append((time.perf_counter() - t0) * 1000)
    print(f'  Mean: {np.mean(latencies):.1f}ms | Median: {np.median(latencies):.1f}ms | '
          f'P95: {np.percentile(latencies, 95):.1f}ms')

    # Save
    eval_results = {
        'models': results,
        'test_size': len(common),
        'latency_ms': {'mean': round(np.mean(latencies), 1),
                        'median': round(np.median(latencies), 1),
                        'p95': round(np.percentile(latencies, 95), 1)},
    }
    with open(RESULTS_DIR / 'evaluation_results.json', 'w') as f:
        json.dump(eval_results, f, indent=2)
    print(f'\nSaved evaluation_results.json')
    return eval_results


if __name__ == '__main__':
    evaluate_on_test_set()
