"""
Inference pipeline for Drug Molecule Property Prediction.

Loads the GIN+CatBoost ensemble and predicts HIV activity for new SMILES.

Usage:
    python -m src.predict --smiles "CC(=O)Oc1ccccc1C(=O)O"
    python -m src.predict --file molecules.txt
"""
import argparse
import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import joblib
import torch
import torch.nn as nn
from catboost import CatBoostClassifier
from rdkit import Chem

warnings.filterwarnings('ignore')

from src.train import GINEdge, ogb_to_pyg
from src.feature_engineering import compute_all_features

BASE_DIR = Path(__file__).parent.parent
MODELS_DIR = BASE_DIR / 'models'


def load_ensemble(config_path: str = 'models/training_meta.json'):
    """Load trained GIN+Edge and CatBoost models."""
    with open(BASE_DIR / config_path) as f:
        meta = json.load(f)
    config = meta['config']
    gc = config['gin']

    gin = GINEdge(gc['hidden_dim'], gc['num_layers'], gc['dropout'], gc['pool_type'])
    gin.load_state_dict(torch.load(MODELS_DIR / 'gin_edge.pt', weights_only=False))
    gin.eval()

    cb = CatBoostClassifier()
    cb.load_model(str(MODELS_DIR / 'catboost_mi400.cbm'))

    selector = joblib.load(MODELS_DIR / 'feature_selector.joblib')

    return gin, cb, selector, config


def predict_single(smiles: str, gin, cb, selector, config) -> dict:
    """Predict HIV activity for a single SMILES string."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return {'smiles': smiles, 'error': 'Invalid SMILES', 'prediction': None}

    t0 = time.perf_counter()

    # CatBoost prediction
    feats = compute_all_features(smiles)
    if feats is None:
        return {'smiles': smiles, 'error': 'Feature extraction failed', 'prediction': None}

    fcols = selector['feature_columns']
    mi_idx = selector['mi_indices']
    feat_vec = np.array([feats.get(c, 0) for c in fcols], dtype=np.float32)
    np.nan_to_num(feat_vec, copy=False)
    cb_prob = cb.predict_proba(feat_vec[mi_idx].reshape(1, -1))[0, 1]

    # GIN prediction (requires OGB-format graph)
    gin_prob = None
    try:
        from ogb.graphproppred import GraphPropPredDataset
        from torch_geometric.data import Data
        from ogb.utils.mol import smiles2graph

        graph = smiles2graph(smiles)
        data = Data(
            x=torch.tensor(graph['node_feat'], dtype=torch.long),
            edge_index=torch.tensor(graph['edge_index'], dtype=torch.long),
            edge_attr=torch.tensor(graph['edge_feat'], dtype=torch.long),
            y=torch.tensor([0.0]))
        data.batch = torch.zeros(data.x.size(0), dtype=torch.long)

        with torch.no_grad():
            logit = gin(data.x, data.edge_index, data.edge_attr, data.batch)
            gin_prob = float(torch.sigmoid(logit).item())
    except Exception:
        pass

    # Ensemble
    ec = config['ensemble']
    if gin_prob is not None:
        ensemble_prob = ec['gin_weight'] * gin_prob + ec['cb_weight'] * cb_prob
    else:
        ensemble_prob = cb_prob

    latency_ms = (time.perf_counter() - t0) * 1000

    return {
        'smiles': smiles,
        'ensemble_probability': round(float(ensemble_prob), 4),
        'prediction': 'HIV-active' if ensemble_prob >= 0.5 else 'HIV-inactive',
        'gin_probability': round(gin_prob, 4) if gin_prob is not None else None,
        'catboost_probability': round(float(cb_prob), 4),
        'latency_ms': round(latency_ms, 1),
        'lipinski': {
            'mol_weight': feats['mol_weight'],
            'logp': feats['logp'],
            'hbd': feats['hbd'],
            'hba': feats['hba'],
            'violations': feats['lipinski_violations'],
        },
    }


def predict_batch(smiles_list: list[str], gin, cb, selector, config) -> list[dict]:
    """Predict HIV activity for a batch of SMILES."""
    return [predict_single(s, gin, cb, selector, config) for s in smiles_list]


def main():
    parser = argparse.ArgumentParser(description='Predict HIV activity for molecules')
    parser.add_argument('--smiles', type=str, help='Single SMILES string')
    parser.add_argument('--file', type=str, help='File with one SMILES per line')
    parser.add_argument('--json', action='store_true', help='Output as JSON')
    args = parser.parse_args()

    if not args.smiles and not args.file:
        parser.error('Provide --smiles or --file')

    gin, cb, selector, config = load_ensemble()

    if args.smiles:
        smiles_list = [args.smiles]
    else:
        with open(args.file) as f:
            smiles_list = [line.strip() for line in f if line.strip()]

    results = predict_batch(smiles_list, gin, cb, selector, config)

    if args.json:
        print(json.dumps(results, indent=2))
    else:
        for r in results:
            if r.get('error'):
                print(f"  {r['smiles']}: ERROR - {r['error']}")
            else:
                print(f"  {r['smiles']}: {r['prediction']} "
                      f"(p={r['ensemble_probability']:.3f}, {r['latency_ms']:.0f}ms)")


if __name__ == '__main__':
    main()
