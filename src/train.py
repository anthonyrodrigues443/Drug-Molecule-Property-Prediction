"""
Production training pipeline for Drug Molecule Property Prediction.

Trains the champion GIN+CatBoost ensemble:
  1. GIN+Edge GNN on molecular graphs (OGB scaffold split)
  2. CatBoost on MI-selected top-400 features
  3. Saves both models + MI selector + ensemble config

Usage:
    python -m src.train [--config config/config.yaml]
"""
import json
import warnings
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
import joblib
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GINConv, global_mean_pool, global_add_pool
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader as PyGDataLoader
from sklearn.metrics import roc_auc_score
from catboost import CatBoostClassifier

warnings.filterwarnings('ignore')

from ogb.graphproppred import GraphPropPredDataset
from ogb.graphproppred.mol_encoder import AtomEncoder, BondEncoder

from src.feature_engineering import compute_all_features, select_features_mi

BASE_DIR = Path(__file__).parent.parent
MODELS_DIR = BASE_DIR / 'models'
RESULTS_DIR = BASE_DIR / 'results'


class GINEdge(nn.Module):
    def __init__(self, hidden_dim=64, num_layers=3, dropout=0.4, pool_type='add'):
        super().__init__()
        self.num_layers = num_layers
        self.atom_encoder = AtomEncoder(hidden_dim)
        self.bond_encoders = nn.ModuleList([BondEncoder(hidden_dim) for _ in range(num_layers)])
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        for _ in range(num_layers):
            mlp = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim), nn.BatchNorm1d(hidden_dim),
                nn.ReLU(), nn.Linear(hidden_dim, hidden_dim))
            self.convs.append(GINConv(mlp, train_eps=True))
            self.bns.append(nn.BatchNorm1d(hidden_dim))
        self.pool = global_add_pool if pool_type == 'add' else global_mean_pool
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Dropout(dropout), nn.Linear(hidden_dim, 1))
        self.dropout = dropout

    def forward(self, x, edge_index, edge_attr, batch):
        h = self.atom_encoder(x)
        for layer in range(self.num_layers):
            bond_emb = self.bond_encoders[layer](edge_attr)
            row = edge_index[0]
            bond_agg = torch.zeros_like(h)
            bond_agg.scatter_reduce_(0, row.unsqueeze(-1).expand_as(bond_emb), bond_emb, reduce='mean')
            h_in = h + bond_agg
            h = self.convs[layer](h_in, edge_index)
            h = self.bns[layer](h)
            if layer < self.num_layers - 1:
                h = F.relu(h)
                h = F.dropout(h, p=self.dropout, training=self.training)
        return self.classifier(self.pool(h, batch)).squeeze(-1)


def ogb_to_pyg(dataset, idx_list):
    data_list = []
    for i in idx_list:
        graph, label = dataset[i]
        data_list.append(Data(
            x=torch.tensor(graph['node_feat'], dtype=torch.long),
            edge_index=torch.tensor(graph['edge_index'], dtype=torch.long),
            edge_attr=torch.tensor(graph['edge_feat'], dtype=torch.long),
            y=torch.tensor([label[0]], dtype=torch.float)))
    return data_list


@torch.no_grad()
def predict_gin(model, loader):
    model.eval()
    preds, labs = [], []
    for data in loader:
        out = model(data.x, data.edge_index, data.edge_attr, data.batch)
        preds.append(torch.sigmoid(out).numpy())
        labs.append(data.y.numpy())
    return np.concatenate(preds), np.concatenate(labs)


def train_gin_model(config, train_data, val_data):
    gc = config['gin']
    bs = gc.get('batch_size', 512)
    train_loader = PyGDataLoader(train_data, batch_size=bs, shuffle=True)
    val_loader = PyGDataLoader(val_data, batch_size=bs)

    model = GINEdge(gc['hidden_dim'], gc['num_layers'], gc['dropout'], gc['pool_type'])
    optimizer = torch.optim.Adam(model.parameters(), lr=gc['lr'], weight_decay=1e-5)

    pos_count = sum(1 for d in train_data if d.y.item() > 0.5)
    pos_weight = torch.tensor([len(train_data) / max(pos_count, 1) - 1], dtype=torch.float)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    best_val, best_state, wait = 0, None, 0
    for ep in range(gc['epochs']):
        model.train()
        for data in train_loader:
            optimizer.zero_grad()
            criterion(model(data.x, data.edge_index, data.edge_attr, data.batch), data.y).backward()
            optimizer.step()
        vp, vl = predict_gin(model, val_loader)
        vauc = roc_auc_score(vl, vp)
        if vauc > best_val:
            best_val = vauc
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1
            if wait >= gc['patience']:
                break

    model.load_state_dict(best_state)
    print(f'  GIN trained {ep+1} epochs, val AUC={best_val:.4f}')
    return model


def train_catboost_model(config, X_train, y_train, X_val, y_val):
    cc = config['catboost']
    cb = CatBoostClassifier(
        depth=cc['depth'], learning_rate=cc['learning_rate'],
        l2_leaf_reg=cc['l2_leaf_reg'], iterations=cc['iterations'],
        min_data_in_leaf=cc['min_data_in_leaf'],
        random_strength=cc['random_strength'],
        bagging_temperature=cc['bagging_temperature'],
        border_count=cc['border_count'],
        auto_class_weights=cc['auto_class_weights'],
        eval_metric=cc['eval_metric'],
        random_seed=config['seed'], verbose=0)
    cb.fit(X_train, y_train, eval_set=(X_val, y_val),
           early_stopping_rounds=cc['early_stopping_rounds'], verbose=0)
    val_pred = cb.predict_proba(X_val)[:, 1]
    print(f'  CatBoost trained, val AUC={roc_auc_score(y_val, val_pred):.4f}')
    return cb


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='config/config.yaml')
    args = parser.parse_args()

    with open(BASE_DIR / args.config) as f:
        config = yaml.safe_load(f)

    torch.manual_seed(config['seed'])
    np.random.seed(config['seed'])
    MODELS_DIR.mkdir(exist_ok=True)
    RESULTS_DIR.mkdir(exist_ok=True)

    print('Loading ogbg-molhiv...')
    dataset = GraphPropPredDataset(name=config['dataset']['name'],
                                    root=str(BASE_DIR / config['dataset']['root']))
    split_idx = dataset.get_idx_split()
    smiles_df = pd.read_csv(BASE_DIR / 'data' / 'raw' / 'ogbg_molhiv' / 'mapping' / 'mol.csv.gz')
    labels = dataset.labels.flatten()

    train_idx = split_idx['train'].tolist()
    val_idx = split_idx['valid'].tolist()
    test_idx = split_idx['test'].tolist()

    # --- GIN ---
    print('\n[1/3] Training GIN+Edge...')
    train_data = ogb_to_pyg(dataset, train_idx)
    val_data = ogb_to_pyg(dataset, val_idx)
    gin_model = train_gin_model(config, train_data, val_data)
    torch.save(gin_model.state_dict(), MODELS_DIR / 'gin_edge.pt')
    print(f'  Saved gin_edge.pt')

    # --- CatBoost features ---
    print('\n[2/3] Computing features + Training CatBoost...')
    all_indices = train_idx + val_idx + test_idx
    feat_rows = []
    valid_map = {}
    for i in all_indices:
        f = compute_all_features(smiles_df['smiles'].iloc[i])
        if f is not None:
            f['idx'] = i
            f['y'] = int(labels[i])
            feat_rows.append(f)
            valid_map[i] = len(feat_rows) - 1

    feat_df = pd.DataFrame(feat_rows)
    ts, vs = set(train_idx), set(val_idx)
    feat_df['split'] = feat_df['idx'].apply(lambda x: 'train' if x in ts else 'val' if x in vs else 'test')

    fcols = [c for c in feat_df.columns if c not in {'idx', 'y', 'split'}]
    X_tr = feat_df[feat_df['split'] == 'train'][fcols].values.astype(np.float32)
    y_tr = feat_df[feat_df['split'] == 'train']['y'].values
    X_va = feat_df[feat_df['split'] == 'val'][fcols].values.astype(np.float32)
    y_va = feat_df[feat_df['split'] == 'val']['y'].values
    for X in [X_tr, X_va]:
        np.nan_to_num(X, copy=False)

    mi_k = config['features']['mi_top_k']
    mi_indices = select_features_mi(X_tr, y_tr, k=mi_k, seed=config['seed'])
    selected_names = [fcols[i] for i in mi_indices]

    cb_model = train_catboost_model(config, X_tr[:, mi_indices], y_tr, X_va[:, mi_indices], y_va)
    cb_model.save_model(str(MODELS_DIR / 'catboost_mi400.cbm'))
    joblib.dump({'mi_indices': mi_indices, 'feature_columns': fcols,
                 'selected_features': selected_names}, MODELS_DIR / 'feature_selector.joblib')
    print(f'  Saved catboost_mi400.cbm + feature_selector.joblib')

    # --- Save config used ---
    print('\n[3/3] Saving training metadata...')
    meta = {
        'config': config,
        'dataset_size': len(all_indices),
        'train_size': len(train_idx),
        'val_size': len(val_idx),
        'test_size': len(test_idx),
        'mi_top_k': mi_k,
        'n_features_total': len(fcols),
    }
    with open(MODELS_DIR / 'training_meta.json', 'w') as f:
        json.dump(meta, f, indent=2, default=str)

    print('\nTraining complete. Models saved to models/')


if __name__ == '__main__':
    main()
