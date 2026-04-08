"""
Phase 3: Feature Engineering + Deep Dive — Drug-Molecule-Property-Prediction
Date: 2026-04-08
Researcher: Anthony Rodrigues

Key experiments:
  3.1  GIN with OGB AtomEncoder + BondEncoder (proper edge features)
  3.2  GIN + Virtual Node (OGB standard baseline)
  3.3  GIN + VN + edge features (full OGB baseline)
  3.4  Hybrid: GNN embedding + Lipinski → CatBoost
  3.5  Advanced fingerprints: MACCS + atom-pair + pharmacophore → CatBoost
  3.6  Ablation: which feature set matters most for CatBoost?

Research refs:
  [1] Hu et al. 2020 — OGB paper, AtomEncoder/BondEncoder for proper feature encoding
  [2] Gilmer et al. 2017 — MPNN framework, edge features critical for bond-aware message passing
  [3] Li et al. 2020 — Virtual nodes improve graph classification by enabling global info exchange
"""
import os, json, time, warnings, sys
warnings.filterwarnings('ignore')
os.environ['MPLBACKEND'] = 'Agg'

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GINConv, global_mean_pool, global_add_pool
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader as PyGDataLoader
from sklearn.metrics import roc_auc_score, average_precision_score
from pathlib import Path

# OGB
_orig_load = torch.load
def _patched_load(*a, **kw):
    kw.setdefault('weights_only', False)
    return _orig_load(*a, **kw)
torch.load = _patched_load

from ogb.graphproppred import GraphPropPredDataset, Evaluator
from ogb.graphproppred.mol_encoder import AtomEncoder, BondEncoder

BASE_DIR = Path('/Users/anthonyrodrigues/Desktop/YC-Portfolio-Projects/Drug-Molecule-Property-Prediction')
RESULTS_DIR = BASE_DIR / 'results'
RESULTS_DIR.mkdir(exist_ok=True)

DEVICE = torch.device('cpu')
HIDDEN_DIM = 128
NUM_LAYERS = 3
DROPOUT = 0.5
LR = 1e-3
EPOCHS = 40
PATIENCE = 12
BATCH_SIZE = 256

torch.manual_seed(42)
np.random.seed(42)

# ══════════════════════════════════════════════════════════════════════════
# DATA LOADING — OGB native format (proper atom + bond features)
# ══════════════════════════════════════════════════════════════════════════
print('Loading ogbg-molhiv via OGB...')
dataset = GraphPropPredDataset(name='ogbg-molhiv', root=str(BASE_DIR / 'data' / 'raw'))
split_idx = dataset.get_idx_split()
evaluator = Evaluator(name='ogbg-molhiv')

def ogb_to_pyg(idx_list):
    """Convert OGB dict format to PyG Data objects."""
    data_list = []
    for i in idx_list:
        graph, label = dataset[i]
        x = torch.tensor(graph['node_feat'], dtype=torch.long)
        edge_index = torch.tensor(graph['edge_index'], dtype=torch.long)
        edge_attr = torch.tensor(graph['edge_feat'], dtype=torch.long)
        y = torch.tensor([label[0]], dtype=torch.float)
        data_list.append(Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y))
    return data_list

train_data = ogb_to_pyg(split_idx['train'].tolist())
val_data = ogb_to_pyg(split_idx['valid'].tolist())
test_data = ogb_to_pyg(split_idx['test'].tolist())

train_loader = PyGDataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
val_loader = PyGDataLoader(val_data, batch_size=BATCH_SIZE)
test_loader = PyGDataLoader(test_data, batch_size=BATCH_SIZE)

print(f'  Train: {len(train_data):,} | Val: {len(val_data):,} | Test: {len(test_data):,}')
print(f'  Node features: 9 (encoded via AtomEncoder) | Edge features: 3 (encoded via BondEncoder)')

# ══════════════════════════════════════════════════════════════════════════
# MODEL DEFINITIONS
# ══════════════════════════════════════════════════════════════════════════

class GINEdge(nn.Module):
    """GIN with OGB AtomEncoder + BondEncoder for proper edge feature usage."""
    def __init__(self, hidden_dim, num_layers, dropout, use_virtual_node=False):
        super().__init__()
        self.num_layers = num_layers
        self.use_virtual_node = use_virtual_node
        self.atom_encoder = AtomEncoder(hidden_dim)
        self.bond_encoders = nn.ModuleList([BondEncoder(hidden_dim) for _ in range(num_layers)])

        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        for _ in range(num_layers):
            mlp = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
            )
            self.convs.append(GINConv(mlp, train_eps=True))
            self.bns.append(nn.BatchNorm1d(hidden_dim))

        if use_virtual_node:
            self.vn_embedding = nn.Embedding(1, hidden_dim)
            nn.init.constant_(self.vn_embedding.weight, 0)
            self.vn_mlps = nn.ModuleList()
            for _ in range(num_layers - 1):
                self.vn_mlps.append(nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.BatchNorm1d(hidden_dim),
                    nn.ReLU(),
                    nn.Linear(hidden_dim, hidden_dim),
                ))

        self.pool = global_mean_pool
        self.lin1 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.lin2 = nn.Linear(hidden_dim // 2, 1)
        self.dropout = dropout

    def forward(self, x, edge_index, edge_attr, batch):
        h = self.atom_encoder(x)

        if self.use_virtual_node:
            vn_h = self.vn_embedding(torch.zeros(batch.max().item() + 1, dtype=torch.long, device=x.device))

        for layer in range(self.num_layers):
            bond_emb = self.bond_encoders[layer](edge_attr)
            # Add bond embeddings to source node embeddings before message passing
            h_with_bond = h.clone()
            row, col = edge_index
            # Scatter bond info: for each edge, add bond emb to source node
            h_msg = h[row] + bond_emb
            # Use GINConv which aggregates neighbor messages
            h = self.convs[layer](h, edge_index)
            # Add bond contribution via residual
            h = h + torch.zeros_like(h).scatter_reduce(0, row.unsqueeze(-1).expand_as(bond_emb), bond_emb, reduce='mean')
            h = self.bns[layer](h)
            h = F.relu(h)
            h = F.dropout(h, p=self.dropout, training=self.training)

            if self.use_virtual_node and layer < self.num_layers - 1:
                # Virtual node aggregation
                vn_h_temp = global_mean_pool(h, batch) + vn_h
                vn_h = F.dropout(self.vn_mlps[layer](vn_h_temp), p=self.dropout, training=self.training)
                h = h + vn_h[batch]

        g = self.pool(h, batch)
        g = F.dropout(F.relu(self.lin1(g)), p=self.dropout, training=self.training)
        return self.lin2(g).squeeze(-1)


class GINEdgeSimple(nn.Module):
    """Simplified GIN with proper edge handling via edge-conditioned message passing."""
    def __init__(self, hidden_dim, num_layers, dropout, use_virtual_node=False):
        super().__init__()
        self.num_layers = num_layers
        self.use_virtual_node = use_virtual_node
        self.atom_encoder = AtomEncoder(hidden_dim)
        self.bond_encoders = nn.ModuleList([BondEncoder(hidden_dim) for _ in range(num_layers)])

        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        for _ in range(num_layers):
            mlp = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
            )
            self.convs.append(GINConv(mlp, train_eps=True))
            self.bns.append(nn.BatchNorm1d(hidden_dim))

        if use_virtual_node:
            self.vn_embedding = nn.Embedding(1, hidden_dim)
            nn.init.constant_(self.vn_embedding.weight, 0)
            self.vn_mlps = nn.ModuleList()
            for _ in range(num_layers - 1):
                self.vn_mlps.append(nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.BatchNorm1d(hidden_dim),
                    nn.ReLU(),
                    nn.Linear(hidden_dim, hidden_dim),
                ))

        self.pool = global_add_pool  # sum pooling like OGB baseline
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )
        self.dropout = dropout

    def forward(self, x, edge_index, edge_attr, batch):
        h = self.atom_encoder(x)

        if self.use_virtual_node:
            vn_h = self.vn_embedding(torch.zeros(batch.max().item() + 1, dtype=torch.long, device=x.device))

        for layer in range(self.num_layers):
            # Edge-conditioned: modify node features with bond info before conv
            bond_emb = self.bond_encoders[layer](edge_attr)
            row = edge_index[0]
            # Aggregate bond embeddings to each node
            bond_agg = torch.zeros_like(h)
            bond_agg.scatter_reduce_(0, row.unsqueeze(-1).expand_as(bond_emb), bond_emb, reduce='mean')

            h_in = h + bond_agg  # inject bond info
            h = self.convs[layer](h_in, edge_index)
            h = self.bns[layer](h)

            if layer < self.num_layers - 1:
                h = F.relu(h)
                h = F.dropout(h, p=self.dropout, training=self.training)

            if self.use_virtual_node and layer < self.num_layers - 1:
                vn_h_temp = global_add_pool(h, batch) + vn_h
                vn_h = F.dropout(self.vn_mlps[layer](vn_h_temp), p=self.dropout, training=self.training) + vn_h
                h = h + vn_h[batch]

        g = self.pool(h, batch)
        return self.classifier(g).squeeze(-1)


def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

# ══════════════════════════════════════════════════════════════════════════
# TRAINING LOOP
# ══════════════════════════════════════════════════════════════════════════

def train_epoch(model, loader, optimizer, device):
    model.train()
    total_loss, n = 0, 0
    pos_weight = torch.tensor([10.0], device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    for batch in loader:
        batch = batch.to(device)
        optimizer.zero_grad()
        logits = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
        labels = batch.y.view(-1)
        mask = ~torch.isnan(labels)
        loss = criterion(logits[mask], labels[mask])
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()
        total_loss += loss.item() * mask.sum().item()
        n += mask.sum().item()
    return total_loss / n


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    preds_all, labels_all = [], []
    for batch in loader:
        batch = batch.to(device)
        logits = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
        labels = batch.y.view(-1)
        mask = ~torch.isnan(labels)
        preds_all.append(torch.sigmoid(logits[mask]).cpu().numpy())
        labels_all.append(labels[mask].cpu().numpy())
    preds = np.concatenate(preds_all)
    labels = np.concatenate(labels_all)
    if labels.sum() == 0 or labels.sum() == len(labels):
        return 0.5, 0.0
    rocauc = roc_auc_score(labels, preds)
    auprc = average_precision_score(labels, preds)
    return rocauc, auprc


def train_gnn(model, name, epochs=EPOCHS, lr=LR, patience=PATIENCE):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=7, min_lr=1e-5)

    best_val, best_test, best_auprc = 0, 0, 0
    patience_count = 0
    history = {'val_auc': [], 'test_auc': [], 'train_loss': []}

    print(f'\n{"="*60}')
    print(f'Training {name} | {count_params(model):,} params')
    print(f'{"="*60}')
    t0 = time.time()

    for epoch in range(1, epochs + 1):
        loss = train_epoch(model, train_loader, optimizer, DEVICE)
        val_auc, val_ap = evaluate(model, val_loader, DEVICE)
        test_auc, test_ap = evaluate(model, test_loader, DEVICE)
        history['train_loss'].append(loss)
        history['val_auc'].append(val_auc)
        history['test_auc'].append(test_auc)
        scheduler.step(val_auc)

        if val_auc > best_val:
            best_val = val_auc
            best_test = test_auc
            best_auprc = test_ap
            patience_count = 0
        else:
            patience_count += 1

        if epoch % 10 == 0:
            print(f'  Ep {epoch:3d} | loss={loss:.4f} | val={val_auc:.4f} | test={test_auc:.4f} | [{time.time()-t0:.0f}s]')

        if patience_count >= patience:
            print(f'  Early stopping at epoch {epoch}')
            break

    elapsed = time.time() - t0
    print(f'  RESULT: val={best_val:.4f} test={best_test:.4f} auprc={best_auprc:.4f} ({elapsed:.0f}s)')

    return {
        'name': name, 'val_auc': round(best_val, 4), 'test_auc': round(best_test, 4),
        'auprc': round(best_auprc, 4), 'params': count_params(model),
        'epochs': len(history['val_auc']), 'time_s': round(elapsed, 1),
        'history': history,
    }


# ══════════════════════════════════════════════════════════════════════════
# EXPERIMENT 3.1: GIN + AtomEncoder + BondEncoder (no virtual node)
# ══════════════════════════════════════════════════════════════════════════
print('\n' + '#' * 60)
print('# Phase 3: Feature Engineering + Deep Dive')
print('#' * 60)

print('\n--- Exp 3.1: GIN + Edge Features (AtomEncoder + BondEncoder) ---')
model_3_1 = GINEdgeSimple(HIDDEN_DIM, NUM_LAYERS, DROPOUT, use_virtual_node=False).to(DEVICE)
result_3_1 = train_gnn(model_3_1, 'GIN+Edge (no VN)')

# ══════════════════════════════════════════════════════════════════════════
# EXPERIMENT 3.2: GIN + Virtual Node (no edge features)
# ══════════════════════════════════════════════════════════════════════════
print('\n--- Exp 3.2: GIN + Virtual Node (no edge features) ---')
# Use AtomEncoder but zero out bond contributions
class GINVNNoEdge(nn.Module):
    """GIN with virtual node but no edge features."""
    def __init__(self, hidden_dim, num_layers, dropout):
        super().__init__()
        self.num_layers = num_layers
        self.atom_encoder = AtomEncoder(hidden_dim)

        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        for _ in range(num_layers):
            mlp = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
            )
            self.convs.append(GINConv(mlp, train_eps=True))
            self.bns.append(nn.BatchNorm1d(hidden_dim))

        self.vn_embedding = nn.Embedding(1, hidden_dim)
        nn.init.constant_(self.vn_embedding.weight, 0)
        self.vn_mlps = nn.ModuleList()
        for _ in range(num_layers - 1):
            self.vn_mlps.append(nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
            ))

        self.pool = global_add_pool
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )
        self.dropout = dropout

    def forward(self, x, edge_index, edge_attr, batch):
        h = self.atom_encoder(x)
        vn_h = self.vn_embedding(torch.zeros(batch.max().item() + 1, dtype=torch.long, device=x.device))

        for layer in range(self.num_layers):
            h = self.convs[layer](h, edge_index)
            h = self.bns[layer](h)
            if layer < self.num_layers - 1:
                h = F.relu(h)
                h = F.dropout(h, p=self.dropout, training=self.training)
                vn_h_temp = global_add_pool(h, batch) + vn_h
                vn_h = F.dropout(self.vn_mlps[layer](vn_h_temp), p=self.dropout, training=self.training) + vn_h
                h = h + vn_h[batch]

        g = self.pool(h, batch)
        return self.classifier(g).squeeze(-1)

model_3_2 = GINVNNoEdge(HIDDEN_DIM, NUM_LAYERS, DROPOUT).to(DEVICE)
result_3_2 = train_gnn(model_3_2, 'GIN+VN (no edge)')

# ══════════════════════════════════════════════════════════════════════════
# EXPERIMENT 3.3: GIN + Edge Features + Virtual Node (full OGB baseline)
# ══════════════════════════════════════════════════════════════════════════
print('\n--- Exp 3.3: GIN + Edge + Virtual Node (full OGB baseline) ---')
model_3_3 = GINEdgeSimple(HIDDEN_DIM, NUM_LAYERS, DROPOUT, use_virtual_node=True).to(DEVICE)
result_3_3 = train_gnn(model_3_3, 'GIN+Edge+VN (OGB)')

# ══════════════════════════════════════════════════════════════════════════
# EXPERIMENT 3.4: Hybrid — GNN embedding + Lipinski → CatBoost
# ══════════════════════════════════════════════════════════════════════════
print('\n--- Exp 3.4: Hybrid GNN embedding + Lipinski → CatBoost ---')

@torch.no_grad()
def extract_gnn_embeddings(model, data_list, device):
    """Extract graph-level embeddings from a trained GNN (before classifier)."""
    model.eval()
    loader = PyGDataLoader(data_list, batch_size=256)
    all_emb = []
    for batch in loader:
        batch = batch.to(device)
        h = model.atom_encoder(batch.x)

        if hasattr(model, 'vn_embedding'):
            vn_h = model.vn_embedding(torch.zeros(batch.batch.max().item() + 1, dtype=torch.long, device=device))

        for layer in range(model.num_layers):
            if hasattr(model, 'bond_encoders'):
                bond_emb = model.bond_encoders[layer](batch.edge_attr)
                row = batch.edge_index[0]
                bond_agg = torch.zeros_like(h)
                bond_agg.scatter_reduce_(0, row.unsqueeze(-1).expand_as(bond_emb), bond_emb, reduce='mean')
                h_in = h + bond_agg
            else:
                h_in = h

            h = model.convs[layer](h_in, batch.edge_index)
            h = model.bns[layer](h)
            if layer < model.num_layers - 1:
                h = F.relu(h)
                if hasattr(model, 'vn_embedding') and layer < model.num_layers - 1:
                    vn_h_temp = global_add_pool(h, batch.batch) + vn_h
                    vn_h = model.vn_mlps[layer](vn_h_temp) + vn_h
                    h = h + vn_h[batch.batch]

        g = model.pool(h, batch.batch)
        all_emb.append(g.cpu().numpy())
    return np.concatenate(all_emb, axis=0)

# Use the best GNN model to extract embeddings
print('  Extracting GNN embeddings from best model...')
best_gnn = model_3_3  # GIN+Edge+VN
train_emb = extract_gnn_embeddings(best_gnn, train_data, DEVICE)
val_emb = extract_gnn_embeddings(best_gnn, val_data, DEVICE)
test_emb = extract_gnn_embeddings(best_gnn, test_data, DEVICE)
print(f'  Embedding shape: {train_emb.shape}')

# Get Lipinski features
sys.path.insert(0, str(BASE_DIR))
from src.data_pipeline import download_ogbg_molhiv, compute_lipinski_features, compute_morgan_fingerprints
print('  Loading Lipinski + Morgan features...')
df = download_ogbg_molhiv()
lip_features = []
fp_features = []
valid_idx = []
for i, smi in enumerate(df['smiles']):
    lip = compute_lipinski_features(smi)
    fp = compute_morgan_fingerprints(smi, radius=2, n_bits=1024)
    if lip is not None and fp is not None:
        lip_features.append(list(lip.values()))
        fp_features.append(fp)
        valid_idx.append(i)

lip_arr = np.array(lip_features)
fp_arr = np.array(fp_features)
lip_cols = list(compute_lipinski_features(df['smiles'].iloc[0]).keys())

# Map OGB splits to feature arrays
split_idx_data = dataset.get_idx_split()
idx_to_pos = {idx: pos for pos, idx in enumerate(valid_idx)}

def get_features_for_split(ogb_indices, lip_arr, fp_arr, emb_arr):
    """Combine features for a split. emb_arr is already split-aligned."""
    lips, fps = [], []
    emb_idx = 0
    valid_emb = []
    for i in ogb_indices:
        if i in idx_to_pos:
            lips.append(lip_arr[idx_to_pos[i]])
            fps.append(fp_arr[idx_to_pos[i]])
            valid_emb.append(emb_arr[emb_idx])
        emb_idx += 1
    return np.array(lips), np.array(fps), np.array(valid_emb)

train_lip, train_fp, train_emb_valid = get_features_for_split(
    split_idx_data['train'].tolist(), lip_arr, fp_arr, train_emb)
val_lip, val_fp, val_emb_valid = get_features_for_split(
    split_idx_data['valid'].tolist(), lip_arr, fp_arr, val_emb)
test_lip, test_fp, test_emb_valid = get_features_for_split(
    split_idx_data['test'].tolist(), lip_arr, fp_arr, test_emb)

train_labels = np.array([dataset[i][1][0] for i in split_idx_data['train'].tolist()])
val_labels = np.array([dataset[i][1][0] for i in split_idx_data['valid'].tolist()])
test_labels = np.array([dataset[i][1][0] for i in split_idx_data['test'].tolist()])

# Trim to match valid features
min_train = min(len(train_lip), len(train_emb_valid))
min_val = min(len(val_lip), len(val_emb_valid))
min_test = min(len(test_lip), len(test_emb_valid))

from catboost import CatBoostClassifier

# 3.4a: GNN embedding only → CatBoost
print('  3.4a: GNN embedding → CatBoost')
cb_emb = CatBoostClassifier(
    iterations=500, depth=6, learning_rate=0.05, auto_class_weights='Balanced',
    verbose=0, random_seed=42, eval_metric='AUC',
)
cb_emb.fit(train_emb_valid[:min_train], train_labels[:min_train],
           eval_set=(val_emb_valid[:min_val], val_labels[:min_val]),
           early_stopping_rounds=50)
pred_emb = cb_emb.predict_proba(test_emb_valid[:min_test])[:, 1]
auc_emb = roc_auc_score(test_labels[:min_test], pred_emb)
ap_emb = average_precision_score(test_labels[:min_test], pred_emb)
print(f'    GNN-embed → CatBoost: AUC={auc_emb:.4f}, AUPRC={ap_emb:.4f}')

# 3.4b: GNN embedding + Lipinski → CatBoost
print('  3.4b: GNN embedding + Lipinski → CatBoost')
X_train_hybrid = np.hstack([train_emb_valid[:min_train], train_lip[:min_train]])
X_val_hybrid = np.hstack([val_emb_valid[:min_val], val_lip[:min_val]])
X_test_hybrid = np.hstack([test_emb_valid[:min_test], test_lip[:min_test]])

cb_hybrid = CatBoostClassifier(
    iterations=500, depth=6, learning_rate=0.05, auto_class_weights='Balanced',
    verbose=0, random_seed=42, eval_metric='AUC',
)
cb_hybrid.fit(X_train_hybrid, train_labels[:min_train],
              eval_set=(X_val_hybrid, val_labels[:min_val]),
              early_stopping_rounds=50)
pred_hybrid = cb_hybrid.predict_proba(X_test_hybrid)[:, 1]
auc_hybrid = roc_auc_score(test_labels[:min_test], pred_hybrid)
ap_hybrid = average_precision_score(test_labels[:min_test], pred_hybrid)
print(f'    GNN-embed+Lipinski → CatBoost: AUC={auc_hybrid:.4f}, AUPRC={ap_hybrid:.4f}')

# 3.4c: GNN embedding + Lipinski + Morgan FP → CatBoost
print('  3.4c: Full hybrid: GNN-embed + Lipinski + Morgan → CatBoost')
X_train_full = np.hstack([train_emb_valid[:min_train], train_lip[:min_train], train_fp[:min_train]])
X_val_full = np.hstack([val_emb_valid[:min_val], val_lip[:min_val], val_fp[:min_val]])
X_test_full = np.hstack([test_emb_valid[:min_test], test_lip[:min_test], test_fp[:min_test]])

cb_full = CatBoostClassifier(
    iterations=500, depth=6, learning_rate=0.05, auto_class_weights='Balanced',
    verbose=0, random_seed=42, eval_metric='AUC',
)
cb_full.fit(X_train_full, train_labels[:min_train],
            eval_set=(X_val_full, val_labels[:min_val]),
            early_stopping_rounds=50)
pred_full = cb_full.predict_proba(X_test_full)[:, 1]
auc_full = roc_auc_score(test_labels[:min_test], pred_full)
ap_full = average_precision_score(test_labels[:min_test], pred_full)
print(f'    Full hybrid → CatBoost: AUC={auc_full:.4f}, AUPRC={ap_full:.4f}')

result_3_4a = {'name': 'GNN-embed→CatBoost', 'test_auc': round(auc_emb, 4), 'auprc': round(ap_emb, 4)}
result_3_4b = {'name': 'GNN+Lip→CatBoost', 'test_auc': round(auc_hybrid, 4), 'auprc': round(ap_hybrid, 4)}
result_3_4c = {'name': 'GNN+Lip+FP→CatBoost', 'test_auc': round(auc_full, 4), 'auprc': round(ap_full, 4)}

# ══════════════════════════════════════════════════════════════════════════
# EXPERIMENT 3.5: Advanced Fingerprints → CatBoost
# ══════════════════════════════════════════════════════════════════════════
print('\n--- Exp 3.5: Advanced Fingerprints → CatBoost ---')
from rdkit import Chem
from rdkit.Chem import MACCSkeys, AllChem, rdMolDescriptors

def compute_advanced_features(smiles):
    """Compute MACCS keys, atom-pair FP, topological torsion, pharmacophore features."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    # MACCS keys (166 bits)
    maccs = np.array(MACCSkeys.GenMACCSKeys(mol))
    # Atom-pair fingerprint (2048 bits)
    ap_fp = np.array(AllChem.GetHashedAtomPairFingerprintAsBitVect(mol, nBits=1024))
    # Topological torsion (2048 bits)
    tt_fp = np.array(AllChem.GetHashedTopologicalTorsionFingerprintAsBitVect(mol, nBits=1024))
    # Additional descriptors beyond Lipinski
    extra = np.array([
        rdMolDescriptors.CalcNumAmideBonds(mol),
        rdMolDescriptors.CalcNumBridgeheadAtoms(mol),
        rdMolDescriptors.CalcNumSpiroAtoms(mol),
        rdMolDescriptors.CalcNumHeteroatoms(mol),
        rdMolDescriptors.CalcNumAliphaticCarbocycles(mol),
        rdMolDescriptors.CalcNumAliphaticHeterocycles(mol),
        rdMolDescriptors.CalcNumAromaticHeterocycles(mol),
        rdMolDescriptors.CalcNumSaturatedCarbocycles(mol),
        rdMolDescriptors.CalcNumSaturatedHeterocycles(mol),
        rdMolDescriptors.CalcLabuteASA(mol),  # ASA
        rdMolDescriptors.CalcExactMolWt(mol),
        Chem.Descriptors.BertzCT(mol),  # topological complexity
    ], dtype=np.float32)
    return np.concatenate([maccs, ap_fp, tt_fp, extra])

print('  Computing advanced features for all molecules...')
adv_features = {}
for i, smi in enumerate(df['smiles']):
    feat = compute_advanced_features(smi)
    if feat is not None:
        adv_features[i] = feat

def get_adv_for_split(ogb_indices):
    feats, labels = [], []
    for i in ogb_indices:
        if i in adv_features and i in idx_to_pos:
            feats.append(adv_features[i])
            labels.append(dataset[i][1][0])
    return np.array(feats), np.array(labels)

train_adv, y_train_adv = get_adv_for_split(split_idx_data['train'].tolist())
val_adv, y_val_adv = get_adv_for_split(split_idx_data['valid'].tolist())
test_adv, y_test_adv = get_adv_for_split(split_idx_data['test'].tolist())
print(f'  Advanced feature shape: {train_adv.shape}')

# 3.5a: Advanced FP only
print('  3.5a: Advanced FP → CatBoost')
cb_adv = CatBoostClassifier(
    iterations=500, depth=6, learning_rate=0.05, auto_class_weights='Balanced',
    verbose=0, random_seed=42, eval_metric='AUC',
)
cb_adv.fit(train_adv, y_train_adv, eval_set=(val_adv, y_val_adv), early_stopping_rounds=50)
pred_adv = cb_adv.predict_proba(test_adv)[:, 1]
auc_adv = roc_auc_score(y_test_adv, pred_adv)
ap_adv = average_precision_score(y_test_adv, pred_adv)
print(f'    Advanced FP → CatBoost: AUC={auc_adv:.4f}, AUPRC={ap_adv:.4f}')

# 3.5b: Lipinski + Advanced FP combined
print('  3.5b: Lipinski + Advanced FP → CatBoost')
train_lip_adv = np.hstack([train_lip[:len(train_adv)], train_adv])
val_lip_adv = np.hstack([val_lip[:len(val_adv)], val_adv])
test_lip_adv = np.hstack([test_lip[:len(test_adv)], test_adv])

cb_lip_adv = CatBoostClassifier(
    iterations=500, depth=6, learning_rate=0.05, auto_class_weights='Balanced',
    verbose=0, random_seed=42, eval_metric='AUC',
)
cb_lip_adv.fit(train_lip_adv, y_train_adv, eval_set=(val_lip_adv, y_val_adv), early_stopping_rounds=50)
pred_lip_adv = cb_lip_adv.predict_proba(test_lip_adv)[:, 1]
auc_lip_adv = roc_auc_score(y_test_adv, pred_lip_adv)
ap_lip_adv = average_precision_score(y_test_adv, pred_lip_adv)
print(f'    Lipinski+AdvFP → CatBoost: AUC={auc_lip_adv:.4f}, AUPRC={ap_lip_adv:.4f}')

result_3_5a = {'name': 'AdvFP→CatBoost', 'test_auc': round(auc_adv, 4), 'auprc': round(ap_adv, 4)}
result_3_5b = {'name': 'Lip+AdvFP→CatBoost', 'test_auc': round(auc_lip_adv, 4), 'auprc': round(ap_lip_adv, 4)}

# ══════════════════════════════════════════════════════════════════════════
# EXPERIMENT 3.6: Ablation — What matters for CatBoost?
# ══════════════════════════════════════════════════════════════════════════
print('\n--- Exp 3.6: Feature Ablation Study ---')

# Lipinski only (Phase 1 baseline, rerun with same setup)
cb_lip_only = CatBoostClassifier(
    iterations=500, depth=6, learning_rate=0.05, auto_class_weights='Balanced',
    verbose=0, random_seed=42, eval_metric='AUC',
)
cb_lip_only.fit(train_lip[:min_train], train_labels[:min_train],
                eval_set=(val_lip[:min_val], val_labels[:min_val]),
                early_stopping_rounds=50)
pred_lip_only = cb_lip_only.predict_proba(test_lip[:min_test])[:, 1]
auc_lip_only = roc_auc_score(test_labels[:min_test], pred_lip_only)
ap_lip_only = average_precision_score(test_labels[:min_test], pred_lip_only)

# Morgan FP only
cb_fp_only = CatBoostClassifier(
    iterations=500, depth=6, learning_rate=0.05, auto_class_weights='Balanced',
    verbose=0, random_seed=42, eval_metric='AUC',
)
cb_fp_only.fit(train_fp[:min_train], train_labels[:min_train],
               eval_set=(val_fp[:min_val], val_labels[:min_val]),
               early_stopping_rounds=50)
pred_fp_only = cb_fp_only.predict_proba(test_fp[:min_test])[:, 1]
auc_fp_only = roc_auc_score(test_labels[:min_test], pred_fp_only)
ap_fp_only = average_precision_score(test_labels[:min_test], pred_fp_only)

# Lipinski + Morgan FP (Phase 1 combined)
X_train_lipfp = np.hstack([train_lip[:min_train], train_fp[:min_train]])
X_val_lipfp = np.hstack([val_lip[:min_val], val_fp[:min_val]])
X_test_lipfp = np.hstack([test_lip[:min_test], test_fp[:min_test]])

cb_lipfp = CatBoostClassifier(
    iterations=500, depth=6, learning_rate=0.05, auto_class_weights='Balanced',
    verbose=0, random_seed=42, eval_metric='AUC',
)
cb_lipfp.fit(X_train_lipfp, train_labels[:min_train],
             eval_set=(X_val_lipfp, val_labels[:min_val]),
             early_stopping_rounds=50)
pred_lipfp = cb_lipfp.predict_proba(X_test_lipfp)[:, 1]
auc_lipfp = roc_auc_score(test_labels[:min_test], pred_lipfp)
ap_lipfp = average_precision_score(test_labels[:min_test], pred_lipfp)

print(f'  Lipinski only:     AUC={auc_lip_only:.4f}')
print(f'  Morgan FP only:    AUC={auc_fp_only:.4f}')
print(f'  Lipinski+Morgan:   AUC={auc_lipfp:.4f}')
print(f'  Advanced FP:       AUC={auc_adv:.4f}')
print(f'  Lip+AdvFP:         AUC={auc_lip_adv:.4f}')
print(f'  GNN-embed:         AUC={auc_emb:.4f}')
print(f'  GNN+Lip:           AUC={auc_hybrid:.4f}')
print(f'  GNN+Lip+FP (full): AUC={auc_full:.4f}')

# ══════════════════════════════════════════════════════════════════════════
# SUMMARY TABLE & PLOTS
# ══════════════════════════════════════════════════════════════════════════
print('\n' + '=' * 90)
print('PHASE 3 HEAD-TO-HEAD COMPARISON')
print('=' * 90)

all_results = [
    # GNN experiments
    result_3_1, result_3_2, result_3_3,
    # Hybrid experiments
    result_3_4a, result_3_4b, result_3_4c,
    # Feature experiments
    result_3_5a, result_3_5b,
    # Ablation
    {'name': 'Lipinski→CB', 'test_auc': round(auc_lip_only, 4), 'auprc': round(ap_lip_only, 4)},
    {'name': 'MorganFP→CB', 'test_auc': round(auc_fp_only, 4), 'auprc': round(ap_fp_only, 4)},
    {'name': 'Lip+FP→CB', 'test_auc': round(auc_lipfp, 4), 'auprc': round(ap_lipfp, 4)},
    # Phase 1-2 baselines
    {'name': 'Phase1 CatBoost (Mark)', 'test_auc': 0.7782, 'auprc': None},
    {'name': 'Phase1 RF+Combined', 'test_auc': 0.7707, 'auprc': 0.3722},
    {'name': 'Phase2 GIN (raw)', 'test_auc': 0.7053, 'auprc': None},
    {'name': 'Phase2 MLP-Domain9 (Mark)', 'test_auc': 0.7670, 'auprc': None},
    {'name': 'SOTA (DeeperGCN+VN)', 'test_auc': 0.8476, 'auprc': None},
]

all_results_sorted = sorted(all_results, key=lambda r: r['test_auc'], reverse=True)

print(f'{"Rank":<5} {"Model":<30} {"Test AUC":>10} {"AUPRC":>10}')
print('-' * 60)
for rank, r in enumerate(all_results_sorted, 1):
    auprc_str = f"{r['auprc']:.4f}" if r.get('auprc') is not None else '—'
    print(f'{rank:<5} {r["name"]:<30} {r["test_auc"]:>10.4f} {auprc_str:>10}')

# ── Plots ──────────────────────────────────────────────────────────────
# 1. Comprehensive model comparison bar chart
fig, ax = plt.subplots(figsize=(14, 8))
plot_results = [r for r in all_results_sorted if r['name'] != 'SOTA (DeeperGCN+VN)']
names = [r['name'] for r in plot_results]
aucs = [r['test_auc'] for r in plot_results]

# Color by type
colors = []
for r in plot_results:
    if 'GNN' in r['name'] or 'GIN' in r['name']:
        colors.append('#4472C4')  # blue for GNN
    elif 'CatBoost' in r['name'] or 'CB' in r['name']:
        colors.append('#70AD47')  # green for traditional
    elif 'Phase' in r['name'] or 'MLP' in r['name']:
        colors.append('#FFC000')  # gold for baselines
    else:
        colors.append('#ED7D31')  # orange for other

bars = ax.barh(names[::-1], aucs[::-1], color=colors[::-1], edgecolor='white', height=0.6, alpha=0.85)
for bar, auc in zip(bars, aucs[::-1]):
    ax.text(bar.get_width() + 0.002, bar.get_y() + bar.get_height()/2, f'{auc:.4f}',
            va='center', fontsize=10, fontweight='bold')

ax.axvline(0.8476, color='red', ls=':', lw=2, alpha=0.7, label='SOTA (0.8476)')
ax.axvline(0.7782, color='gray', ls='--', lw=1.5, alpha=0.7, label='Phase 1 CatBoost (0.7782)')
ax.set(xlabel='Test ROC-AUC', title='Phase 3: Feature Engineering — All Experiments\nogbg-molhiv (scaffold split)')
ax.set_xlim(0.60, 0.90)
ax.grid(axis='x', alpha=0.3)
from matplotlib.patches import Patch
ax.legend(handles=[
    Patch(fc='#4472C4', label='GNN models'), Patch(fc='#70AD47', label='Feature→CatBoost'),
    Patch(fc='#FFC000', label='Prior baselines'), Patch(fc='red', alpha=0.7, label='SOTA'),
], loc='lower right')
plt.tight_layout()
plt.savefig(RESULTS_DIR / 'phase3_model_comparison.png', dpi=150, bbox_inches='tight')
print('\nSaved: phase3_model_comparison.png')

# 2. GNN training curves
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
gnn_results = [result_3_1, result_3_2, result_3_3]
gnn_colors = ['#4472C4', '#ED7D31', '#70AD47']
for r, c in zip(gnn_results, gnn_colors):
    axes[0].plot(r['history']['val_auc'], label=f'{r["name"]} ({r["test_auc"]:.4f})', color=c, lw=2)
    axes[1].plot(r['history']['train_loss'], label=r['name'], color=c, lw=2)
axes[0].axhline(0.7782, color='gray', ls='--', lw=1.5, alpha=0.7, label='CatBoost (0.7782)')
axes[0].axhline(0.8476, color='red', ls=':', lw=2, alpha=0.7, label='SOTA (0.8476)')
axes[0].set(xlabel='Epoch', ylabel='Val ROC-AUC', title='Phase 3 GNN Training — Validation AUC')
axes[0].legend(fontsize=8); axes[0].set_ylim(0.5, 0.90); axes[0].grid(alpha=0.3)
axes[1].set(xlabel='Epoch', ylabel='Training Loss', title='Phase 3 GNN Training — Loss')
axes[1].legend(fontsize=9); axes[1].grid(alpha=0.3)
plt.tight_layout()
plt.savefig(RESULTS_DIR / 'phase3_gnn_training.png', dpi=150, bbox_inches='tight')
print('Saved: phase3_gnn_training.png')

# 3. Feature ablation bar chart
fig, ax = plt.subplots(figsize=(10, 6))
ablation_names = ['Lipinski (14)', 'Morgan FP (1024)', 'Lip+FP (1038)',
                  'Adv FP (2226)', 'Lip+AdvFP (2240)',
                  'GNN-embed (256)', 'GNN+Lip (270)', 'GNN+Lip+FP (1294)']
ablation_aucs = [auc_lip_only, auc_fp_only, auc_lipfp,
                 auc_adv, auc_lip_adv,
                 auc_emb, auc_hybrid, auc_full]
ablation_colors = ['#70AD47']*5 + ['#4472C4']*3
bars = ax.bar(range(len(ablation_names)), ablation_aucs, color=ablation_colors, edgecolor='white', alpha=0.85)
for bar, auc in zip(bars, ablation_aucs):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.003, f'{auc:.4f}',
            ha='center', va='bottom', fontsize=9, fontweight='bold')
ax.axhline(0.7782, color='gray', ls='--', lw=1.5, alpha=0.7, label='Phase 1 CatBoost')
ax.axhline(0.8476, color='red', ls=':', lw=2, alpha=0.7, label='SOTA')
ax.set_xticks(range(len(ablation_names)))
ax.set_xticklabels(ablation_names, rotation=45, ha='right', fontsize=9)
ax.set(ylabel='Test ROC-AUC', title='Feature Ablation Study: What Matters for HIV Activity Prediction?')
ax.set_ylim(0.60, 0.90)
ax.grid(axis='y', alpha=0.3)
ax.legend()
plt.tight_layout()
plt.savefig(RESULTS_DIR / 'phase3_ablation.png', dpi=150, bbox_inches='tight')
print('Saved: phase3_ablation.png')

# ── Save Results ──────────────────────────────────────────────────────
phase3_data = {
    'gnn_experiments': {
        r['name']: {'test_auc': r['test_auc'], 'auprc': r.get('auprc', None),
                     'params': r.get('params', None), 'time_s': r.get('time_s', None)}
        for r in [result_3_1, result_3_2, result_3_3]
    },
    'hybrid_experiments': {
        r['name']: {'test_auc': r['test_auc'], 'auprc': r['auprc']}
        for r in [result_3_4a, result_3_4b, result_3_4c]
    },
    'feature_experiments': {
        r['name']: {'test_auc': r['test_auc'], 'auprc': r['auprc']}
        for r in [result_3_5a, result_3_5b]
    },
    'ablation': {
        'lipinski_only': round(auc_lip_only, 4),
        'morgan_fp_only': round(auc_fp_only, 4),
        'lip_plus_fp': round(auc_lipfp, 4),
        'adv_fp': round(auc_adv, 4),
        'lip_plus_adv_fp': round(auc_lip_adv, 4),
        'gnn_embed': round(auc_emb, 4),
        'gnn_plus_lip': round(auc_hybrid, 4),
        'gnn_plus_lip_plus_fp': round(auc_full, 4),
    },
    'champion': all_results_sorted[0],
}

metrics_path = RESULTS_DIR / 'metrics.json'
with open(metrics_path) as f:
    metrics = json.load(f)
metrics['phase3_feature_engineering'] = phase3_data
with open(metrics_path, 'w') as f:
    json.dump(metrics, f, indent=2)

with open(RESULTS_DIR / 'phase3_results.json', 'w') as f:
    json.dump(phase3_data, f, indent=2)

print('\n' + '=' * 60)
champion = all_results_sorted[0]
print(f'Phase 3 Champion: {champion["name"]} (AUC = {champion["test_auc"]:.4f})')
print(f'Gap to SOTA: {0.8476 - champion["test_auc"]:+.4f}')
print(f'Δ vs Phase 1 CatBoost: {champion["test_auc"] - 0.7782:+.4f}')
print(f'Δ vs Phase 2 GIN: {champion["test_auc"] - 0.7053:+.4f}')
print('=' * 60)
print('\nDONE — Phase 3 complete.')
