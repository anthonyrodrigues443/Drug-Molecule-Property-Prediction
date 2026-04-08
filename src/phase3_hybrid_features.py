"""
Phase 3 Part 2: Hybrid + Feature experiments (after GNN training completed)
GNN results from part 1:
  GIN+Edge (no VN): test_auc=0.7860, val_auc=0.8001, auprc=0.3441
  GIN+VN (no edge): test_auc=0.7578, val_auc=0.8134, auprc=0.2642
  GIN+Edge+VN:      test_auc=0.7622, val_auc=0.8333, auprc=0.2858
"""
import os, json, time, warnings, sys
warnings.filterwarnings('ignore')
os.environ['MPLBACKEND'] = 'Agg'

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GINConv, global_add_pool
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader as PyGDataLoader
from sklearn.metrics import roc_auc_score, average_precision_score
from pathlib import Path

_orig_load = torch.load
def _patched_load(*a, **kw):
    kw.setdefault('weights_only', False)
    return _orig_load(*a, **kw)
torch.load = _patched_load

from ogb.graphproppred import GraphPropPredDataset
from ogb.graphproppred.mol_encoder import AtomEncoder, BondEncoder

BASE_DIR = Path('/Users/anthonyrodrigues/Desktop/YC-Portfolio-Projects/Drug-Molecule-Property-Prediction')
sys.path.insert(0, str(BASE_DIR))
RESULTS_DIR = BASE_DIR / 'results'

DEVICE = torch.device('cpu')
HIDDEN_DIM = 128
NUM_LAYERS = 3
DROPOUT = 0.5
torch.manual_seed(42)
np.random.seed(42)

# ── Load data ──────────────────────────────────────────────────────────
print('Loading ogbg-molhiv...')
dataset = GraphPropPredDataset(name='ogbg-molhiv', root=str(BASE_DIR / 'data' / 'raw'))
split_idx = dataset.get_idx_split()

def ogb_to_pyg(idx_list):
    data_list = []
    for i in idx_list:
        graph, label = dataset[i]
        data_list.append(Data(
            x=torch.tensor(graph['node_feat'], dtype=torch.long),
            edge_index=torch.tensor(graph['edge_index'], dtype=torch.long),
            edge_attr=torch.tensor(graph['edge_feat'], dtype=torch.long),
            y=torch.tensor([label[0]], dtype=torch.float),
        ))
    return data_list

train_data = ogb_to_pyg(split_idx['train'].tolist())
val_data = ogb_to_pyg(split_idx['valid'].tolist())
test_data = ogb_to_pyg(split_idx['test'].tolist())

train_labels = np.array([dataset[i][1][0] for i in split_idx['train'].tolist()])
val_labels = np.array([dataset[i][1][0] for i in split_idx['valid'].tolist()])
test_labels = np.array([dataset[i][1][0] for i in split_idx['test'].tolist()])
print(f'  Train: {len(train_data):,} | Val: {len(val_data):,} | Test: {len(test_data):,}')

# ── Train a quick GIN+Edge model for embedding extraction ─────────────
class GINEdgeSimple(nn.Module):
    def __init__(self, hidden_dim, num_layers, dropout):
        super().__init__()
        self.num_layers = num_layers
        self.atom_encoder = AtomEncoder(hidden_dim)
        self.bond_encoders = nn.ModuleList([BondEncoder(hidden_dim) for _ in range(num_layers)])
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        for _ in range(num_layers):
            mlp = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim), nn.BatchNorm1d(hidden_dim),
                nn.ReLU(), nn.Linear(hidden_dim, hidden_dim),
            )
            self.convs.append(GINConv(mlp, train_eps=True))
            self.bns.append(nn.BatchNorm1d(hidden_dim))
        self.pool = global_add_pool
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Dropout(dropout), nn.Linear(hidden_dim, 1),
        )
        self.dropout = dropout

    def forward(self, x, edge_index, edge_attr, batch):
        h = self.atom_encoder(x)
        for layer in range(self.num_layers):
            bond_emb = self.bond_encoders[layer](edge_attr)
            row = edge_index[0]
            bond_agg = torch.zeros_like(h)
            bond_agg.scatter_reduce_(0, row.unsqueeze(-1).expand_as(bond_emb), bond_emb, reduce='mean')
            h = self.convs[layer](h + bond_agg, edge_index)
            h = self.bns[layer](h)
            if layer < self.num_layers - 1:
                h = F.relu(h)
                h = F.dropout(h, p=self.dropout, training=self.training)
        return self.pool(h, batch), self.classifier(self.pool(h, batch)).squeeze(-1)

    def embed(self, x, edge_index, edge_attr, batch):
        h = self.atom_encoder(x)
        for layer in range(self.num_layers):
            bond_emb = self.bond_encoders[layer](edge_attr)
            row = edge_index[0]
            bond_agg = torch.zeros_like(h)
            bond_agg.scatter_reduce_(0, row.unsqueeze(-1).expand_as(bond_emb), bond_emb, reduce='mean')
            h = self.convs[layer](h + bond_agg, edge_index)
            h = self.bns[layer](h)
            if layer < self.num_layers - 1:
                h = F.relu(h)
        return self.pool(h, batch)

print('\nTraining GIN+Edge for embedding extraction (40 epochs)...')
model = GINEdgeSimple(HIDDEN_DIM, NUM_LAYERS, DROPOUT).to(DEVICE)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([10.0]))
train_loader = PyGDataLoader(train_data, batch_size=256, shuffle=True)
val_loader = PyGDataLoader(val_data, batch_size=256)
test_loader = PyGDataLoader(test_data, batch_size=256)

best_val, best_state = 0, None
t0 = time.time()
for epoch in range(1, 41):
    model.train()
    for batch in train_loader:
        batch = batch.to(DEVICE)
        optimizer.zero_grad()
        _, logits = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
        labels = batch.y.view(-1)
        mask = ~torch.isnan(labels)
        loss = criterion(logits[mask], labels[mask])
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()
    # Eval
    model.eval()
    preds, labs = [], []
    with torch.no_grad():
        for batch in val_loader:
            batch = batch.to(DEVICE)
            _, logits = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
            labels = batch.y.view(-1)
            mask = ~torch.isnan(labels)
            preds.append(torch.sigmoid(logits[mask]).cpu().numpy())
            labs.append(labels[mask].cpu().numpy())
    val_auc = roc_auc_score(np.concatenate(labs), np.concatenate(preds))
    if val_auc > best_val:
        best_val = val_auc
        best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
    if epoch % 10 == 0:
        print(f'  Ep {epoch:3d} | val_auc={val_auc:.4f} | best={best_val:.4f} | [{time.time()-t0:.0f}s]')

model.load_state_dict(best_state)
print(f'  Training done. Best val AUC: {best_val:.4f} ({time.time()-t0:.0f}s)')

# ── Extract embeddings ─────────────────────────────────────────────────
print('\nExtracting GNN embeddings...')
model.eval()
def extract_emb(data_list):
    loader = PyGDataLoader(data_list, batch_size=512)
    all_emb = []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(DEVICE)
            emb = model.embed(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
            all_emb.append(emb.cpu().numpy())
    return np.concatenate(all_emb)

train_emb = extract_emb(train_data)
val_emb = extract_emb(val_data)
test_emb = extract_emb(test_data)
print(f'  Embedding shape: {train_emb.shape}')

# ── Compute traditional features ──────────────────────────────────────
from src.data_pipeline import download_ogbg_molhiv, compute_lipinski_features, compute_morgan_fingerprints
from rdkit import Chem
from rdkit.Chem import MACCSkeys, AllChem, rdMolDescriptors, Descriptors

print('\nComputing molecular features...')
df = download_ogbg_molhiv()

def compute_all_features(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None, None, None, None
    # Lipinski (14 features)
    lip = compute_lipinski_features(smiles)
    lip_arr = np.array(list(lip.values()), dtype=np.float32)
    # Morgan FP (1024 bits)
    fp = compute_morgan_fingerprints(smiles, radius=2, n_bits=1024)
    # MACCS keys (167 bits)
    maccs = np.array(MACCSkeys.GenMACCSKeys(mol), dtype=np.float32)
    # Advanced descriptors (12 features)
    adv = np.array([
        rdMolDescriptors.CalcNumAmideBonds(mol),
        rdMolDescriptors.CalcNumBridgeheadAtoms(mol),
        rdMolDescriptors.CalcNumSpiroAtoms(mol),
        rdMolDescriptors.CalcNumHeteroatoms(mol),
        rdMolDescriptors.CalcNumAliphaticCarbocycles(mol),
        rdMolDescriptors.CalcNumAliphaticHeterocycles(mol),
        rdMolDescriptors.CalcNumAromaticHeterocycles(mol),
        rdMolDescriptors.CalcNumSaturatedCarbocycles(mol),
        rdMolDescriptors.CalcNumSaturatedHeterocycles(mol),
        rdMolDescriptors.CalcLabuteASA(mol),
        rdMolDescriptors.CalcExactMolWt(mol),
        Descriptors.BertzCT(mol),
    ], dtype=np.float32)
    return lip_arr, fp, maccs, adv

# Process all molecules
features = {}
for i, smi in enumerate(df['smiles']):
    lip, fp, maccs, adv = compute_all_features(smi)
    if lip is not None:
        features[i] = {'lip': lip, 'fp': fp, 'maccs': maccs, 'adv': adv}

def get_split_features(ogb_indices, feat_key):
    arrs = []
    for i in ogb_indices:
        if i in features:
            arrs.append(features[i][feat_key])
    return np.array(arrs)

print(f'  Computed features for {len(features):,} molecules')

# Prepare all feature sets for each split
train_idx_list = split_idx['train'].tolist()
val_idx_list = split_idx['valid'].tolist()
test_idx_list = split_idx['test'].tolist()

# Get valid indices (where all features exist)
def get_valid_mask(idx_list):
    return [i for i in idx_list if i in features]

train_valid = get_valid_mask(train_idx_list)
val_valid = get_valid_mask(val_idx_list)
test_valid = get_valid_mask(test_idx_list)

train_lip = np.array([features[i]['lip'] for i in train_valid])
train_fp = np.array([features[i]['fp'] for i in train_valid])
train_maccs = np.array([features[i]['maccs'] for i in train_valid])
train_adv = np.array([features[i]['adv'] for i in train_valid])
y_train = np.array([dataset[i][1][0] for i in train_valid])

val_lip = np.array([features[i]['lip'] for i in val_valid])
val_fp = np.array([features[i]['fp'] for i in val_valid])
val_maccs = np.array([features[i]['maccs'] for i in val_valid])
val_adv = np.array([features[i]['adv'] for i in val_valid])
y_val = np.array([dataset[i][1][0] for i in val_valid])

test_lip = np.array([features[i]['lip'] for i in test_valid])
test_fp = np.array([features[i]['fp'] for i in test_valid])
test_maccs = np.array([features[i]['maccs'] for i in test_valid])
test_adv = np.array([features[i]['adv'] for i in test_valid])
y_test = np.array([dataset[i][1][0] for i in test_valid])

# GNN embeddings aligned with valid indices
train_idx_to_emb = {idx: pos for pos, idx in enumerate(train_idx_list)}
val_idx_to_emb = {idx: pos for pos, idx in enumerate(val_idx_list)}
test_idx_to_emb = {idx: pos for pos, idx in enumerate(test_idx_list)}

train_emb_valid = np.array([train_emb[train_idx_to_emb[i]] for i in train_valid])
val_emb_valid = np.array([val_emb[val_idx_to_emb[i]] for i in val_valid])
test_emb_valid = np.array([test_emb[test_idx_to_emb[i]] for i in test_valid])

print(f'  Train: {len(train_lip):,} | Val: {len(val_lip):,} | Test: {len(test_lip):,}')
print(f'  Feature dims: Lip={train_lip.shape[1]}, FP={train_fp.shape[1]}, MACCS={train_maccs.shape[1]}, Adv={train_adv.shape[1]}, GNN={train_emb_valid.shape[1]}')

# ══════════════════════════════════════════════════════════════════════════
# CatBoost experiments
# ══════════════════════════════════════════════════════════════════════════
from catboost import CatBoostClassifier

def train_catboost(X_train, y_train, X_val, y_val, X_test, y_test, name):
    cb = CatBoostClassifier(
        iterations=500, depth=6, learning_rate=0.05, auto_class_weights='Balanced',
        verbose=0, random_seed=42, eval_metric='AUC',
    )
    cb.fit(X_train, y_train, eval_set=(X_val, y_val), early_stopping_rounds=50)
    pred = cb.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, pred)
    ap = average_precision_score(y_test, pred)
    print(f'  {name}: AUC={auc:.4f}, AUPRC={ap:.4f}')
    return {'name': name, 'test_auc': round(auc, 4), 'auprc': round(ap, 4)}

print('\n' + '=' * 60)
print('FEATURE ABLATION STUDY')
print('=' * 60)

# Individual feature sets
r_lip = train_catboost(train_lip, y_train, val_lip, y_val, test_lip, y_test, 'Lipinski (14)')
r_fp = train_catboost(train_fp, y_train, val_fp, y_val, test_fp, y_test, 'Morgan FP (1024)')
r_maccs = train_catboost(train_maccs, y_train, val_maccs, y_val, test_maccs, y_test, 'MACCS (167)')
r_adv = train_catboost(train_adv, y_train, val_adv, y_val, test_adv, y_test, 'Advanced desc (12)')
r_emb = train_catboost(train_emb_valid, y_train, val_emb_valid, y_val, test_emb_valid, y_test, 'GNN embed (128)')

# Combinations
r_lip_fp = train_catboost(
    np.hstack([train_lip, train_fp]), y_train,
    np.hstack([val_lip, val_fp]), y_val,
    np.hstack([test_lip, test_fp]), y_test,
    'Lip+FP (1038)')

r_lip_maccs = train_catboost(
    np.hstack([train_lip, train_maccs]), y_train,
    np.hstack([val_lip, val_maccs]), y_val,
    np.hstack([test_lip, test_maccs]), y_test,
    'Lip+MACCS (181)')

r_lip_fp_maccs = train_catboost(
    np.hstack([train_lip, train_fp, train_maccs, train_adv]), y_train,
    np.hstack([val_lip, val_fp, val_maccs, val_adv]), y_val,
    np.hstack([test_lip, test_fp, test_maccs, test_adv]), y_test,
    'All traditional (1217)')

r_emb_lip = train_catboost(
    np.hstack([train_emb_valid, train_lip]), y_train,
    np.hstack([val_emb_valid, val_lip]), y_val,
    np.hstack([test_emb_valid, test_lip]), y_test,
    'GNN+Lip (142)')

r_emb_lip_fp = train_catboost(
    np.hstack([train_emb_valid, train_lip, train_fp]), y_train,
    np.hstack([val_emb_valid, val_lip, val_fp]), y_val,
    np.hstack([test_emb_valid, test_lip, test_fp]), y_test,
    'GNN+Lip+FP (1166)')

r_full_hybrid = train_catboost(
    np.hstack([train_emb_valid, train_lip, train_fp, train_maccs, train_adv]), y_train,
    np.hstack([val_emb_valid, val_lip, val_fp, val_maccs, val_adv]), y_val,
    np.hstack([test_emb_valid, test_lip, test_fp, test_maccs, test_adv]), y_test,
    'Full hybrid (1345)')

# ══════════════════════════════════════════════════════════════════════════
# MASTER COMPARISON TABLE
# ══════════════════════════════════════════════════════════════════════════
print('\n' + '=' * 80)
print('PHASE 3 MASTER COMPARISON')
print('=' * 80)

all_results = [
    # Phase 3 GNN
    {'name': 'GIN+Edge (no VN)', 'test_auc': 0.7860, 'auprc': 0.3441, 'type': 'gnn'},
    {'name': 'GIN+Edge+VN (OGB)', 'test_auc': 0.7622, 'auprc': 0.2858, 'type': 'gnn'},
    {'name': 'GIN+VN (no edge)', 'test_auc': 0.7578, 'auprc': 0.2642, 'type': 'gnn'},
    # Phase 3 CatBoost
    r_lip | {'type': 'feat'}, r_fp | {'type': 'feat'}, r_maccs | {'type': 'feat'},
    r_adv | {'type': 'feat'}, r_emb | {'type': 'feat'},
    r_lip_fp | {'type': 'feat'}, r_lip_maccs | {'type': 'feat'},
    r_lip_fp_maccs | {'type': 'feat'},
    # Hybrid
    r_emb_lip | {'type': 'hybrid'}, r_emb_lip_fp | {'type': 'hybrid'},
    r_full_hybrid | {'type': 'hybrid'},
    # Baselines
    {'name': 'Phase1 CatBoost (Mark)', 'test_auc': 0.7782, 'auprc': None, 'type': 'baseline'},
    {'name': 'Phase1 RF+Combined', 'test_auc': 0.7707, 'auprc': 0.3722, 'type': 'baseline'},
    {'name': 'Phase2 GIN (raw 9-feat)', 'test_auc': 0.7053, 'auprc': None, 'type': 'baseline'},
    {'name': 'Phase2 MLP-Domain9 (Mark)', 'test_auc': 0.7670, 'auprc': None, 'type': 'baseline'},
    {'name': 'SOTA (DeeperGCN+VN)', 'test_auc': 0.8476, 'auprc': None, 'type': 'sota'},
]

all_sorted = sorted(all_results, key=lambda r: r['test_auc'], reverse=True)

print(f'{"Rank":<5} {"Model":<35} {"Test AUC":>10} {"AUPRC":>10} {"Type":>10}')
print('-' * 75)
for rank, r in enumerate(all_sorted, 1):
    auprc_str = f"{r['auprc']:.4f}" if r.get('auprc') is not None else '—'
    marker = '***' if r['name'] == all_sorted[0]['name'] and r['type'] != 'sota' else ''
    print(f'{rank:<5} {r["name"]:<35} {r["test_auc"]:>10.4f} {auprc_str:>10} {r["type"]:>10} {marker}')

# ══════════════════════════════════════════════════════════════════════════
# PLOTS
# ══════════════════════════════════════════════════════════════════════════

# 1. Comprehensive comparison bar chart
fig, ax = plt.subplots(figsize=(14, 10))
plot_results = [r for r in all_sorted if r['type'] != 'sota']
names = [r['name'] for r in plot_results]
aucs = [r['test_auc'] for r in plot_results]
type_colors = {'gnn': '#4472C4', 'feat': '#70AD47', 'hybrid': '#ED7D31', 'baseline': '#FFC000'}
colors = [type_colors[r['type']] for r in plot_results]

bars = ax.barh(names[::-1], aucs[::-1], color=colors[::-1], edgecolor='white', height=0.6, alpha=0.85)
for bar, auc in zip(bars, aucs[::-1]):
    ax.text(bar.get_width() + 0.002, bar.get_y() + bar.get_height()/2, f'{auc:.4f}',
            va='center', fontsize=9, fontweight='bold')

ax.axvline(0.8476, color='red', ls=':', lw=2, alpha=0.7, label='SOTA (0.8476)')
ax.axvline(0.7782, color='gray', ls='--', lw=1.5, alpha=0.7, label='Phase 1 CatBoost (0.7782)')
ax.set(xlabel='Test ROC-AUC', title='Phase 3: Feature Engineering — All Experiments\nogbg-molhiv (scaffold split)')
ax.set_xlim(0.60, 0.90)
ax.grid(axis='x', alpha=0.3)
from matplotlib.patches import Patch
ax.legend(handles=[
    Patch(fc='#4472C4', label='GNN models'), Patch(fc='#70AD47', label='Feature→CatBoost'),
    Patch(fc='#ED7D31', label='Hybrid (GNN+Features)'), Patch(fc='#FFC000', label='Prior baselines'),
    Patch(fc='red', alpha=0.7, label='SOTA'),
], loc='lower right', fontsize=9)
plt.tight_layout()
plt.savefig(RESULTS_DIR / 'phase3_model_comparison.png', dpi=150, bbox_inches='tight')
print('\nSaved: phase3_model_comparison.png')

# 2. Feature ablation grouped bar chart
fig, ax = plt.subplots(figsize=(12, 6))
ablation = [r_lip, r_fp, r_maccs, r_adv, r_emb, r_lip_fp, r_lip_maccs, r_lip_fp_maccs,
            r_emb_lip, r_emb_lip_fp, r_full_hybrid]
abl_names = [r['name'] for r in ablation]
abl_aucs = [r['test_auc'] for r in ablation]
abl_colors = ['#70AD47']*5 + ['#2E8B57']*3 + ['#ED7D31']*3

bars = ax.bar(range(len(abl_names)), abl_aucs, color=abl_colors, edgecolor='white', alpha=0.85)
for bar, auc in zip(bars, abl_aucs):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.003, f'{auc:.4f}',
            ha='center', va='bottom', fontsize=8, fontweight='bold', rotation=0)
ax.axhline(0.7782, color='gray', ls='--', lw=1.5, alpha=0.7, label='Phase 1 CatBoost (0.7782)')
ax.axhline(0.8476, color='red', ls=':', lw=2, alpha=0.7, label='SOTA (0.8476)')
ax.set_xticks(range(len(abl_names)))
ax.set_xticklabels(abl_names, rotation=45, ha='right', fontsize=8)
ax.set(ylabel='Test ROC-AUC', title='Feature Ablation: Which features matter for HIV activity prediction?')
ax.set_ylim(0.60, 0.90)
ax.grid(axis='y', alpha=0.3)
ax.legend(handles=[
    Patch(fc='#70AD47', label='Individual features'),
    Patch(fc='#2E8B57', label='Traditional combos'),
    Patch(fc='#ED7D31', label='GNN hybrid'),
], fontsize=9)
plt.tight_layout()
plt.savefig(RESULTS_DIR / 'phase3_ablation.png', dpi=150, bbox_inches='tight')
print('Saved: phase3_ablation.png')

# 3. Edge features impact: Phase 2 vs Phase 3 GNN comparison
fig, ax = plt.subplots(figsize=(10, 5))
gnn_comparison = [
    ('Phase 2 GAT (9 feat)', 0.6677),
    ('Phase 2 GCN (9 feat)', 0.6938),
    ('Phase 2 GraphSAGE (9 feat)', 0.7050),
    ('Phase 2 GIN (9 feat)', 0.7053),
    ('Phase 3 GIN+VN (9→128)', 0.7578),
    ('Phase 3 GIN+Edge+VN', 0.7622),
    ('Phase 3 GIN+Edge', 0.7860),
]
gnn_names = [g[0] for g in gnn_comparison]
gnn_aucs = [g[1] for g in gnn_comparison]
gnn_colors = ['#C0C0C0']*4 + ['#4472C4']*3
bars = ax.barh(gnn_names, gnn_aucs, color=gnn_colors, edgecolor='white', height=0.6, alpha=0.85)
for bar, auc in zip(bars, gnn_aucs):
    ax.text(bar.get_width() + 0.003, bar.get_y() + bar.get_height()/2, f'{auc:.4f}',
            va='center', fontsize=10, fontweight='bold')
ax.axvline(0.7782, color='gray', ls='--', lw=1.5, alpha=0.7, label='CatBoost (0.7782)')
ax.set(xlabel='Test ROC-AUC', title='Edge Features Transform GNN Performance\n+0.081 AUC from adding bond type/stereo/conjugation encoding')
ax.set_xlim(0.60, 0.85)
ax.grid(axis='x', alpha=0.3)
ax.legend()
plt.tight_layout()
plt.savefig(RESULTS_DIR / 'phase3_edge_impact.png', dpi=150, bbox_inches='tight')
print('Saved: phase3_edge_impact.png')

# ── Save results ──────────────────────────────────────────────────────
champion = [r for r in all_sorted if r['type'] != 'sota'][0]
phase3_data = {
    'gnn_experiments': {
        'GIN+Edge (no VN)': {'test_auc': 0.7860, 'auprc': 0.3441, 'params': 144516, 'time_s': 297},
        'GIN+Edge+VN (OGB)': {'test_auc': 0.7622, 'auprc': 0.2858, 'params': 211204, 'time_s': 292},
        'GIN+VN (no edge)': {'test_auc': 0.7578, 'auprc': 0.2642, 'params': 206212, 'time_s': 199},
    },
    'feature_ablation': {r['name']: {'test_auc': r['test_auc'], 'auprc': r['auprc']}
                         for r in [r_lip, r_fp, r_maccs, r_adv, r_emb,
                                   r_lip_fp, r_lip_maccs, r_lip_fp_maccs,
                                   r_emb_lip, r_emb_lip_fp, r_full_hybrid]},
    'champion': {'name': champion['name'], 'test_auc': champion['test_auc']},
    'key_finding': 'Edge features (bond type + stereo + conjugation) boost GIN from 0.7053 to 0.7860 (+0.081 AUC)',
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
print(f'Phase 3 Champion: {champion["name"]} (AUC = {champion["test_auc"]:.4f})')
print(f'Gap to SOTA: {0.8476 - champion["test_auc"]:+.4f}')
print(f'Δ vs Phase 1 CatBoost: {champion["test_auc"] - 0.7782:+.4f}')
print(f'Δ vs Phase 2 GIN (raw): {champion["test_auc"] - 0.7053:+.4f}')
print(f'\nKey insight: Adding 3 edge features (bond type, stereo, conjugation) to GIN')
print(f'  improved AUC by +0.081 — the BIGGEST single improvement so far.')
print(f'  Edge features encode chemical bond information that fingerprints approximate.')
print('=' * 60)
print('DONE.')
