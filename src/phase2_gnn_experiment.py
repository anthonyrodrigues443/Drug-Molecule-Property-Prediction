"""
Phase 2: GNN Architecture Comparison — Drug-Molecule-Property-Prediction
Date: 2026-04-07
Researcher: Anthony Rodrigues
"""
import os, json, time, warnings, sys
warnings.filterwarnings('ignore')
os.environ['MPLBACKEND'] = 'Agg'  # non-interactive backend

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.datasets import MoleculeNet
from torch_geometric.loader import DataLoader as PyGDataLoader
from torch_geometric.nn import (
    GCNConv, GINConv, GATConv, SAGEConv,
    global_mean_pool, global_add_pool
)
from sklearn.metrics import roc_auc_score
from pathlib import Path

# ── Config ──────────────────────────────────────────────────────────────
BASE_DIR = Path('/Users/anthonyrodrigues/Desktop/YC-Portfolio-Projects/Drug-Molecule-Property-Prediction')
DATA_DIR = BASE_DIR / 'data' / 'raw'
RESULTS_DIR = BASE_DIR / 'results'
RESULTS_DIR.mkdir(exist_ok=True)

DEVICE = torch.device('cpu')
BATCH_SIZE = 128
HIDDEN_DIM = 128
NUM_LAYERS = 3
DROPOUT = 0.5
LR = 1e-3
EPOCHS = 50
PATIENCE = 12

torch.manual_seed(42)
np.random.seed(42)

# ── Data Loading ────────────────────────────────────────────────────────
print('Loading ogbg-molhiv via PyG MoleculeNet...')
dataset = MoleculeNet(root=str(DATA_DIR / 'pyg'), name='HIV')
mol_df = pd.read_csv(DATA_DIR / 'ogbg_molhiv' / 'mapping' / 'mol.csv.gz')
pyg_smiles_to_idx = {dataset[i].smiles: i for i in range(len(dataset))}

def map_ogb_to_pyg(csv_path):
    ogb = pd.read_csv(csv_path, header=None)[0].tolist()
    return [pyg_smiles_to_idx[mol_df.iloc[i]['smiles']] for i in ogb if mol_df.iloc[i]['smiles'] in pyg_smiles_to_idx]

train_idx = map_ogb_to_pyg(DATA_DIR / 'ogbg_molhiv' / 'split' / 'scaffold' / 'train.csv.gz')
val_idx   = map_ogb_to_pyg(DATA_DIR / 'ogbg_molhiv' / 'split' / 'scaffold' / 'valid.csv.gz')
test_idx  = map_ogb_to_pyg(DATA_DIR / 'ogbg_molhiv' / 'split' / 'scaffold' / 'test.csv.gz')

train_loader = PyGDataLoader([dataset[i] for i in train_idx], batch_size=BATCH_SIZE, shuffle=True)
val_loader   = PyGDataLoader([dataset[i] for i in val_idx],   batch_size=BATCH_SIZE)
test_loader  = PyGDataLoader([dataset[i] for i in test_idx],  batch_size=BATCH_SIZE)

IN_DIM = dataset.num_features
print(f'Data: Train {len(train_idx):,} | Val {len(val_idx):,} | Test {len(test_idx):,} | Features {IN_DIM}')

# ── Model Definitions ───────────────────────────────────────────────────
class GCN(nn.Module):
    def __init__(self, in_dim, hidden_dim, num_layers, dropout, **kw):
        super().__init__()
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        self.convs.append(GCNConv(in_dim, hidden_dim))
        self.bns.append(nn.BatchNorm1d(hidden_dim))
        for _ in range(num_layers - 1):
            self.convs.append(GCNConv(hidden_dim, hidden_dim))
            self.bns.append(nn.BatchNorm1d(hidden_dim))
        self.lin1 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.lin2 = nn.Linear(hidden_dim // 2, 1)
        self.dropout = dropout

    def forward(self, x, edge_index, batch):
        for conv, bn in zip(self.convs, self.bns):
            x = F.relu(bn(conv(x, edge_index)))
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = global_mean_pool(x, batch)
        x = F.dropout(F.relu(self.lin1(x)), p=self.dropout, training=self.training)
        return self.lin2(x).squeeze(-1)


class GIN(nn.Module):
    def __init__(self, in_dim, hidden_dim, num_layers, dropout, **kw):
        super().__init__()
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        mlp0 = nn.Sequential(nn.Linear(in_dim, hidden_dim), nn.BatchNorm1d(hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, hidden_dim))
        self.convs.append(GINConv(mlp0, train_eps=True))
        self.bns.append(nn.BatchNorm1d(hidden_dim))
        for _ in range(num_layers - 1):
            mlp = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.BatchNorm1d(hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, hidden_dim))
            self.convs.append(GINConv(mlp, train_eps=True))
            self.bns.append(nn.BatchNorm1d(hidden_dim))
        self.lin1 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.lin2 = nn.Linear(hidden_dim // 2, 1)
        self.dropout = dropout

    def forward(self, x, edge_index, batch):
        for conv, bn in zip(self.convs, self.bns):
            x = F.relu(bn(conv(x, edge_index)))
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = global_mean_pool(x, batch)
        x = F.dropout(F.relu(self.lin1(x)), p=self.dropout, training=self.training)
        return self.lin2(x).squeeze(-1)


class GAT(nn.Module):
    def __init__(self, in_dim, hidden_dim, num_layers, dropout, heads=4, **kw):
        super().__init__()
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        self.convs.append(GATConv(in_dim, hidden_dim // heads, heads=heads, dropout=dropout, concat=True))
        self.bns.append(nn.BatchNorm1d(hidden_dim))
        for _ in range(num_layers - 1):
            self.convs.append(GATConv(hidden_dim, hidden_dim // heads, heads=heads, dropout=dropout, concat=True))
            self.bns.append(nn.BatchNorm1d(hidden_dim))
        self.lin1 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.lin2 = nn.Linear(hidden_dim // 2, 1)
        self.dropout = dropout

    def forward(self, x, edge_index, batch):
        for conv, bn in zip(self.convs, self.bns):
            x = F.elu(bn(conv(x, edge_index)))
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = global_mean_pool(x, batch)
        x = F.dropout(F.relu(self.lin1(x)), p=self.dropout, training=self.training)
        return self.lin2(x).squeeze(-1)


class GraphSAGE(nn.Module):
    def __init__(self, in_dim, hidden_dim, num_layers, dropout, **kw):
        super().__init__()
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        self.convs.append(SAGEConv(in_dim, hidden_dim))
        self.bns.append(nn.BatchNorm1d(hidden_dim))
        for _ in range(num_layers - 1):
            self.convs.append(SAGEConv(hidden_dim, hidden_dim))
            self.bns.append(nn.BatchNorm1d(hidden_dim))
        self.lin1 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.lin2 = nn.Linear(hidden_dim // 2, 1)
        self.dropout = dropout

    def forward(self, x, edge_index, batch):
        for conv, bn in zip(self.convs, self.bns):
            x = F.relu(bn(conv(x, edge_index)))
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = global_mean_pool(x, batch)
        x = F.dropout(F.relu(self.lin1(x)), p=self.dropout, training=self.training)
        return self.lin2(x).squeeze(-1)


def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

# ── Training Functions ──────────────────────────────────────────────────
def train_epoch(model, loader, optimizer, device):
    model.train()
    total_loss = 0
    n = 0
    pos_weight = torch.tensor([10.0], device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    for batch in loader:
        batch = batch.to(device)
        optimizer.zero_grad()
        logits = model(batch.x.float(), batch.edge_index, batch.batch)
        labels = batch.y.float().view(-1)
        mask = ~torch.isnan(labels)
        loss = criterion(logits[mask], labels[mask])
        loss.backward()
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
        logits = model(batch.x.float(), batch.edge_index, batch.batch)
        labels = batch.y.float().view(-1)
        mask = ~torch.isnan(labels)
        preds_all.append(torch.sigmoid(logits[mask]).cpu().numpy())
        labels_all.append(labels[mask].cpu().numpy())
    preds = np.concatenate(preds_all)
    labels = np.concatenate(labels_all)
    if labels.sum() == 0 or labels.sum() == len(labels):
        return 0.5
    return roc_auc_score(labels, preds)


def train_model(model_cls, model_kwargs, name):
    model = model_cls(IN_DIM, HIDDEN_DIM, NUM_LAYERS, DROPOUT, **model_kwargs).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=7, min_lr=1e-5)

    best_val, best_test = 0, 0
    patience_count = 0
    history = {'train_loss': [], 'val_auc': [], 'test_auc': []}
    best_state = None

    print('\n' + '=' * 60)
    print('Training {} | {:,} params'.format(name, count_params(model)))
    print('=' * 60)
    t0 = time.time()

    for epoch in range(1, EPOCHS + 1):
        loss = train_epoch(model, train_loader, optimizer, DEVICE)
        val_auc = evaluate(model, val_loader, DEVICE)
        test_auc = evaluate(model, test_loader, DEVICE)
        history['train_loss'].append(loss)
        history['val_auc'].append(val_auc)
        history['test_auc'].append(test_auc)
        scheduler.step(val_auc)

        if val_auc > best_val:
            best_val = val_auc
            best_test = test_auc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_count = 0
        else:
            patience_count += 1

        if epoch % 10 == 0:
            print('  Ep {:3d} | loss={:.4f} | val={:.4f} | test={:.4f} | [{:.0f}s]'.format(
                epoch, loss, val_auc, test_auc, time.time() - t0))

        if patience_count >= PATIENCE:
            print('  Early stopping at epoch {}'.format(epoch))
            break

    elapsed = time.time() - t0
    print('  RESULT: val={:.4f} test={:.4f} ({:.0f}s, {} epochs)'.format(best_val, best_test, elapsed, len(history['val_auc'])))

    return {
        'name': name, 'best_val_auc': best_val, 'best_test_auc': best_test,
        'params': count_params(model), 'epochs_run': len(history['val_auc']),
        'train_time_s': elapsed, 'history': history, 'best_state': best_state,
        'model_cls': model_cls, 'model_kwargs': model_kwargs,
    }


# ── Run All Models ──────────────────────────────────────────────────────
print('\n' + '#' * 60)
print('# Phase 2: GNN Architecture Comparison')
print('#' * 60)

result_gcn  = train_model(GCN,       {},            'GCN')
result_gin  = train_model(GIN,       {},            'GIN')
result_gat  = train_model(GAT,       {'heads': 4},  'GAT')
result_sage = train_model(GraphSAGE, {},            'GraphSAGE')

all_results = [result_gcn, result_gin, result_gat, result_sage]
results_sorted = sorted(all_results, key=lambda r: r['best_test_auc'], reverse=True)

# ── Print Comparison Table ──────────────────────────────────────────────
print('\n' + '=' * 80)
print('HEAD-TO-HEAD: Phase 2 GNNs vs Phase 1 Traditional ML')
print('=' * 80)
print('{:<5} {:<25} {:>10} {:>10} {:>10} {:>8}'.format('Rank', 'Model', 'Test AUC', 'Val AUC', 'Params', 'Time'))
print('-' * 80)
for rank, r in enumerate(results_sorted, 1):
    print('{:<5} {:<25} {:>10.4f} {:>10.4f} {:>10,} {:>6.0f}s'.format(
        rank, r['name'], r['best_test_auc'], r['best_val_auc'], r['params'], r['train_time_s']))
print()
baselines = [('CatBoost (Mark)', 0.7782), ('RF+Combined', 0.7707), ('XGBoost+Combined', 0.7613)]
print('--- Phase 1 Traditional ML ---')
for name, auc in baselines:
    print('{:<5} {:<25} {:>10.4f}'.format('-', name, auc))
print('{:<5} {:<25} {:>10.4f}'.format('-', 'DeeperGCN (SOTA)', 0.8476))

winner = results_sorted[0]
print('\nPhase 2 Champion: {} (Test AUC = {:.4f})'.format(winner['name'], winner['best_test_auc']))
print('Gap to SOTA: {:+.4f} | Gap to Phase 1 CatBoost: {:+.4f}'.format(
    0.8476 - winner['best_test_auc'], winner['best_test_auc'] - 0.7782))

# ── Plots ───────────────────────────────────────────────────────────────
colors = ['#4472C4', '#ED7D31', '#70AD47', '#FFC000']

# Training curves
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
for r, color in zip(all_results, colors):
    axes[0].plot(r['history']['val_auc'], label='{} ({:.4f})'.format(r['name'], r['best_val_auc']), color=color, lw=2)
    axes[1].plot(r['history']['train_loss'], label=r['name'], color=color, lw=2)
axes[0].axhline(0.7782, color='gray', ls='--', lw=1.5, alpha=0.7, label='Phase 1 CatBoost (0.7782)')
axes[0].axhline(0.8476, color='red', ls=':', lw=2, alpha=0.7, label='SOTA DeeperGCN (0.8476)')
axes[0].set(xlabel='Epoch', ylabel='Val ROC-AUC', title='Validation AUC — All GNN Architectures')
axes[0].legend(fontsize=8); axes[0].set_ylim(0.5, 0.90); axes[0].grid(alpha=0.3)
axes[1].set(xlabel='Epoch', ylabel='Training Loss', title='Training Loss Curves')
axes[1].legend(fontsize=9); axes[1].grid(alpha=0.3)
plt.tight_layout()
plt.savefig(RESULTS_DIR / 'phase2_training_curves.png', dpi=150, bbox_inches='tight')
print('\nSaved: phase2_training_curves.png')

# Bar chart
fig, ax = plt.subplots(figsize=(13, 6))
all_names = [r['name'] for r in results_sorted] + [b[0] for b in baselines] + ['DeeperGCN (SOTA)']
all_aucs = [r['best_test_auc'] for r in results_sorted] + [b[1] for b in baselines] + [0.8476]
all_colors = ['#4472C4'] * len(results_sorted) + ['#70AD47'] * len(baselines) + ['#FF4444']
bars = ax.barh(all_names, all_aucs, color=all_colors, edgecolor='white', height=0.6, alpha=0.85)
for bar, auc in zip(bars, all_aucs):
    ax.text(bar.get_width() + 0.001, bar.get_y() + bar.get_height() / 2, '{:.4f}'.format(auc), va='center', fontsize=11, fontweight='bold')
ax.axvline(0.7782, color='#70AD47', ls='--', lw=1.5, alpha=0.7)
ax.set(xlabel='Test ROC-AUC', title='Phase 2: GNN Architectures vs Traditional ML\nogbg-molhiv (scaffold split)')
ax.set_xlim(0.65, 0.92); ax.grid(axis='x', alpha=0.3)
from matplotlib.patches import Patch
ax.legend(handles=[Patch(fc='#4472C4', label='Phase 2 GNNs'), Patch(fc='#70AD47', label='Phase 1 Trad ML'), Patch(fc='#FF4444', label='SOTA')], loc='lower right')
plt.tight_layout()
plt.savefig(RESULTS_DIR / 'phase2_model_comparison.png', dpi=150, bbox_inches='tight')
print('Saved: phase2_model_comparison.png')

# ── Error analysis ──────────────────────────────────────────────────────
print('\nPer-model test set metrics (threshold=0.5):')
print('{:<15} {:>5} {:>5} {:>5} {:>10} {:>10} {:>8}'.format('Model', 'TP', 'FP', 'FN', 'Sens', 'Spec', 'AUC'))
print('-' * 65)
for r in results_sorted:
    m = r['model_cls'](IN_DIM, HIDDEN_DIM, NUM_LAYERS, DROPOUT, **r['model_kwargs'])
    m.load_state_dict(r['best_state'])
    m.eval()
    preds, labs = [], []
    with torch.no_grad():
        for batch in test_loader:
            logits = m(batch.x.float(), batch.edge_index, batch.batch)
            labels = batch.y.float().view(-1)
            mask = ~torch.isnan(labels)
            preds.append(torch.sigmoid(logits[mask]).numpy())
            labs.append(labels[mask].numpy())
    p, l = np.concatenate(preds), np.concatenate(labs)
    pb = (p >= 0.5).astype(int)
    tp = int(((pb == 1) & (l == 1)).sum())
    fp = int(((pb == 1) & (l == 0)).sum())
    fn = int(((pb == 0) & (l == 1)).sum())
    tn = int(((pb == 0) & (l == 0)).sum())
    print('{:<15} {:>5} {:>5} {:>5} {:>10.3f} {:>10.3f} {:>8.4f}'.format(
        r['name'], tp, fp, fn, tp/(tp+fn+1e-8), tn/(tn+fp+1e-8), r['best_test_auc']))

# ── Save Results ────────────────────────────────────────────────────────
phase2_data = {}
for r in all_results:
    phase2_data[r['name']] = {
        'val_auc': round(r['best_val_auc'], 4),
        'test_auc': round(r['best_test_auc'], 4),
        'params': r['params'],
        'epochs_run': r['epochs_run'],
        'train_time_s': round(r['train_time_s'], 1),
    }

# Update metrics.json
metrics_path = RESULTS_DIR / 'metrics.json'
if metrics_path.exists():
    with open(metrics_path) as f:
        metrics = json.load(f)
    if isinstance(metrics, list):
        metrics = {'phase1_experiments': metrics}
else:
    metrics = {}
metrics['phase2_gnn_comparison'] = phase2_data
metrics['phase2_champion'] = {
    'model': results_sorted[0]['name'],
    'test_auc': results_sorted[0]['best_test_auc'],
    'val_auc': results_sorted[0]['best_val_auc'],
}
with open(metrics_path, 'w') as f:
    json.dump(metrics, f, indent=2)

with open(RESULTS_DIR / 'phase2_gnn_results.json', 'w') as f:
    json.dump(phase2_data, f, indent=2)

print('\nAll results saved to results/')
print('DONE.')
