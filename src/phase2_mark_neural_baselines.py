"""
Phase 2 Mark - Neural Baselines on Molecular Fingerprints (ultra-lightweight).
Date: 2026-04-07
Researcher: Mark Rodrigues

Complementary angle to Anthony's Phase 2 GNN comparison:
  Anthony ran 4 GNN architectures (GCN, GIN, GAT, GraphSAGE); all lost to CatBoost
  (best = GIN 0.7053 vs CatBoost 0.7782, gap -0.073). His hypothesis: domain
  expertise beats raw graph convolution.

Mark's question:
  Is this failure GNN-specific, or do *any* neural models lose to tree-based
  learners on molecular fingerprints? If PyTorch MLPs on the exact same
  1033-feature matrix CatBoost used also underperform, the bottleneck is not
  the GNN architecture - it is that dense neural networks struggle with sparse
  binary fingerprints while trees handle them natively.

Why so lightweight:
  Earlier attempts hit MemoryError importing scipy/sklearn on this machine.
  This script uses only numpy, pandas, and torch (cpu). ROC-AUC is computed
  directly from tensor ranks. No sklearn, no scipy, no torch_geometric.
"""
import os, sys, json, time, warnings, gc
warnings.filterwarnings('ignore')
os.environ['MPLBACKEND'] = 'Agg'

import numpy as np
import pandas as pd
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

PROJ     = Path(__file__).resolve().parent.parent
DATA_DIR = PROJ / 'data'
RESULTS  = PROJ / 'results'
RESULTS.mkdir(exist_ok=True)

DEVICE = torch.device('cpu')
torch.manual_seed(42)
np.random.seed(42)

# Anthony + Phase 1 baselines
CATBOOST_P1    = 0.7782   # Mark Phase 1 champion (CatBoost Combined, auto_weight)
ANTHONY_RF_P1  = 0.7707   # Anthony Phase 1 champion
ANTHONY_GIN_P2 = 0.7053   # Anthony Phase 2 best (GIN)
ANTHONY_SAGE   = 0.7050
ANTHONY_GCN    = 0.6938
ANTHONY_GAT    = 0.6677
OGB_SOTA       = 0.8476


def log(m):
    print(m, flush=True)


# ---- Hand-rolled ROC-AUC (no sklearn) ----
def roc_auc_np(y_true, y_score):
    """Mann-Whitney U formulation - O(n log n)."""
    y_true = np.asarray(y_true).ravel()
    y_score = np.asarray(y_score).ravel()
    pos = y_true == 1
    neg = ~pos
    n_pos = pos.sum()
    n_neg = neg.sum()
    if n_pos == 0 or n_neg == 0:
        return 0.5
    order = np.argsort(y_score, kind='mergesort')
    ranks = np.empty(len(y_score), dtype=np.float64)
    ranks[order] = np.arange(1, len(y_score) + 1)
    # Handle ties by averaging
    _, inv, counts = np.unique(y_score, return_inverse=True, return_counts=True)
    tie_sum = np.cumsum(counts)
    tie_start = np.concatenate([[0], tie_sum[:-1]])
    tie_avg_rank = (tie_start + tie_sum + 1) / 2.0
    ranks = tie_avg_rank[inv]
    sum_ranks_pos = ranks[pos].sum()
    auc = (sum_ranks_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)
    return float(auc)


def f1_np(y_true, y_pred):
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()
    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())
    if tp + fp == 0 or tp + fn == 0:
        return 0.0
    prec = tp / (tp + fp)
    rec  = tp / (tp + fn)
    if prec + rec == 0:
        return 0.0
    return 2 * prec * rec / (prec + rec)


log('=' * 72)
log('Phase 2 Mark - Neural Baselines on Molecular Fingerprints')
log('=' * 72)
log('PyTorch: {}'.format(torch.__version__))

# ---- Load parquet ----
t0 = time.time()
feat_df = pd.read_parquet(DATA_DIR / 'processed' / 'ogbg_molhiv_features.parquet')
log('[{:.0f}s] Loaded parquet: {}'.format(time.time() - t0, feat_df.shape))

# Available domain columns from the parquet
dom_cols = ['n_atoms', 'n_bonds', 'mol_weight', 'hbd', 'hba', 'rotatable_bonds',
            'aromatic_rings', 'ring_count', 'heavy_atom_count']
fp_cols  = [c for c in feat_df.columns if c.startswith('fp_')]
log('Domain features: {}  Morgan FP: {}  Combined: {}'.format(
    len(dom_cols), len(fp_cols), len(dom_cols) + len(fp_cols)))

# Train / val / test masks
tr_mask = feat_df['split'] == 'train'
va_mask = feat_df['split'] == 'val'
te_mask = feat_df['split'] == 'test'
log('Train: {:,}  Val: {:,}  Test: {:,}'.format(
    int(tr_mask.sum()), int(va_mask.sum()), int(te_mask.sum())))

pos_rate = float(feat_df.loc[tr_mask, 'hiv_active'].mean())
log('Positive rate (train): {:.4f}'.format(pos_rate))


def standardise(train_X, val_X, test_X):
    mean = train_X.mean(axis=0, keepdims=True)
    std  = train_X.std(axis=0, keepdims=True) + 1e-6
    return (train_X - mean) / std, (val_X - mean) / std, (test_X - mean) / std


def get_split(cols):
    Xtr = feat_df.loc[tr_mask, cols].values.astype(np.float32)
    Xva = feat_df.loc[va_mask, cols].values.astype(np.float32)
    Xte = feat_df.loc[te_mask, cols].values.astype(np.float32)
    Xtr, Xva, Xte = standardise(Xtr, Xva, Xte)
    ytr = feat_df.loc[tr_mask, 'hiv_active'].values.astype(np.float32)
    yva = feat_df.loc[va_mask, 'hiv_active'].values.astype(np.float32)
    yte = feat_df.loc[te_mask, 'hiv_active'].values.astype(np.float32)
    return Xtr, ytr, Xva, yva, Xte, yte


class MLP(nn.Module):
    def __init__(self, in_dim, hidden=128, depth=3, dropout=0.3):
        super().__init__()
        layers = []
        prev = in_dim
        h = hidden
        for _ in range(depth):
            layers.append(nn.Linear(prev, h))
            layers.append(nn.BatchNorm1d(h))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev = h
            h = max(h // 2, 32)
        layers.append(nn.Linear(prev, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


def train_mlp(Xtr, ytr, Xva, yva, Xte, yte, hidden, depth, dropout,
              lr=1e-3, n_epochs=30, batch=256, name='MLP'):
    in_dim = Xtr.shape[1]
    model = MLP(in_dim, hidden=hidden, depth=depth, dropout=dropout).to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters())

    pos_w = torch.tensor([(1 - pos_rate) / pos_rate], dtype=torch.float32)
    crit = nn.BCEWithLogitsLoss(pos_weight=pos_w)
    opt  = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)

    Xtr_t = torch.from_numpy(Xtr).float()
    ytr_t = torch.from_numpy(ytr).float().unsqueeze(1)
    Xva_t = torch.from_numpy(Xva).float()
    Xte_t = torch.from_numpy(Xte).float()
    loader = DataLoader(TensorDataset(Xtr_t, ytr_t), batch_size=batch, shuffle=True)

    best_val = best_test = 0.0
    best_ep = 0
    best_test_preds = None
    no_improve = 0
    hist = []
    t0 = time.time()

    for ep in range(1, n_epochs + 1):
        model.train()
        total_loss = 0.0
        nseen = 0
        for xb, yb in loader:
            opt.zero_grad()
            out = model(xb)
            loss = crit(out, yb)
            loss.backward()
            opt.step()
            total_loss += loss.item() * xb.size(0)
            nseen += xb.size(0)
        train_loss = total_loss / max(nseen, 1)

        model.eval()
        with torch.no_grad():
            va_pred = torch.sigmoid(model(Xva_t)).numpy().ravel()
            te_pred = torch.sigmoid(model(Xte_t)).numpy().ravel()
        va_auc = roc_auc_np(yva, va_pred)
        te_auc = roc_auc_np(yte, te_pred)
        hist.append({'epoch': ep, 'loss': train_loss, 'val': va_auc, 'test': te_auc})

        if va_auc > best_val:
            best_val = va_auc
            best_test = te_auc
            best_ep = ep
            best_test_preds = te_pred.copy()
            no_improve = 0
        else:
            no_improve += 1

        if ep % 5 == 0 or ep == n_epochs:
            log('  [{}] ep {:2d}  loss={:.4f}  val={:.4f}  best_val={:.4f}  best_test={:.4f}  ({:.0f}s)'.format(
                name, ep, train_loss, va_auc, best_val, best_test, time.time() - t0))
        if no_improve >= 8:
            log('  [{}] early stop at ep {}'.format(name, ep))
            break

    f1 = f1_np(yte, (best_test_preds >= 0.5).astype(int))
    return {
        'name': name,
        'val_auc': best_val,
        'test_auc': best_test,
        'best_epoch': best_ep,
        'params': int(n_params),
        'train_s': time.time() - t0,
        'test_f1_t0p5': f1,
        'hist': hist,
    }


results = []

log('\n[E1] MLP-Domain9  (neural on just 9 lightweight domain/graph features)')
log('-' * 60)
Xtr, ytr, Xva, yva, Xte, yte = get_split(dom_cols)
r1 = train_mlp(Xtr, ytr, Xva, yva, Xte, yte, hidden=64, depth=3, dropout=0.3,
               n_epochs=30, batch=256, name='MLP-Domain9')
results.append(r1)
del Xtr, ytr, Xva, yva, Xte, yte; gc.collect()

log('\n[E2] MLP-Morgan1024  (neural on Morgan FP only)')
log('-' * 60)
Xtr, ytr, Xva, yva, Xte, yte = get_split(fp_cols)
r2 = train_mlp(Xtr, ytr, Xva, yva, Xte, yte, hidden=256, depth=3, dropout=0.4,
               n_epochs=30, batch=256, name='MLP-Morgan1024')
results.append(r2)
del Xtr, ytr, Xva, yva, Xte, yte; gc.collect()

log('\n[E3] MLP-Combined1033  (neural on same Combined feature set as CatBoost)')
log('-' * 60)
Xtr, ytr, Xva, yva, Xte, yte = get_split(fp_cols + dom_cols)
r3 = train_mlp(Xtr, ytr, Xva, yva, Xte, yte, hidden=256, depth=3, dropout=0.4,
               n_epochs=30, batch=256, name='MLP-Combined1033')
results.append(r3)
del Xtr, ytr, Xva, yva, Xte, yte; gc.collect()

log('\n[E4] MLP-Wide-Combined  (double hidden, deeper - test capacity effect)')
log('-' * 60)
Xtr, ytr, Xva, yva, Xte, yte = get_split(fp_cols + dom_cols)
r4 = train_mlp(Xtr, ytr, Xva, yva, Xte, yte, hidden=512, depth=4, dropout=0.5,
               n_epochs=30, batch=256, name='MLP-Wide-Combined')
results.append(r4)
del Xtr, ytr, Xva, yva, Xte, yte; gc.collect()

# ---- Summary ----
log('\n' + '=' * 72)
log('PHASE 2 MARK SUMMARY - Neural (MLP) vs Tree (CatBoost) vs GNN (Anthony)')
log('=' * 72)
log('{:<22} {:<18} {:>10} {:>15} {:>12}'.format(
    'Model', 'Features', 'Test AUC', 'Delta vs CB', 'Time(s)'))
log('-' * 80)

def dlt(v):
    return '{:+.4f}'.format(v - CATBOOST_P1)

label_map = {
    'MLP-Domain9':        'Domain 9',
    'MLP-Morgan1024':     'Morgan FP 1024',
    'MLP-Combined1033':   'Combined 1033',
    'MLP-Wide-Combined':  'Combined 1033',
}
for r in results:
    log('{:<22} {:<18} {:>10.4f} {:>15} {:>12.1f}'.format(
        r['name'], label_map[r['name']], r['test_auc'], dlt(r['test_auc']), r['train_s']))

log('')
log('Reference baselines:')
log('{:<22} {:<18} {:>10.4f} {:>15}'.format(
    'CatBoost (Mark P1)',   'Combined 1036', CATBOOST_P1,    '(champion)'))
log('{:<22} {:<18} {:>10.4f} {:>15}'.format(
    'RF (Anthony P1)',      'Combined 1036', ANTHONY_RF_P1,  dlt(ANTHONY_RF_P1)))
log('{:<22} {:<18} {:>10.4f} {:>15}'.format(
    'GIN best (Anthony P2)','9 node feat',   ANTHONY_GIN_P2, dlt(ANTHONY_GIN_P2)))
log('{:<22} {:<18} {:>10.4f} {:>15}'.format(
    'DeeperGCN (SOTA)',     'full graph',    OGB_SOTA,       dlt(OGB_SOTA)))

# ---- Save JSON ----
out = {
    'phase': 2,
    'author': 'Mark',
    'date': '2026-04-07',
    'dataset': 'ogbg-molhiv (OGB)',
    'primary_metric': 'ROC-AUC',
    'split': 'OGB scaffold split',
    'research_question': (
        "Is Anthony Phase 2 GNN failure (-0.073 AUC vs CatBoost) GNN-specific, "
        "or do any neural models lose to tree-based learners on molecular fingerprints?"
    ),
    'baselines': {
        'catboost_mark_p1': CATBOOST_P1,
        'rf_anthony_p1': ANTHONY_RF_P1,
        'gin_anthony_p2_best': ANTHONY_GIN_P2,
        'graphsage_anthony_p2': ANTHONY_SAGE,
        'gcn_anthony_p2': ANTHONY_GCN,
        'gat_anthony_p2': ANTHONY_GAT,
        'ogb_sota_deepergcn': OGB_SOTA,
    },
    'experiments': [
        {
            'model': r['name'],
            'features': label_map[r['name']],
            'test_roc_auc': round(r['test_auc'], 4),
            'val_roc_auc':  round(r['val_auc'], 4),
            'test_f1_t0p5': round(r['test_f1_t0p5'], 4),
            'params': r['params'],
            'best_epoch': r['best_epoch'],
            'train_s': round(r['train_s'], 1),
            'delta_vs_catboost_p1': round(r['test_auc'] - CATBOOST_P1, 4),
            'delta_vs_anthony_gin_p2': round(r['test_auc'] - ANTHONY_GIN_P2, 4),
        } for r in results
    ],
    'histories': {r['name']: r['hist'] for r in results},
}
with open(RESULTS / 'phase2_mark_neural_baselines.json', 'w') as f:
    json.dump(out, f, indent=2)
log('\nSaved: results/phase2_mark_neural_baselines.json')

# Append to metrics.json (preserve Anthony's dict structure)
metrics_path = RESULTS / 'metrics.json'
with open(metrics_path) as f:
    raw = json.load(f)
if isinstance(raw, dict):
    raw['phase2_mark'] = out
else:
    raw.append(out)
with open(metrics_path, 'w') as f:
    json.dump(raw, f, indent=2)
log('Appended to: {}'.format(metrics_path))

# ---- Plots ----
log('\nGenerating plots (matplotlib)...')
try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    # ---------- Plot 1: full phase-2 comparison bar chart ----------
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    bar_models = [
        'DeeperGCN\n(SOTA)',
        'CatBoost\n(Phase 1)',
        'RF\n(Anthony P1)',
        'GIN\n(Anthony P2)',
        'SAGE\n(Anthony P2)',
        'GCN\n(Anthony P2)',
        'GAT\n(Anthony P2)',
        'MLP-Dom9\n(Mark)',
        'MLP-FP1024\n(Mark)',
        'MLP-Combined\n(Mark)',
        'MLP-Wide\n(Mark)',
    ]
    bar_aucs = [
        OGB_SOTA, CATBOOST_P1, ANTHONY_RF_P1,
        ANTHONY_GIN_P2, ANTHONY_SAGE, ANTHONY_GCN, ANTHONY_GAT,
        r1['test_auc'], r2['test_auc'], r3['test_auc'], r4['test_auc'],
    ]
    bar_colors = (
        ['#d62728'] + ['#2ca02c'] * 2 +
        ['#9ecae1'] * 4 +
        ['#6baed6', '#3182bd', '#08519c', '#084594']
    )

    bars = axes[0].bar(bar_models, bar_aucs, color=bar_colors,
                        edgecolor='white', linewidth=0.6, alpha=0.9)
    axes[0].axhline(CATBOOST_P1,    color='#2ca02c', linestyle='--', linewidth=1.2, alpha=0.6)
    axes[0].axhline(ANTHONY_GIN_P2, color='#9ecae1', linestyle=':',  linewidth=1.2, alpha=0.7)
    for bar, val in zip(bars, bar_aucs):
        axes[0].text(bar.get_x() + bar.get_width() / 2., bar.get_height() + 0.002,
                     '{:.4f}'.format(val), ha='center', va='bottom',
                     fontsize=6.5, fontweight='bold')
    axes[0].set_ylim(min(bar_aucs) - 0.02, max(bar_aucs) + 0.02)
    axes[0].set_ylabel('Test ROC-AUC')
    axes[0].set_title("Phase 2: Neural vs Tree vs GNN on ogbg-molhiv\n(scaffold split)")
    axes[0].tick_params(axis='x', labelsize=7)
    axes[0].legend(handles=[
        Patch(fc='#d62728', label='SOTA'),
        Patch(fc='#2ca02c', label='Phase 1 trees'),
        Patch(fc='#9ecae1', label='Anthony P2 GNNs'),
        Patch(fc='#3182bd', label='Mark P2 neural MLPs'),
    ], fontsize=7, loc='lower right')

    # Learning curves
    for r, col in zip([r1, r2, r3, r4], ['#6baed6', '#3182bd', '#08519c', '#084594']):
        vals = [h['val'] for h in r['hist']]
        axes[1].plot(range(1, len(vals) + 1), vals, label=r['name'],
                     color=col, linewidth=2, marker='o', markersize=3)
    axes[1].axhline(CATBOOST_P1, color='#2ca02c', linestyle='--', linewidth=1.5, alpha=0.6,
                    label='CatBoost P1 ({})'.format(CATBOOST_P1))
    axes[1].axhline(ANTHONY_GIN_P2, color='#9ecae1', linestyle=':', linewidth=1.5, alpha=0.8,
                    label='Anthony GIN P2 ({})'.format(ANTHONY_GIN_P2))
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Val ROC-AUC')
    axes[1].set_title('MLP Learning Curves (Mark Phase 2)')
    axes[1].legend(fontsize=8, loc='lower right')
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(RESULTS / 'phase2_mark_neural_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()
    log('Saved: results/phase2_mark_neural_comparison.png')

    # ---------- Plot 2: neural vs tree on identical features ----------
    fig2, ax = plt.subplots(figsize=(10, 6))
    feat_groups = ['Domain-only', 'Morgan FP 1024', 'Combined\n(default MLP)', 'Combined\n(wide MLP)']
    mlp_aucs    = [r1['test_auc'], r2['test_auc'], r3['test_auc'], r4['test_auc']]
    # Reference tree values from Phase 1 experiments (metrics.json)
    tree_aucs   = [0.7463, 0.7732, CATBOOST_P1, CATBOOST_P1]

    bar_w = 0.35
    x = np.arange(len(feat_groups))
    b1 = ax.bar(x - bar_w / 2, mlp_aucs,  bar_w, label='MLP (neural)',    color='#3182bd', alpha=0.85)
    b2 = ax.bar(x + bar_w / 2, tree_aucs, bar_w, label='Tree (CatBoost/LGBM/LR)', color='#2ca02c', alpha=0.85)
    for group in [b1, b2]:
        for bar in group:
            ax.text(bar.get_x() + bar.get_width() / 2., bar.get_height() + 0.002,
                    '{:.4f}'.format(bar.get_height()),
                    ha='center', va='bottom', fontsize=8, fontweight='bold')
    ax.axhline(ANTHONY_GIN_P2, color='#d62728', linestyle=':', linewidth=1.5,
               label='Anthony GIN P2 ({})'.format(ANTHONY_GIN_P2))
    ax.set_xticks(x)
    ax.set_xticklabels(feat_groups)
    ax.set_ylabel('Test ROC-AUC')
    ax.set_title("Phase 2 Mark: Neural vs Tree on identical feature sets\n(isolating the model-family effect)")
    ax.set_ylim(min(min(mlp_aucs), min(tree_aucs), ANTHONY_GIN_P2) - 0.02,
                max(max(mlp_aucs), max(tree_aucs)) + 0.03)
    ax.legend(loc='upper left')
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(RESULTS / 'phase2_mark_neural_vs_tree.png', dpi=150, bbox_inches='tight')
    plt.close()
    log('Saved: results/phase2_mark_neural_vs_tree.png')

except Exception as e:
    log('Plot step skipped due to: {}'.format(e))

log('\nDONE.')
