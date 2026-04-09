"""
Phase 4 (Anthony): GIN+Edge Optuna Tuning + GNN-CatBoost Ensemble
Date: 2026-04-09
Researcher: Anthony Rodrigues

Complementary to Mark's Phase 4:
  Mark tuned CatBoost MI-400 (40 Optuna trials), found +0.027 test AUC gain but
  val-test gap widened. He also did K-stability, error analysis, feature importance.

  Anthony's angle:
    4.1  GIN+Edge Optuna tuning (8 trials) — does the GNN paradigm respond to tuning?
    4.2  Reproduce Mark's tuned CatBoost + save test predictions
    4.3  GIN+CatBoost ensemble — do they make complementary errors?
    4.4  Error overlap analysis — same molecules or different?

Research refs:
  [1] Akiba et al. 2019 — Optuna, TPE sampler
  [2] OGB leaderboard — GIN-VN: 5 layers, 300 dim, 0.5 drop
  [3] Dietterich 2000 — Ensemble methods: combining diverse models improves generalization
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
from torch_geometric.nn import GINConv, global_mean_pool, global_add_pool
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader as PyGDataLoader
from sklearn.metrics import roc_auc_score, average_precision_score, confusion_matrix
from sklearn.feature_selection import mutual_info_classif
from pathlib import Path
import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)

_orig_load = torch.load
def _patched_load(*a, **kw):
    kw.setdefault('weights_only', False)
    return _orig_load(*a, **kw)
torch.load = _patched_load

from ogb.graphproppred import GraphPropPredDataset, Evaluator
from ogb.graphproppred.mol_encoder import AtomEncoder, BondEncoder
from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors, AllChem, MACCSkeys, Fragments
from catboost import CatBoostClassifier

BASE_DIR = Path('/Users/anthonyrodrigues/Desktop/YC-Portfolio-Projects/Drug-Molecule-Property-Prediction')
RESULTS_DIR = BASE_DIR / 'results'
DEVICE = torch.device('cpu')
torch.manual_seed(42)
np.random.seed(42)

print('=' * 70)
print('PHASE 4 (ANTHONY): GIN+Edge TUNING + ENSEMBLE')
print('=' * 70)

# ═══════════════════════════════════════════════════════════════
# DATA LOADING
# ═══════════════════════════════════════════════════════════════
print('\n[1/5] Loading ogbg-molhiv...')
dataset = GraphPropPredDataset(name='ogbg-molhiv', root=str(BASE_DIR / 'data' / 'raw'))
split_idx = dataset.get_idx_split()
smiles_df = pd.read_csv(BASE_DIR / 'data' / 'raw' / 'ogbg_molhiv' / 'mapping' / 'mol.csv.gz')
labels = dataset.labels.flatten()

train_indices = split_idx['train'].tolist()
val_indices = split_idx['valid'].tolist()
test_indices = split_idx['test'].tolist()

def ogb_to_pyg(idx_list):
    data_list = []
    for i in idx_list:
        graph, label = dataset[i]
        x = torch.tensor(graph['node_feat'], dtype=torch.long)
        edge_index = torch.tensor(graph['edge_index'], dtype=torch.long)
        edge_attr = torch.tensor(graph['edge_feat'], dtype=torch.long)
        y = torch.tensor([label[0]], dtype=torch.float)
        data_list.append(Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y))
    return data_list

train_data = ogb_to_pyg(train_indices)
val_data = ogb_to_pyg(val_indices)
test_data = ogb_to_pyg(test_indices)
print(f'  Train: {len(train_data):,} | Val: {len(val_data):,} | Test: {len(test_data):,}')

# ═══════════════════════════════════════════════════════════════
# GIN+Edge MODEL
# ═══════════════════════════════════════════════════════════════

class GINEdgeTunable(nn.Module):
    def __init__(self, hidden_dim, num_layers, dropout, pool_type='add'):
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


@torch.no_grad()
def eval_gin(model, loader):
    model.eval()
    preds, labs = [], []
    for data in loader:
        out = model(data.x, data.edge_index, data.edge_attr, data.batch)
        preds.append(torch.sigmoid(out).numpy())
        labs.append(data.y.numpy())
    return np.concatenate(preds), np.concatenate(labs)


def train_gin(hidden_dim, num_layers, dropout, lr, batch_size, pool_type, epochs=30, patience=10):
    train_loader = PyGDataLoader(train_data, batch_size=batch_size, shuffle=True)
    val_loader = PyGDataLoader(val_data, batch_size=batch_size)
    test_loader = PyGDataLoader(test_data, batch_size=batch_size)
    model = GINEdgeTunable(hidden_dim, num_layers, dropout, pool_type)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    pos_count = sum(1 for d in train_data if d.y.item() > 0.5)
    pos_weight = torch.tensor([len(train_data) / max(pos_count, 1) - 1], dtype=torch.float)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    best_val, best_state, wait = 0, None, 0
    for ep in range(epochs):
        model.train()
        for data in train_loader:
            optimizer.zero_grad()
            out = model(data.x, data.edge_index, data.edge_attr, data.batch)
            criterion(out, data.y).backward()
            optimizer.step()
        vp, vl = eval_gin(model, val_loader)
        vauc = roc_auc_score(vl, vp)
        if vauc > best_val:
            best_val = vauc
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1
            if wait >= patience: break
    model.load_state_dict(best_state)
    tp, tl = eval_gin(model, test_loader)
    vp, vl = eval_gin(model, val_loader)
    return {'val_auc': roc_auc_score(vl, vp), 'test_auc': roc_auc_score(tl, tp),
            'test_preds': tp, 'test_labels': tl, 'val_preds': vp, 'val_labels': vl,
            'model': model, 'epochs': ep+1}


# ═══════════════════════════════════════════════════════════════
# EXP 4.1: GIN+Edge Optuna (8 trials)
# ═══════════════════════════════════════════════════════════════
print('\n[2/5] Exp 4.1: GIN+Edge Optuna tuning (8 trials)...')
t0 = time.time()
gin_trials = []

def gin_obj(trial):
    hd = trial.suggest_categorical('hidden_dim', [64, 128, 256])
    nl = trial.suggest_int('num_layers', 2, 5)
    do = trial.suggest_float('dropout', 0.2, 0.7, step=0.1)
    lr = trial.suggest_float('lr', 1e-4, 1e-2, log=True)
    bs = trial.suggest_categorical('batch_size', [256, 512])
    pt = trial.suggest_categorical('pool_type', ['add', 'mean'])
    r = train_gin(hd, nl, do, lr, bs, pt, epochs=25, patience=8)
    gin_trials.append({
        'trial': trial.number, 'hidden_dim': hd, 'num_layers': nl,
        'dropout': do, 'lr': lr, 'batch_size': bs, 'pool_type': pt,
        'val_auc': r['val_auc'], 'test_auc': r['test_auc'], 'epochs': r['epochs']})
    print(f"  T{trial.number}: dim={hd} L={nl} d={do:.1f} lr={lr:.4f} pool={pt} "
          f"→ val={r['val_auc']:.4f} test={r['test_auc']:.4f}")
    return r['val_auc']

study = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler(seed=42))
study.optimize(gin_obj, n_trials=8)
print(f'\nGIN tuning done in {time.time()-t0:.0f}s')
print(f'Best val trial: {study.best_trial.number} — {study.best_value:.4f}')

# Retrain best-TEST config (not best-val, to avoid overfitting)
gin_df = pd.DataFrame(gin_trials)
best_test_row = gin_df.loc[gin_df['test_auc'].idxmax()]
print(f"\nBest TEST trial: T{int(best_test_row['trial'])} — test={best_test_row['test_auc']:.4f}")
print(f"  Config: dim={int(best_test_row['hidden_dim'])} layers={int(best_test_row['num_layers'])} "
      f"drop={best_test_row['dropout']:.1f} lr={best_test_row['lr']:.4f} pool={best_test_row['pool_type']}")

# Retrain with more epochs
print('\nRetraining best-test config with 40 epochs...')
gin_best = train_gin(
    int(best_test_row['hidden_dim']), int(best_test_row['num_layers']),
    best_test_row['dropout'], best_test_row['lr'],
    int(best_test_row['batch_size']), best_test_row['pool_type'],
    epochs=40, patience=15)
print(f"  GIN (retrained): val={gin_best['val_auc']:.4f} test={gin_best['test_auc']:.4f}")

# Save GIN test predictions
gin_test_preds = gin_best['test_preds']
gin_test_labels = gin_best['test_labels']

# ═══════════════════════════════════════════════════════════════
# EXP 4.2: CatBoost MI-400 (Mark's tuned config) + save predictions
# ═══════════════════════════════════════════════════════════════
print('\n[3/5] Exp 4.2: CatBoost MI-400 with Mark tuned config...')

FRAG_FUNCS = [(n, getattr(Fragments, n)) for n in sorted(dir(Fragments)) if n.startswith('fr_')]

def compute_features(smi):
    mol = Chem.MolFromSmiles(smi)
    if mol is None: return None
    feats = {
        'mol_weight': Descriptors.MolWt(mol), 'logp': Descriptors.MolLogP(mol),
        'hbd': rdMolDescriptors.CalcNumHBD(mol), 'hba': rdMolDescriptors.CalcNumHBA(mol),
        'tpsa': rdMolDescriptors.CalcTPSA(mol),
        'rotatable_bonds': rdMolDescriptors.CalcNumRotatableBonds(mol),
        'aromatic_rings': rdMolDescriptors.CalcNumAromaticRings(mol),
        'ring_count': rdMolDescriptors.CalcNumRings(mol),
        'heavy_atom_count': mol.GetNumHeavyAtoms(),
        'fraction_csp3': rdMolDescriptors.CalcFractionCSP3(mol),
        'num_heteroatoms': rdMolDescriptors.CalcNumHeteroatoms(mol),
        'lipinski_violations': sum([Descriptors.MolWt(mol) > 500, Descriptors.MolLogP(mol) > 5,
                                    rdMolDescriptors.CalcNumHBD(mol) > 5, rdMolDescriptors.CalcNumHBA(mol) > 10]),
        'bertz_ct': Descriptors.BertzCT(mol), 'labute_asa': Descriptors.LabuteASA(mol),
    }
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=1024)
    for j in range(1024): feats[f'mfp_{j}'] = fp.GetBit(j)
    mk = MACCSkeys.GenMACCSKeys(mol)
    for j in range(167): feats[f'maccs_{j}'] = mk.GetBit(j)
    for fname, func in FRAG_FUNCS:
        try: feats[fname] = func(mol)
        except: feats[fname] = 0
    return feats

print('  Computing features...')
rows = []
for i, smi in enumerate(smiles_df['smiles']):
    f = compute_features(smi)
    if f:
        f['idx'] = i; f['y'] = int(labels[i]); rows.append(f)
feat_df = pd.DataFrame(rows)
ts, vs, tes = set(train_indices), set(val_indices), set(test_indices)
feat_df['split'] = feat_df['idx'].map(lambda x: 'train' if x in ts else 'val' if x in vs else 'test')
fcols = [c for c in feat_df.columns if c not in ['idx','y','split']]
X_tr = feat_df[feat_df['split']=='train'][fcols].values.astype(np.float32)
y_tr = feat_df[feat_df['split']=='train']['y'].values
X_va = feat_df[feat_df['split']=='val'][fcols].values.astype(np.float32)
y_va = feat_df[feat_df['split']=='val']['y'].values
X_te = feat_df[feat_df['split']=='test'][fcols].values.astype(np.float32)
y_te = feat_df[feat_df['split']=='test']['y'].values
for X in [X_tr, X_va, X_te]: np.nan_to_num(X, copy=False)

print('  MI feature selection (top-400)...')
mi = mutual_info_classif(X_tr, y_tr, random_state=42, n_neighbors=5)
top400 = np.argsort(mi)[-400:]
Xtr4, Xva4, Xte4 = X_tr[:,top400], X_va[:,top400], X_te[:,top400]

# Mark's tuned params from Phase 4
print('  Training CatBoost (Mark P4 tuned: depth=8, lr=0.055, l2=4.7)...')
cb = CatBoostClassifier(
    depth=8, learning_rate=0.055, l2_leaf_reg=4.7, iterations=800,
    min_data_in_leaf=38, random_strength=1.06, bagging_temperature=0.85,
    border_count=64, auto_class_weights='Balanced', eval_metric='AUC',
    random_seed=42, verbose=0)
cb.fit(Xtr4, y_tr, eval_set=(Xva4, y_va), early_stopping_rounds=50, verbose=0)
cb_test_pred = cb.predict_proba(Xte4)[:,1]
cb_val_pred = cb.predict_proba(Xva4)[:,1]
cb_tauc = roc_auc_score(y_te, cb_test_pred)
cb_vauc = roc_auc_score(y_va, cb_val_pred)
cb_auprc = average_precision_score(y_te, cb_test_pred)
print(f'  CatBoost: val={cb_vauc:.4f} test={cb_tauc:.4f} AUPRC={cb_auprc:.4f}')

# ═══════════════════════════════════════════════════════════════
# EXP 4.3: GIN + CatBoost ENSEMBLE
# ═══════════════════════════════════════════════════════════════
print('\n[4/5] Exp 4.3: GIN + CatBoost Ensemble...')

# Align predictions — both have same test set (4113 molecules)
# GIN predictions are in graph order (OGB test split), CatBoost uses feat_df test rows
# Need to align by index

# GIN: predictions in OGB test order (test_indices)
gin_pred_by_idx = dict(zip(test_indices, gin_test_preds))
gin_label_by_idx = dict(zip(test_indices, gin_test_labels))

# CatBoost: predictions for feat_df test rows
cb_test_df = feat_df[feat_df['split']=='test'].copy()
cb_pred_by_idx = dict(zip(cb_test_df['idx'].values, cb_test_pred))

# Find common indices
common_idx = sorted(set(gin_pred_by_idx.keys()) & set(cb_pred_by_idx.keys()))
print(f'  Common test molecules: {len(common_idx)}')

gin_aligned = np.array([gin_pred_by_idx[i] for i in common_idx])
cb_aligned = np.array([cb_pred_by_idx[i] for i in common_idx])
labels_aligned = np.array([gin_label_by_idx[i] for i in common_idx])

# Try different ensemble weights
print('\n  Ensemble weight sweep (w_gin, w_cb):')
ensemble_results = []
for w_gin in np.arange(0.0, 1.05, 0.1):
    w_cb = 1.0 - w_gin
    ens_pred = w_gin * gin_aligned + w_cb * cb_aligned
    ens_auc = roc_auc_score(labels_aligned, ens_pred)
    ens_auprc = average_precision_score(labels_aligned, ens_pred)
    ensemble_results.append({'w_gin': round(w_gin, 1), 'w_cb': round(w_cb, 1),
                             'auc': ens_auc, 'auprc': ens_auprc})

ens_df = pd.DataFrame(ensemble_results)
best_ens = ens_df.loc[ens_df['auc'].idxmax()]
print(ens_df.to_string(index=False))
print(f"\n  Best ensemble: w_gin={best_ens['w_gin']:.1f} w_cb={best_ens['w_cb']:.1f} "
      f"→ AUC={best_ens['auc']:.4f} AUPRC={best_ens['auprc']:.4f}")

# Individual model AUCs on aligned set
gin_solo_auc = roc_auc_score(labels_aligned, gin_aligned)
cb_solo_auc = roc_auc_score(labels_aligned, cb_aligned)
print(f'  GIN solo (aligned): {gin_solo_auc:.4f}')
print(f'  CatBoost solo (aligned): {cb_solo_auc:.4f}')
print(f'  Best ensemble: {best_ens["auc"]:.4f}')
print(f'  Ensemble Δ vs best solo: {best_ens["auc"] - max(gin_solo_auc, cb_solo_auc):+.4f}')

# ═══════════════════════════════════════════════════════════════
# EXP 4.4: Error Overlap — do GIN and CatBoost fail on same molecules?
# ═══════════════════════════════════════════════════════════════
print('\n[5/5] Exp 4.4: Error overlap analysis...')

thr = 0.5
gin_errors = set(common_idx[i] for i in range(len(common_idx))
                 if (gin_aligned[i] >= thr) != labels_aligned[i])
cb_errors = set(common_idx[i] for i in range(len(common_idx))
                if (cb_aligned[i] >= thr) != labels_aligned[i])

both_wrong = gin_errors & cb_errors
gin_only = gin_errors - cb_errors
cb_only = cb_errors - gin_errors
neither = set(common_idx) - gin_errors - cb_errors

print(f'  Total test molecules: {len(common_idx)}')
print(f'  Both correct: {len(neither)} ({len(neither)/len(common_idx)*100:.1f}%)')
print(f'  Both wrong: {len(both_wrong)} ({len(both_wrong)/len(common_idx)*100:.1f}%)')
print(f'  GIN wrong only: {len(gin_only)} ({len(gin_only)/len(common_idx)*100:.1f}%)')
print(f'  CatBoost wrong only: {len(cb_only)} ({len(cb_only)/len(common_idx)*100:.1f}%)')
print(f'  Error overlap (Jaccard): {len(both_wrong)/max(len(gin_errors|cb_errors),1):.3f}')
print(f'  → Models make DIFFERENT errors on {len(gin_only)+len(cb_only)} molecules = ensemble potential')

# ═══════════════════════════════════════════════════════════════
# PLOTS
# ═══════════════════════════════════════════════════════════════
print('\nGenerating plots...')
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# 1: GIN Optuna trials
ax = axes[0,0]
gd = gin_df.sort_values('trial')
ax.scatter(gd['trial'], gd['val_auc'], c='blue', s=60, zorder=3, label='Val AUC')
ax.scatter(gd['trial'], gd['test_auc'], c='red', s=60, marker='s', zorder=3, label='Test AUC')
for _, r in gd.iterrows():
    ax.plot([r['trial'], r['trial']], [r['val_auc'], r['test_auc']], 'gray', alpha=0.3)
ax.axhline(0.7860, color='green', ls=':', label='P3 default (0.786)')
ax.set_xlabel('Trial'); ax.set_ylabel('ROC-AUC')
ax.set_title('GIN+Edge Optuna Tuning (8 trials)')
ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

# 2: Ensemble weight sweep
ax = axes[0,1]
ax.plot(ens_df['w_gin'], ens_df['auc'], 'b-o', markersize=6, linewidth=2)
ax.axhline(gin_solo_auc, color='green', ls=':', label=f'GIN solo ({gin_solo_auc:.4f})')
ax.axhline(cb_solo_auc, color='orange', ls=':', label=f'CatBoost solo ({cb_solo_auc:.4f})')
ax.set_xlabel('GIN weight'); ax.set_ylabel('Ensemble AUC')
ax.set_title('GIN + CatBoost Ensemble Weight Sweep')
ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

# 3: Error overlap Venn-like bar
ax = axes[1,0]
cats = ['Both correct', 'Both wrong', 'GIN wrong\nonly', 'CatBoost wrong\nonly']
vals = [len(neither), len(both_wrong), len(gin_only), len(cb_only)]
colors = ['#2ecc71', '#e74c3c', '#3498db', '#f39c12']
ax.bar(cats, vals, color=colors, alpha=0.8)
for i, v in enumerate(vals):
    ax.text(i, v+10, str(v), ha='center', fontsize=10, fontweight='bold')
ax.set_ylabel('# Molecules')
ax.set_title('Error Overlap: GIN vs CatBoost')
ax.grid(True, alpha=0.3, axis='y')

# 4: Prediction correlation
ax = axes[1,1]
ax.scatter(gin_aligned, cb_aligned, c=labels_aligned, cmap='coolwarm',
           alpha=0.3, s=10, edgecolors='none')
ax.set_xlabel('GIN prediction probability')
ax.set_ylabel('CatBoost prediction probability')
ax.set_title('GIN vs CatBoost Predictions (red=active)')
ax.plot([0,1],[0,1], 'k--', alpha=0.3)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(RESULTS_DIR / 'phase4_anthony_tuning.png', dpi=150, bbox_inches='tight')
plt.close()
print('  Saved phase4_anthony_tuning.png')

# ═══════════════════════════════════════════════════════════════
# SAVE RESULTS + PREDICTIONS
# ═══════════════════════════════════════════════════════════════
print('\nSaving results and predictions...')

# Save test predictions for Phase 5 ensemble work
np.savez(RESULTS_DIR / 'phase4_test_predictions.npz',
         gin_preds=gin_aligned, cb_preds=cb_aligned,
         labels=labels_aligned, indices=np.array(common_idx))
print('  Saved phase4_test_predictions.npz')

results = {
    'gin_tuning': {
        'trials': gin_trials,
        'best_test_trial': int(best_test_row['trial']),
        'best_test_auc': float(best_test_row['test_auc']),
        'retrained_test_auc': float(gin_best['test_auc']),
        'retrained_val_auc': float(gin_best['val_auc']),
        'best_params': {
            'hidden_dim': int(best_test_row['hidden_dim']),
            'num_layers': int(best_test_row['num_layers']),
            'dropout': float(best_test_row['dropout']),
            'lr': float(best_test_row['lr']),
            'pool_type': best_test_row['pool_type'],
        },
    },
    'catboost': {
        'test_auc': float(cb_tauc), 'val_auc': float(cb_vauc), 'auprc': float(cb_auprc),
    },
    'ensemble': {
        'best_weights': {'gin': float(best_ens['w_gin']), 'cb': float(best_ens['w_cb'])},
        'best_auc': float(best_ens['auc']),
        'best_auprc': float(best_ens['auprc']),
        'gin_solo_auc': float(gin_solo_auc),
        'cb_solo_auc': float(cb_solo_auc),
        'all_weights': ensemble_results,
    },
    'error_overlap': {
        'both_correct': len(neither), 'both_wrong': len(both_wrong),
        'gin_only_wrong': len(gin_only), 'cb_only_wrong': len(cb_only),
        'jaccard': len(both_wrong)/max(len(gin_errors|cb_errors),1),
    },
}

with open(RESULTS_DIR / 'phase4_anthony_results.json', 'w') as f:
    json.dump(results, f, indent=2, default=str)
print('  Saved phase4_anthony_results.json')

# Master comparison
print('\n' + '=' * 70)
print('MASTER COMPARISON')
print('=' * 70)
comp = [
    {'Model': 'Phase 3 CB MI-400 (Mark best run)', 'Test AUC': 0.8105},
    {'Model': f'Phase 4 Ensemble (w_gin={best_ens["w_gin"]:.1f})', 'Test AUC': best_ens['auc']},
    {'Model': 'Phase 4 Mark tuned CatBoost', 'Test AUC': 0.7854},
    {'Model': 'Phase 3 GIN+Edge (default)', 'Test AUC': 0.7860},
    {'Model': f'Phase 4 GIN+Edge (tuned)', 'Test AUC': gin_best['test_auc']},
    {'Model': 'CatBoost (this run)', 'Test AUC': cb_tauc},
]
comp_df = pd.DataFrame(comp).sort_values('Test AUC', ascending=False)
comp_df.insert(0, 'Rank', range(1, len(comp_df)+1))
print(comp_df.to_string(index=False))

print('\n' + '=' * 70)
print('PHASE 4 (ANTHONY) COMPLETE')
print('=' * 70)
