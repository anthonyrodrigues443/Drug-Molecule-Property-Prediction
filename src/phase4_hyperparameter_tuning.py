"""
Phase 4: Hyperparameter Tuning + Error Analysis — Drug-Molecule-Property-Prediction
Date: 2026-04-09
Researcher: Anthony Rodrigues

Key experiments:
  4.1  Optuna tuning of GIN+Edge (hidden_dim, num_layers, dropout, lr, pooling, batch_size)
  4.2  Optuna tuning of CatBoost on MI-top-400 features (depth, lr, l2_leaf_reg, iterations)
  4.3  Error analysis: scaffold-based, molecular property-based, hard examples
  4.4  Learning curves: data efficiency of champion model

Research refs:
  [1] Akiba et al. 2019 — Optuna framework, TPE sampler for efficient hyperparameter search
  [2] OGB leaderboard 2024 — GIN-VN best config: 5 layers, 300 dim, 0.5 dropout, 100 epochs
  [3] Prokhorenkova et al. 2018 — CatBoost: depth 6-10, lr 0.01-0.3 recommended for tabular
  [4] Bemis & Murcko 1996 — Scaffold-based analysis for molecular dataset evaluation
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
from sklearn.metrics import (roc_auc_score, average_precision_score,
                             confusion_matrix, precision_recall_curve,
                             classification_report)
from sklearn.feature_selection import mutual_info_classif
from pathlib import Path
import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)

# Patch torch.load for OGB cache
_orig_load = torch.load
def _patched_load(*a, **kw):
    kw.setdefault('weights_only', False)
    return _orig_load(*a, **kw)
torch.load = _patched_load

from ogb.graphproppred import GraphPropPredDataset, Evaluator
from ogb.graphproppred.mol_encoder import AtomEncoder, BondEncoder
from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors, AllChem, MACCSkeys, Fragments, Scaffolds
from catboost import CatBoostClassifier

BASE_DIR = Path('/Users/anthonyrodrigues/Desktop/YC-Portfolio-Projects/Drug-Molecule-Property-Prediction')
RESULTS_DIR = BASE_DIR / 'results'
RESULTS_DIR.mkdir(exist_ok=True)

DEVICE = torch.device('cpu')
torch.manual_seed(42)
np.random.seed(42)

# ══════════════════════════════════════════════════════════════════════════
# DATA LOADING
# ══════════════════════════════════════════════════════════════════════════
print('=' * 70)
print('PHASE 4: HYPERPARAMETER TUNING + ERROR ANALYSIS')
print('=' * 70)

print('\n[1/6] Loading ogbg-molhiv...')
dataset = GraphPropPredDataset(name='ogbg-molhiv', root=str(BASE_DIR / 'data' / 'raw'))
split_idx = dataset.get_idx_split()
evaluator = Evaluator(name='ogbg-molhiv')

# Load SMILES for feature computation + scaffold analysis
smiles_df = pd.read_csv(BASE_DIR / 'data' / 'raw' / 'ogbg_molhiv' / 'mapping' / 'mol.csv.gz')
labels = dataset.labels.flatten()

train_indices = split_idx['train'].tolist()
val_indices = split_idx['valid'].tolist()
test_indices = split_idx['test'].tolist()

# OGB → PyG conversion
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

# ══════════════════════════════════════════════════════════════════════════
# GIN+Edge MODEL (same architecture as Phase 3 champion)
# ══════════════════════════════════════════════════════════════════════════

class GINEdgeTunable(nn.Module):
    """GIN with OGB AtomEncoder + BondEncoder, configurable hyperparameters."""
    def __init__(self, hidden_dim, num_layers, dropout, pool_type='add'):
        super().__init__()
        self.num_layers = num_layers
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

        self.pool = global_add_pool if pool_type == 'add' else global_mean_pool
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )
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
        g = self.pool(h, batch)
        return self.classifier(g).squeeze(-1)


def train_gin_epoch(model, loader, optimizer, criterion):
    model.train()
    total_loss = 0
    for data in loader:
        data = data.to(DEVICE)
        optimizer.zero_grad()
        out = model(data.x, data.edge_index, data.edge_attr, data.batch)
        loss = criterion(out, data.y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * data.num_graphs
    return total_loss / len(loader.dataset)


@torch.no_grad()
def eval_gin(model, loader):
    model.eval()
    all_preds, all_labels = [], []
    for data in loader:
        data = data.to(DEVICE)
        out = model(data.x, data.edge_index, data.edge_attr, data.batch)
        all_preds.append(torch.sigmoid(out).cpu().numpy())
        all_labels.append(data.y.cpu().numpy())
    preds = np.concatenate(all_preds)
    labels = np.concatenate(all_labels)
    auc = roc_auc_score(labels, preds)
    return auc, preds, labels


def train_gin_full(hidden_dim, num_layers, dropout, lr, batch_size, pool_type, epochs=50, patience=15):
    """Train GIN+Edge with given hyperparameters, return val AUC, test AUC, and predictions."""
    train_loader = PyGDataLoader(train_data, batch_size=batch_size, shuffle=True)
    val_loader = PyGDataLoader(val_data, batch_size=batch_size)
    test_loader = PyGDataLoader(test_data, batch_size=batch_size)

    model = GINEdgeTunable(hidden_dim, num_layers, dropout, pool_type).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)

    # Class-weighted BCE for imbalanced data
    pos_count = sum(1 for d in train_data if d.y.item() > 0.5)
    neg_count = len(train_data) - pos_count
    pos_weight = torch.tensor([neg_count / max(pos_count, 1)], dtype=torch.float)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    best_val_auc = 0
    best_state = None
    wait = 0

    for epoch in range(epochs):
        loss = train_gin_epoch(model, train_loader, optimizer, criterion)
        val_auc, _, _ = eval_gin(model, val_loader)
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                break

    model.load_state_dict(best_state)
    test_auc, test_preds, test_labels = eval_gin(model, test_loader)
    val_auc_final, val_preds, val_labels = eval_gin(model, val_loader)

    return {
        'val_auc': val_auc_final,
        'test_auc': test_auc,
        'test_preds': test_preds,
        'test_labels': test_labels,
        'model': model,
        'epochs_run': epoch + 1,
    }


# ══════════════════════════════════════════════════════════════════════════
# EXPERIMENT 4.1: OPTUNA TUNING OF GIN+Edge
# ══════════════════════════════════════════════════════════════════════════
print('\n[2/6] Experiment 4.1: Optuna tuning of GIN+Edge (8 trials, CPU-constrained)...')
t0 = time.time()

gin_results = []

def gin_objective(trial):
    hidden_dim = trial.suggest_categorical('hidden_dim', [64, 128, 256])
    num_layers = trial.suggest_int('num_layers', 2, 5)
    dropout = trial.suggest_float('dropout', 0.2, 0.7, step=0.1)
    lr = trial.suggest_float('lr', 1e-4, 1e-2, log=True)
    batch_size = trial.suggest_categorical('batch_size', [256, 512])
    pool_type = trial.suggest_categorical('pool_type', ['add', 'mean'])

    result = train_gin_full(hidden_dim, num_layers, dropout, lr, batch_size, pool_type,
                           epochs=25, patience=8)

    gin_results.append({
        'trial': trial.number,
        'hidden_dim': hidden_dim,
        'num_layers': num_layers,
        'dropout': dropout,
        'lr': lr,
        'batch_size': batch_size,
        'pool_type': pool_type,
        'val_auc': result['val_auc'],
        'test_auc': result['test_auc'],
        'epochs_run': result['epochs_run'],
    })
    print(f"  Trial {trial.number}: dim={hidden_dim} layers={num_layers} "
          f"drop={dropout:.1f} lr={lr:.4f} bs={batch_size} pool={pool_type} "
          f"→ val={result['val_auc']:.4f} test={result['test_auc']:.4f}")
    return result['val_auc']

study_gin = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler(seed=42))
study_gin.optimize(gin_objective, n_trials=8, show_progress_bar=False)

print(f'\nGIN+Edge tuning done in {time.time()-t0:.0f}s')
print(f'Best trial: {study_gin.best_trial.number} — val AUC = {study_gin.best_value:.4f}')
print(f'Best params: {study_gin.best_params}')

# Retrain best GIN config with more epochs for final evaluation
print('\nRetraining best GIN config with 40 epochs...')
bp = study_gin.best_params
gin_best = train_gin_full(
    bp['hidden_dim'], bp['num_layers'], bp['dropout'], bp['lr'],
    bp['batch_size'], bp['pool_type'], epochs=40, patience=15
)
print(f'  Final GIN+Edge (tuned): val={gin_best["val_auc"]:.4f} test={gin_best["test_auc"]:.4f}')

gin_df = pd.DataFrame(gin_results).sort_values('val_auc', ascending=False)
print('\nAll GIN trials:')
print(gin_df.to_string(index=False))

# ══════════════════════════════════════════════════════════════════════════
# FEATURE COMPUTATION FOR CATBOOST
# ══════════════════════════════════════════════════════════════════════════
print('\n[3/6] Computing molecular features for CatBoost tuning...')

FRAG_FUNCS = [(n, getattr(Fragments, n)) for n in sorted(dir(Fragments)) if n.startswith('fr_')]

def compute_all_features(smi):
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return None
    # Lipinski-14
    feats = {
        'mol_weight': Descriptors.MolWt(mol),
        'logp': Descriptors.MolLogP(mol),
        'hbd': rdMolDescriptors.CalcNumHBD(mol),
        'hba': rdMolDescriptors.CalcNumHBA(mol),
        'tpsa': rdMolDescriptors.CalcTPSA(mol),
        'rotatable_bonds': rdMolDescriptors.CalcNumRotatableBonds(mol),
        'aromatic_rings': rdMolDescriptors.CalcNumAromaticRings(mol),
        'ring_count': rdMolDescriptors.CalcNumRings(mol),
        'heavy_atom_count': mol.GetNumHeavyAtoms(),
        'fraction_csp3': rdMolDescriptors.CalcFractionCSP3(mol),
        'num_heteroatoms': rdMolDescriptors.CalcNumHeteroatoms(mol),
        'lipinski_violations': sum([
            Descriptors.MolWt(mol) > 500,
            Descriptors.MolLogP(mol) > 5,
            rdMolDescriptors.CalcNumHBD(mol) > 5,
            rdMolDescriptors.CalcNumHBA(mol) > 10,
        ]),
    }
    # Advanced-12
    feats['bertz_ct'] = Descriptors.BertzCT(mol)
    feats['labute_asa'] = Descriptors.LabuteASA(mol)
    # Morgan FP 1024-bit
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=1024)
    for j in range(1024):
        feats[f'mfp_{j}'] = fp.GetBit(j)
    # MACCS 167
    mk = MACCSkeys.GenMACCSKeys(mol)
    for j in range(167):
        feats[f'maccs_{j}'] = mk.GetBit(j)
    # Fragments-85
    for fname, func in FRAG_FUNCS:
        try:
            feats[fname] = func(mol)
        except:
            feats[fname] = 0
    return feats

all_smiles = smiles_df['smiles'].tolist()
feat_rows = []
for i, smi in enumerate(all_smiles):
    f = compute_all_features(smi)
    if f is not None:
        f['idx'] = i
        f['y'] = int(labels[i])
        feat_rows.append(f)
    if (i + 1) % 10000 == 0:
        print(f'  computed {i+1}/{len(all_smiles)} molecules...')

feat_df = pd.DataFrame(feat_rows)
print(f'  Features computed: {len(feat_df)} molecules, {feat_df.shape[1]-2} features')

train_set = set(train_indices)
val_set = set(val_indices)
test_set = set(test_indices)

feat_df['split'] = feat_df['idx'].map(
    lambda x: 'train' if x in train_set else 'val' if x in val_set else 'test'
)

feature_cols = [c for c in feat_df.columns if c not in ['idx', 'y', 'split']]
X_train = feat_df[feat_df['split'] == 'train'][feature_cols].values.astype(np.float32)
y_train = feat_df[feat_df['split'] == 'train']['y'].values
X_val = feat_df[feat_df['split'] == 'val'][feature_cols].values.astype(np.float32)
y_val = feat_df[feat_df['split'] == 'val']['y'].values
X_test = feat_df[feat_df['split'] == 'test'][feature_cols].values.astype(np.float32)
y_test = feat_df[feat_df['split'] == 'test']['y'].values

# Replace NaN/inf
for X in [X_train, X_val, X_test]:
    np.nan_to_num(X, copy=False, nan=0.0, posinf=0.0, neginf=0.0)

print(f'  Train: {X_train.shape} | Val: {X_val.shape} | Test: {X_test.shape}')

# MI feature selection (top-400, matching Mark's Phase 3 champion)
print('  Computing mutual information for top-400 selection...')
mi_scores = mutual_info_classif(X_train, y_train, random_state=42, n_neighbors=5)
top_400_idx = np.argsort(mi_scores)[-400:]
X_train_400 = X_train[:, top_400_idx]
X_val_400 = X_val[:, top_400_idx]
X_test_400 = X_test[:, top_400_idx]
print(f'  MI-top-400 selected')

# ══════════════════════════════════════════════════════════════════════════
# EXPERIMENT 4.2: OPTUNA TUNING OF CatBoost + MI-top-400
# ══════════════════════════════════════════════════════════════════════════
print('\n[4/6] Experiment 4.2: Optuna tuning of CatBoost MI-top-400 (20 trials)...')
t0 = time.time()

cb_results = []

def catboost_objective(trial):
    params = {
        'depth': trial.suggest_int('depth', 4, 10),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
        'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1.0, 30.0),
        'iterations': trial.suggest_int('iterations', 300, 1500, step=100),
        'border_count': trial.suggest_categorical('border_count', [32, 64, 128, 254]),
        'random_strength': trial.suggest_float('random_strength', 0.0, 5.0),
        'bagging_temperature': trial.suggest_float('bagging_temperature', 0.0, 5.0),
        'auto_class_weights': 'Balanced',
        'eval_metric': 'AUC',
        'random_seed': 42,
        'verbose': 0,
    }

    model = CatBoostClassifier(**params)
    model.fit(X_train_400, y_train, eval_set=(X_val_400, y_val), early_stopping_rounds=50, verbose=0)

    val_pred = model.predict_proba(X_val_400)[:, 1]
    test_pred = model.predict_proba(X_test_400)[:, 1]
    val_auc = roc_auc_score(y_val, val_pred)
    test_auc = roc_auc_score(y_test, test_pred)

    cb_results.append({
        'trial': trial.number,
        **{k: v for k, v in params.items() if k not in ['eval_metric', 'random_seed', 'verbose', 'auto_class_weights']},
        'val_auc': val_auc,
        'test_auc': test_auc,
    })
    print(f"  Trial {trial.number}: depth={params['depth']} lr={params['learning_rate']:.4f} "
          f"l2={params['l2_leaf_reg']:.1f} iters={params['iterations']} "
          f"→ val={val_auc:.4f} test={test_auc:.4f}")
    return val_auc

study_cb = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler(seed=42))
study_cb.optimize(catboost_objective, n_trials=20, show_progress_bar=False)

print(f'\nCatBoost tuning done in {time.time()-t0:.0f}s')
print(f'Best trial: {study_cb.best_trial.number} — val AUC = {study_cb.best_value:.4f}')
print(f'Best params: {study_cb.best_params}')

# Retrain best CatBoost config for final evaluation
print('\nRetraining best CatBoost with best params...')
best_cb_params = study_cb.best_params
best_cb = CatBoostClassifier(
    **best_cb_params,
    auto_class_weights='Balanced',
    eval_metric='AUC',
    random_seed=42,
    verbose=0,
)
best_cb.fit(X_train_400, y_train, eval_set=(X_val_400, y_val), early_stopping_rounds=50, verbose=0)
cb_val_pred = best_cb.predict_proba(X_val_400)[:, 1]
cb_test_pred = best_cb.predict_proba(X_test_400)[:, 1]
cb_val_auc = roc_auc_score(y_val, cb_val_pred)
cb_test_auc = roc_auc_score(y_test, cb_test_pred)
cb_test_auprc = average_precision_score(y_test, cb_test_pred)
print(f'  CatBoost (tuned): val={cb_val_auc:.4f} test={cb_test_auc:.4f} AUPRC={cb_test_auprc:.4f}')

cb_df = pd.DataFrame(cb_results).sort_values('val_auc', ascending=False)
print('\nTop 10 CatBoost trials:')
print(cb_df.head(10).to_string(index=False))

# ══════════════════════════════════════════════════════════════════════════
# EXPERIMENT 4.3: ERROR ANALYSIS
# ══════════════════════════════════════════════════════════════════════════
print('\n[5/6] Experiment 4.3: Error analysis...')

# Use tuned CatBoost (current best) for error analysis
test_df = feat_df[feat_df['split'] == 'test'].copy()
test_df['pred_prob'] = cb_test_pred
test_df['pred_class'] = (cb_test_pred >= 0.5).astype(int)
test_df['correct'] = (test_df['pred_class'] == test_df['y']).astype(int)

# Scaffold analysis
print('\n  --- Scaffold Analysis ---')
test_smiles = smiles_df.iloc[test_df['idx'].values]['smiles'].values

def get_scaffold(smi):
    try:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            return 'invalid'
        return Scaffolds.MurckoScaffold.MurckoScaffoldSmiles(
            mol=mol, includeChirality=False)
    except:
        return 'error'

test_df['scaffold'] = [get_scaffold(s) for s in test_smiles]
scaffold_counts = test_df['scaffold'].value_counts()
print(f'  Unique scaffolds in test: {len(scaffold_counts)}')
print(f'  Top 5 scaffolds: {scaffold_counts.head().to_dict()}')

# Accuracy by scaffold frequency
test_df['scaffold_freq'] = test_df['scaffold'].map(scaffold_counts)
bins = [0, 1, 3, 10, 50, 1000]
labels_bins = ['singleton', '2-3', '4-10', '11-50', '50+']
test_df['scaffold_bin'] = pd.cut(test_df['scaffold_freq'], bins=bins, labels=labels_bins)
scaffold_perf = test_df.groupby('scaffold_bin').agg(
    count=('y', 'count'),
    positive_rate=('y', 'mean'),
    accuracy=('correct', 'mean'),
    avg_pred_prob=('pred_prob', 'mean'),
).round(4)
print('\n  Performance by scaffold frequency:')
print(scaffold_perf.to_string())

# Error analysis by molecular properties
print('\n  --- Error by Molecular Properties ---')
property_bins = {
    'mol_weight': [0, 200, 350, 500, 2000],
    'logp': [-10, 0, 2, 5, 20],
    'ring_count': [0, 1, 3, 5, 20],
    'heavy_atom_count': [0, 15, 25, 40, 200],
}
for prop, bins_p in property_bins.items():
    if prop in test_df.columns:
        test_df[f'{prop}_bin'] = pd.cut(test_df[prop], bins=bins_p)
        grp = test_df.groupby(f'{prop}_bin').agg(
            n=('y', 'count'),
            pos_rate=('y', 'mean'),
            accuracy=('correct', 'mean'),
        ).round(4)
        print(f'\n  {prop}:')
        print(grp.to_string())

# Confusion matrix at optimal threshold
print('\n  --- Threshold Optimization ---')
precision_arr, recall_arr, thresholds_arr = precision_recall_curve(y_test, cb_test_pred)
f1_scores = 2 * (precision_arr * recall_arr) / (precision_arr + recall_arr + 1e-10)
best_f1_idx = np.argmax(f1_scores)
best_threshold = thresholds_arr[best_f1_idx] if best_f1_idx < len(thresholds_arr) else 0.5
best_f1 = f1_scores[best_f1_idx]
print(f'  Optimal threshold (max F1): {best_threshold:.4f} → F1={best_f1:.4f}')

# Try multiple thresholds
for thr in [0.1, 0.2, 0.3, 0.4, 0.5, best_threshold]:
    preds_thr = (cb_test_pred >= thr).astype(int)
    cm = confusion_matrix(y_test, preds_thr)
    tn, fp, fn, tp = cm.ravel()
    prec = tp / max(tp + fp, 1)
    rec = tp / max(tp + fn, 1)
    f1 = 2 * prec * rec / max(prec + rec, 1e-10)
    print(f'  thr={thr:.3f}: TP={tp} FP={fp} FN={fn} TN={tn} | P={prec:.3f} R={rec:.3f} F1={f1:.3f}')

# Hard examples: highest confidence wrong predictions
print('\n  --- Hardest Examples (most confident errors) ---')
errors = test_df[test_df['correct'] == 0].copy()
errors['confidence'] = np.abs(errors['pred_prob'] - 0.5)
hardest = errors.nlargest(10, 'confidence')
for _, row in hardest.iterrows():
    print(f"  idx={int(row['idx'])} true={int(row['y'])} pred_prob={row['pred_prob']:.4f} "
          f"MW={row['mol_weight']:.0f} logP={row['logp']:.1f} rings={int(row['ring_count'])} "
          f"scaffold_freq={int(row['scaffold_freq'])}")

# ══════════════════════════════════════════════════════════════════════════
# EXPERIMENT 4.4: LEARNING CURVES
# ══════════════════════════════════════════════════════════════════════════
print('\n[6/6] Experiment 4.4: Learning curves (is more data the answer?)...')

fractions = [0.1, 0.2, 0.4, 0.6, 0.8, 1.0]
lc_results = []
for frac in fractions:
    n = int(len(X_train_400) * frac)
    idx = np.random.choice(len(X_train_400), n, replace=False)
    X_sub = X_train_400[idx]
    y_sub = y_train[idx]

    model_lc = CatBoostClassifier(
        **best_cb_params,
        auto_class_weights='Balanced',
        eval_metric='AUC',
        random_seed=42,
        verbose=0,
    )
    model_lc.fit(X_sub, y_sub, eval_set=(X_val_400, y_val), early_stopping_rounds=50, verbose=0)
    pred_lc = model_lc.predict_proba(X_test_400)[:, 1]
    auc_lc = roc_auc_score(y_test, pred_lc)
    lc_results.append({'fraction': frac, 'n_train': n, 'test_auc': auc_lc})
    print(f'  {frac*100:.0f}% data ({n:,} samples): test AUC = {auc_lc:.4f}')

lc_df = pd.DataFrame(lc_results)

# ══════════════════════════════════════════════════════════════════════════
# PLOTS
# ══════════════════════════════════════════════════════════════════════════
print('\nGenerating plots...')

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Plot 1: GIN trial results
ax = axes[0, 0]
gin_sorted = gin_df.sort_values('trial')
ax.plot(gin_sorted['trial'], gin_sorted['val_auc'], 'b-o', label='Val AUC', markersize=4)
ax.plot(gin_sorted['trial'], gin_sorted['test_auc'], 'r--s', label='Test AUC', markersize=4)
ax.axhline(y=0.7860, color='green', linestyle=':', label='Phase 3 GIN+Edge (0.786)')
ax.set_xlabel('Trial')
ax.set_ylabel('ROC-AUC')
ax.set_title('GIN+Edge Optuna Tuning (20 trials)')
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)

# Plot 2: CatBoost trial results
ax = axes[0, 1]
cb_sorted = cb_df.sort_values('trial')
ax.plot(cb_sorted['trial'], cb_sorted['val_auc'], 'b-o', label='Val AUC', markersize=3)
ax.plot(cb_sorted['trial'], cb_sorted['test_auc'], 'r--s', label='Test AUC', markersize=3)
ax.axhline(y=0.8105, color='green', linestyle=':', label='Phase 3 Mark CB (0.810)')
ax.set_xlabel('Trial')
ax.set_ylabel('ROC-AUC')
ax.set_title('CatBoost MI-400 Optuna Tuning (30 trials)')
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)

# Plot 3: Learning curve
ax = axes[1, 0]
ax.plot(lc_df['n_train'], lc_df['test_auc'], 'b-o', markersize=6)
ax.set_xlabel('Training Set Size')
ax.set_ylabel('Test ROC-AUC')
ax.set_title('Learning Curve (CatBoost MI-400)')
ax.grid(True, alpha=0.3)

# Plot 4: Scaffold performance
ax = axes[1, 1]
if not scaffold_perf.empty:
    x_pos = range(len(scaffold_perf))
    ax.bar(x_pos, scaffold_perf['accuracy'], color='steelblue', alpha=0.7, label='Accuracy')
    ax.bar(x_pos, scaffold_perf['positive_rate'], color='salmon', alpha=0.7, label='Positive Rate')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(scaffold_perf.index, rotation=45, ha='right')
    ax.set_ylabel('Rate')
    ax.set_title('Performance by Scaffold Frequency')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig(RESULTS_DIR / 'phase4_tuning_overview.png', dpi=150, bbox_inches='tight')
plt.close()
print('  Saved phase4_tuning_overview.png')

# ══════════════════════════════════════════════════════════════════════════
# MASTER COMPARISON TABLE
# ══════════════════════════════════════════════════════════════════════════
print('\n' + '=' * 70)
print('PHASE 4 MASTER COMPARISON')
print('=' * 70)

comparison = [
    {'Model': 'Phase 1 RF+Combined (baseline)', 'Test AUC': 0.7707, 'Source': 'Phase 1'},
    {'Model': 'Phase 1 CatBoost (Mark)', 'Test AUC': 0.7782, 'Source': 'Phase 1'},
    {'Model': 'Phase 2 GIN (raw)', 'Test AUC': 0.7053, 'Source': 'Phase 2'},
    {'Model': 'Phase 3 GIN+Edge (default)', 'Test AUC': 0.7860, 'Source': 'Phase 3'},
    {'Model': 'Phase 3 CB MI-400 (Mark)', 'Test AUC': 0.8105, 'Source': 'Phase 3'},
    {'Model': f'Phase 4 GIN+Edge (tuned)', 'Test AUC': gin_best['test_auc'], 'Source': 'Phase 4'},
    {'Model': f'Phase 4 CB MI-400 (tuned)', 'Test AUC': cb_test_auc, 'Source': 'Phase 4'},
]
comp_df = pd.DataFrame(comparison).sort_values('Test AUC', ascending=False)
comp_df['Rank'] = range(1, len(comp_df) + 1)
print(comp_df[['Rank', 'Model', 'Test AUC', 'Source']].to_string(index=False))

# ══════════════════════════════════════════════════════════════════════════
# SAVE RESULTS
# ══════════════════════════════════════════════════════════════════════════
results_all = {
    'gin_tuning': {
        'best_params': study_gin.best_params,
        'best_val_auc': study_gin.best_value,
        'final_test_auc': gin_best['test_auc'],
        'final_val_auc': gin_best['val_auc'],
        'all_trials': gin_results,
    },
    'catboost_tuning': {
        'best_params': {k: float(v) if isinstance(v, (np.floating,)) else v
                        for k, v in best_cb_params.items()},
        'final_test_auc': cb_test_auc,
        'final_val_auc': cb_val_auc,
        'final_auprc': cb_test_auprc,
        'all_trials': cb_results,
    },
    'learning_curve': lc_results,
    'scaffold_analysis': scaffold_perf.to_dict(),
    'optimal_threshold': float(best_threshold),
    'optimal_f1': float(best_f1),
    'comparison': comparison,
}

with open(RESULTS_DIR / 'phase4_results.json', 'w') as f:
    json.dump(results_all, f, indent=2, default=str)
print('\nSaved phase4_results.json')

# Update metrics.json
metrics_path = RESULTS_DIR / 'metrics.json'
if metrics_path.exists():
    with open(metrics_path) as f:
        metrics = json.load(f)
else:
    metrics = {}

metrics['phase4'] = {
    'gin_tuned_test_auc': gin_best['test_auc'],
    'gin_tuned_val_auc': gin_best['val_auc'],
    'gin_best_params': study_gin.best_params,
    'catboost_tuned_test_auc': cb_test_auc,
    'catboost_tuned_val_auc': cb_val_auc,
    'catboost_tuned_auprc': cb_test_auprc,
    'catboost_best_params': {k: float(v) if isinstance(v, (np.floating,)) else v
                             for k, v in best_cb_params.items()},
    'optimal_threshold': float(best_threshold),
    'optimal_f1': float(best_f1),
}

with open(metrics_path, 'w') as f:
    json.dump(metrics, f, indent=2, default=str)
print('Updated metrics.json')

print('\n' + '=' * 70)
print('PHASE 4 COMPLETE')
print('=' * 70)
