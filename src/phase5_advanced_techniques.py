"""
Phase 5 (Anthony): Advanced Techniques + Ablation + LLM Comparison
Date: 2026-04-10
Researcher: Anthony Rodrigues

Building on Phase 4 champion (GIN+CatBoost ensemble = 0.8114 AUC).

Experiments:
  5.1  Ablation study — remove each component, measure contribution
  5.2  Stacking meta-learner — LogReg on [GIN_prob, CB_prob] vs simple weighted avg
  5.3  Focal loss GIN — one config to test impact on hard examples
  5.4  Error analysis on hard molecules
  5.5  Master comparison table across all phases

Research refs:
  [1] Lin et al. 2017 — Focal loss for dense object detection
  [2] Wolpert 1992 — Stacked generalization
  [3] Hu et al. 2020 — OGB leaderboard: GIN-VN 0.7707±0.0149
"""
import os, json, time, warnings, sys, logging, gc
warnings.filterwarnings('ignore')
os.environ['MPLBACKEND'] = 'Agg'
os.environ['PYTHONWARNINGS'] = 'ignore'
logging.getLogger('rdkit').setLevel(logging.CRITICAL)
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')

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
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import mutual_info_classif
from pathlib import Path

_orig_load = torch.load
def _patched_load(*a, **kw):
    kw.setdefault('weights_only', False)
    return _orig_load(*a, **kw)
torch.load = _patched_load

from ogb.graphproppred import GraphPropPredDataset
from ogb.graphproppred.mol_encoder import AtomEncoder, BondEncoder
from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors, AllChem, MACCSkeys, Fragments
from catboost import CatBoostClassifier

BASE_DIR = Path('/Users/anthonyrodrigues/Desktop/YC-Portfolio-Projects/Drug-Molecule-Property-Prediction')
RESULTS_DIR = BASE_DIR / 'results'
torch.manual_seed(42)
np.random.seed(42)

print('=' * 70)
print('PHASE 5 (ANTHONY): ADVANCED TECHNIQUES + ABLATION')
print('=' * 70)

# ═══════════════════════════════════════════════════════════════
# DATA LOADING + FEATURES
# ═══════════════════════════════════════════════════════════════
print('\n[1/6] Loading ogbg-molhiv + computing features...')
dataset = GraphPropPredDataset(name='ogbg-molhiv', root=str(BASE_DIR / 'data' / 'raw'))
split_idx = dataset.get_idx_split()
smiles_df = pd.read_csv(BASE_DIR / 'data' / 'raw' / 'ogbg_molhiv' / 'mapping' / 'mol.csv.gz')
labels = dataset.labels.flatten()

train_indices = split_idx['train'].tolist()
val_indices = split_idx['valid'].tolist()
test_indices = split_idx['test'].tolist()

FRAG_FUNCS = [(n, getattr(Fragments, n)) for n in sorted(dir(Fragments)) if n.startswith('fr_')]

def compute_features(smi):
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return None
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
    for j in range(1024):
        feats[f'mfp_{j}'] = fp.GetBit(j)
    mk = MACCSkeys.GenMACCSKeys(mol)
    for j in range(167):
        feats[f'maccs_{j}'] = mk.GetBit(j)
    for fname, func in FRAG_FUNCS:
        try:
            feats[fname] = func(mol)
        except:
            feats[fname] = 0
    return feats

rows, valid_indices = [], []
for i, smi in enumerate(smiles_df['smiles']):
    f = compute_features(smi)
    if f:
        f['idx'] = i; f['y'] = int(labels[i]); rows.append(f); valid_indices.append(i)
feat_df = pd.DataFrame(rows)
valid_set = set(valid_indices)

ts, vs, tes = set(train_indices), set(val_indices), set(test_indices)
feat_df['split'] = feat_df['idx'].map(lambda x: 'train' if x in ts else 'val' if x in vs else 'test')
fcols = [c for c in feat_df.columns if c not in ['idx', 'y', 'split']]

X_tr = feat_df[feat_df['split'] == 'train'][fcols].values.astype(np.float32)
y_tr = feat_df[feat_df['split'] == 'train']['y'].values
X_va = feat_df[feat_df['split'] == 'val'][fcols].values.astype(np.float32)
y_va = feat_df[feat_df['split'] == 'val']['y'].values
X_te = feat_df[feat_df['split'] == 'test'][fcols].values.astype(np.float32)
y_te = feat_df[feat_df['split'] == 'test']['y'].values
for X in [X_tr, X_va, X_te]:
    np.nan_to_num(X, copy=False)

print(f'  Valid: {len(feat_df):,} / {len(smiles_df):,} | Test={len(X_te)} Val={len(X_va)}')

mi = mutual_info_classif(X_tr, y_tr, random_state=42, n_neighbors=5)
top400 = np.argsort(mi)[-400:]
X_tr_400 = X_tr[:, top400]; X_va_400 = X_va[:, top400]; X_te_400 = X_te[:, top400]

print('  Training CatBoost MI-400...')
cb = CatBoostClassifier(iterations=1000, depth=6, learning_rate=0.05, auto_class_weights='Balanced',
                        random_seed=42, verbose=0, eval_metric='AUC', early_stopping_rounds=50)
cb.fit(X_tr_400, y_tr, eval_set=(X_va_400, y_va))
cb_test_preds = cb.predict_proba(X_te_400)[:, 1]
cb_val_preds = cb.predict_proba(X_va_400)[:, 1]
cb_test_auc = roc_auc_score(y_te, cb_test_preds)
cb_test_auprc = average_precision_score(y_te, cb_test_preds)
print(f'  CatBoost MI-400: test={cb_test_auc:.4f} AUPRC={cb_test_auprc:.4f}')

# ═══════════════════════════════════════════════════════════════
# GIN+Edge (aligned with CatBoost valid indices)
# ═══════════════════════════════════════════════════════════════
def ogb_to_pyg(idx_list):
    data_list = []
    for i in idx_list:
        if i not in valid_set:
            continue
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
assert len(test_data) == len(X_te), f"Alignment: GIN={len(test_data)} vs CB={len(X_te)}"
print(f'  Aligned: Train={len(train_data)} Val={len(val_data)} Test={len(test_data)}')

class GINEdge(nn.Module):
    def __init__(self, hidden_dim, num_layers, dropout, pool_type='add'):
        super().__init__()
        self.num_layers = num_layers
        self.atom_encoder = AtomEncoder(hidden_dim)
        self.bond_encoders = nn.ModuleList([BondEncoder(hidden_dim) for _ in range(num_layers)])
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        for _ in range(num_layers):
            mlp = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.BatchNorm1d(hidden_dim),
                                nn.ReLU(), nn.Linear(hidden_dim, hidden_dim))
            self.convs.append(GINConv(mlp, train_eps=True))
            self.bns.append(nn.BatchNorm1d(hidden_dim))
        self.pool = global_add_pool if pool_type == 'add' else global_mean_pool
        self.classifier = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
                                        nn.Dropout(dropout), nn.Linear(hidden_dim, 1))
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
        return self.classifier(self.pool(h, batch)).squeeze(-1)

class GINNoEdge(nn.Module):
    def __init__(self, hidden_dim, num_layers, dropout, pool_type='add'):
        super().__init__()
        self.num_layers = num_layers
        self.atom_encoder = AtomEncoder(hidden_dim)
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        for _ in range(num_layers):
            mlp = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.BatchNorm1d(hidden_dim),
                                nn.ReLU(), nn.Linear(hidden_dim, hidden_dim))
            self.convs.append(GINConv(mlp, train_eps=True))
            self.bns.append(nn.BatchNorm1d(hidden_dim))
        self.pool = global_add_pool if pool_type == 'add' else global_mean_pool
        self.classifier = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
                                        nn.Dropout(dropout), nn.Linear(hidden_dim, 1))
        self.dropout = dropout

    def forward(self, x, edge_index, edge_attr, batch):
        h = self.atom_encoder(x)
        for layer in range(self.num_layers):
            h = self.convs[layer](h, edge_index)
            h = self.bns[layer](h)
            if layer < self.num_layers - 1:
                h = F.relu(h)
                h = F.dropout(h, p=self.dropout, training=self.training)
        return self.classifier(self.pool(h, batch)).squeeze(-1)

class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0):
        super().__init__()
        self.alpha = alpha; self.gamma = gamma
    def forward(self, logits, targets):
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
        p_t = torch.sigmoid(logits) * targets + (1 - torch.sigmoid(logits)) * (1 - targets)
        return (self.alpha * ((1 - p_t) ** self.gamma) * bce).mean()

@torch.no_grad()
def eval_model(model, loader):
    model.eval()
    preds, labs = [], []
    for data in loader:
        out = model(data.x, data.edge_index, data.edge_attr, data.batch)
        preds.append(torch.sigmoid(out).numpy()); labs.append(data.y.numpy())
    return np.concatenate(preds), np.concatenate(labs)

def train_gnn(model_class, use_focal=False, focal_gamma=2.0, focal_alpha=0.25):
    train_loader = PyGDataLoader(train_data, batch_size=512, shuffle=True)
    val_loader = PyGDataLoader(val_data, batch_size=512)
    test_loader = PyGDataLoader(test_data, batch_size=512)
    model = model_class(64, 3, 0.4, 'add')
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0037, weight_decay=1e-5)
    pos_count = sum(1 for d in train_data if d.y.item() > 0.5)
    pos_weight = torch.tensor([len(train_data) / max(pos_count, 1) - 1])
    criterion = FocalLoss(focal_alpha, focal_gamma) if use_focal else nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    best_val, best_state, wait = 0, None, 0
    for ep in range(40):
        model.train()
        for data in train_loader:
            optimizer.zero_grad()
            criterion(model(data.x, data.edge_index, data.edge_attr, data.batch), data.y).backward()
            optimizer.step()
        vp, vl = eval_model(model, val_loader)
        vauc = roc_auc_score(vl, vp)
        if vauc > best_val:
            best_val = vauc; best_state = {k: v.clone() for k, v in model.state_dict().items()}; wait = 0
        else:
            wait += 1
            if wait >= 15: break
    model.load_state_dict(best_state)
    tp, tl = eval_model(model, test_loader)
    vp, vl = eval_model(model, val_loader)
    del train_loader, val_loader, test_loader, optimizer; gc.collect()
    return {'val_auc': roc_auc_score(vl, vp), 'test_auc': roc_auc_score(tl, tp),
            'test_auprc': average_precision_score(tl, tp),
            'test_preds': tp, 'test_labels': tl, 'val_preds': vp, 'val_labels': vl, 'epochs': ep+1}

# ═══════════════════════════════════════════════════════════════
# TRAIN GIN+Edge
# ═══════════════════════════════════════════════════════════════
print('\n[2/6] Training GIN+Edge (tuned: 64d, 3L, drop=0.4)...')
t0 = time.time()
gin = train_gnn(GINEdge)
print(f'  GIN+Edge: val={gin["val_auc"]:.4f} test={gin["test_auc"]:.4f} AUPRC={gin["test_auprc"]:.4f} ({time.time()-t0:.0f}s)')

ens_preds = 0.3 * gin['test_preds'] + 0.7 * cb_test_preds
ens_auc = roc_auc_score(y_te, ens_preds)
ens_auprc = average_precision_score(y_te, ens_preds)
print(f'  Ensemble (0.3/0.7): test={ens_auc:.4f} AUPRC={ens_auprc:.4f}')

results = {}

# ═══════════════════════════════════════════════════════════════
# EXP 5.1: ABLATION
# ═══════════════════════════════════════════════════════════════
print('\n[3/6] ABLATION STUDY')
ablation = [{'config': 'Full Ensemble', 'auc': ens_auc, 'auprc': ens_auprc, 'delta': 0.0, 'verdict': 'BASELINE'}]

ablation.append({'config': 'Remove GIN', 'auc': cb_test_auc, 'auprc': cb_test_auprc,
                 'delta': cb_test_auc - ens_auc, 'verdict': ''})
ablation.append({'config': 'Remove CatBoost', 'auc': gin['test_auc'], 'auprc': gin['test_auprc'],
                 'delta': gin['test_auc'] - ens_auc, 'verdict': ''})

# Remove MI selection
print('  CatBoost without MI selection...')
cb_all = CatBoostClassifier(iterations=1000, depth=6, learning_rate=0.05, auto_class_weights='Balanced',
                            random_seed=42, verbose=0, eval_metric='AUC', early_stopping_rounds=50)
cb_all.fit(X_tr, y_tr, eval_set=(X_va, y_va))
cb_all_preds = cb_all.predict_proba(X_te)[:, 1]
ens_no_mi = 0.3 * gin['test_preds'] + 0.7 * cb_all_preds
ens_no_mi_auc = roc_auc_score(y_te, ens_no_mi)
ablation.append({'config': 'Remove MI selection', 'auc': ens_no_mi_auc,
                 'auprc': average_precision_score(y_te, ens_no_mi),
                 'delta': ens_no_mi_auc - ens_auc, 'verdict': ''})
del cb_all; gc.collect()

# Remove edge features
print('  GIN without edge features...')
gin_ne = train_gnn(GINNoEdge)
ens_no_edge = 0.3 * gin_ne['test_preds'] + 0.7 * cb_test_preds
ens_no_edge_auc = roc_auc_score(y_te, ens_no_edge)
ablation.append({'config': 'Remove edge features', 'auc': ens_no_edge_auc,
                 'auprc': average_precision_score(y_te, ens_no_edge),
                 'delta': ens_no_edge_auc - ens_auc, 'verdict': ''})
del gin_ne; gc.collect()

# Remove class weighting
print('  CatBoost without class weighting...')
cb_nw = CatBoostClassifier(iterations=1000, depth=6, learning_rate=0.05,
                           random_seed=42, verbose=0, eval_metric='AUC', early_stopping_rounds=50)
cb_nw.fit(X_tr_400, y_tr, eval_set=(X_va_400, y_va))
cb_nw_preds = cb_nw.predict_proba(X_te_400)[:, 1]
ens_no_wt = 0.3 * gin['test_preds'] + 0.7 * cb_nw_preds
ens_no_wt_auc = roc_auc_score(y_te, ens_no_wt)
ablation.append({'config': 'Remove class weights', 'auc': ens_no_wt_auc,
                 'auprc': average_precision_score(y_te, ens_no_wt),
                 'delta': ens_no_wt_auc - ens_auc, 'verdict': ''})
del cb_nw; gc.collect()

for a in ablation[1:]:
    a['verdict'] = 'CRITICAL' if a['delta'] < -0.01 else 'IMPORTANT' if a['delta'] < -0.005 else 'MARGINAL' if a['delta'] < 0 else 'NOT NEEDED'

print('\n  ABLATION RESULTS:')
for a in ablation:
    d = f'{a["delta"]:+.4f}' if a['delta'] != 0 else '  —  '
    print(f'    {a["config"]:<25s} AUC={a["auc"]:.4f} Δ={d} {a["verdict"]}')
results['ablation'] = ablation

# ═══════════════════════════════════════════════════════════════
# EXP 5.2: STACKING
# ═══════════════════════════════════════════════════════════════
print('\n[4/6] STACKING META-LEARNER')
stack_val = np.column_stack([gin['val_preds'], cb_val_preds])
stack_test = np.column_stack([gin['test_preds'], cb_test_preds])

lr_stack = LogisticRegression(random_state=42, max_iter=1000)
lr_stack.fit(stack_val, y_va)
lr_preds = lr_stack.predict_proba(stack_test)[:, 1]
lr_auc = roc_auc_score(y_te, lr_preds)
lr_auprc = average_precision_score(y_te, lr_preds)
print(f'  LogReg stacking: AUC={lr_auc:.4f} AUPRC={lr_auprc:.4f}')
print(f'    Learned: GIN coef={lr_stack.coef_[0][0]:.3f}, CB coef={lr_stack.coef_[0][1]:.3f}')

results['stacking'] = {'logreg': {'auc': lr_auc, 'auprc': lr_auprc,
                                   'gin_coef': float(lr_stack.coef_[0][0]),
                                   'cb_coef': float(lr_stack.coef_[0][1])},
                        'weighted_avg': {'auc': ens_auc, 'auprc': ens_auprc}}

# ═══════════════════════════════════════════════════════════════
# EXP 5.3: FOCAL LOSS
# ═══════════════════════════════════════════════════════════════
print('\n[5/6] FOCAL LOSS GIN (γ=2, α=0.75)')
t0 = time.time()
fl = train_gnn(GINEdge, use_focal=True, focal_gamma=2.0, focal_alpha=0.75)
fl_ens = 0.3 * fl['test_preds'] + 0.7 * cb_test_preds
fl_ens_auc = roc_auc_score(y_te, fl_ens)
fl_ens_auprc = average_precision_score(y_te, fl_ens)
print(f'  Focal GIN: test={fl["test_auc"]:.4f} (Δ={fl["test_auc"]-gin["test_auc"]:+.4f} vs BCE)')
print(f'  Focal Ensemble: {fl_ens_auc:.4f} (Δ={fl_ens_auc-ens_auc:+.4f}) [{time.time()-t0:.0f}s]')

results['focal_loss'] = {'gamma': 2.0, 'alpha': 0.75, 'gin_auc': fl['test_auc'],
                          'ens_auc': fl_ens_auc, 'ens_auprc': fl_ens_auprc,
                          'delta_gin': fl['test_auc'] - gin['test_auc'],
                          'delta_ens': fl_ens_auc - ens_auc}

# ═══════════════════════════════════════════════════════════════
# ERROR ANALYSIS
# ═══════════════════════════════════════════════════════════════
print('\n  ERROR ANALYSIS:')
gin_cls = (gin['test_preds'] > 0.5).astype(int)
cb_cls = (cb_test_preds > 0.5).astype(int)
both_wrong = (gin_cls != y_te) & (cb_cls != y_te)
gin_only = (gin_cls != y_te) & (cb_cls == y_te)
cb_only = (gin_cls == y_te) & (cb_cls != y_te)
both_right = (gin_cls == y_te) & (cb_cls == y_te)
n_bw, n_gow, n_cow, n_br = both_wrong.sum(), gin_only.sum(), cb_only.sum(), both_right.sum()
jaccard = n_bw / (n_bw + n_gow + n_cow) if (n_bw + n_gow + n_cow) > 0 else 0

print(f'    Both correct: {n_br} ({n_br/len(y_te)*100:.1f}%)')
print(f'    Both wrong:   {n_bw} ({n_bw/len(y_te)*100:.1f}%)')
print(f'    GIN only:     {n_gow} ({n_gow/len(y_te)*100:.1f}%)')
print(f'    CB only:      {n_cow} ({n_cow/len(y_te)*100:.1f}%)')
print(f'    Jaccard: {jaccard:.3f}')

# Hard molecule profiling
lip_features = ['mol_weight', 'logp', 'hbd', 'hba', 'tpsa', 'rotatable_bonds', 'aromatic_rings']
test_lip = feat_df[feat_df['split'] == 'test'][lip_features].values.astype(np.float32)
print(f'\n    {"Feature":<20s} {"Hard":<10s} {"Easy":<10s} {"Ratio":<8s}')
for j, feat in enumerate(lip_features):
    hm = test_lip[both_wrong, j].mean() if n_bw > 0 else 0
    em = test_lip[both_right, j].mean() if n_br > 0 else 0
    print(f'    {feat:<20s} {hm:<10.1f} {em:<10.1f} {hm/em if em else 0:<8.2f}')

hard_pos = y_te[both_wrong].mean() if n_bw > 0 else 0
easy_pos = y_te[both_right].mean() if n_br > 0 else 0
print(f'\n    Positive rate: Hard={hard_pos:.3f} vs Easy={easy_pos:.3f}')

results['error_analysis'] = {'both_correct': int(n_br), 'both_wrong': int(n_bw),
                              'gin_only': int(n_gow), 'cb_only': int(n_cow),
                              'jaccard': float(jaccard)}

# ═══════════════════════════════════════════════════════════════
# MASTER COMPARISON + PLOTS
# ═══════════════════════════════════════════════════════════════
print('\n[6/6] MASTER COMPARISON')
master = [
    {'phase': 1, 'model': 'RF Combined', 'auc': 0.7707, 'src': 'A'},
    {'phase': 1, 'model': 'CatBoost auto_wt', 'auc': 0.7782, 'src': 'M'},
    {'phase': 2, 'model': 'GIN (no edge)', 'auc': 0.7053, 'src': 'A'},
    {'phase': 2, 'model': 'MLP-Domain9', 'auc': 0.7670, 'src': 'M'},
    {'phase': 3, 'model': 'GIN+Edge', 'auc': 0.7860, 'src': 'A'},
    {'phase': 3, 'model': 'CatBoost MI-400', 'auc': 0.8105, 'src': 'M'},
    {'phase': 4, 'model': 'GIN+Edge tuned', 'auc': 0.7982, 'src': 'A'},
    {'phase': 4, 'model': 'GIN+CB Ensemble', 'auc': 0.8114, 'src': 'A'},
    {'phase': 5, 'model': 'Weighted Avg (this run)', 'auc': ens_auc, 'src': 'A'},
    {'phase': 5, 'model': 'LogReg Stacking', 'auc': lr_auc, 'src': 'A'},
    {'phase': 5, 'model': 'Focal GIN+CB', 'auc': fl_ens_auc, 'src': 'A'},
]
master_df = pd.DataFrame(master).sort_values('auc', ascending=False)
print(f'\n  {"Rk":<4s} {"Ph":<4s} {"Model":<28s} {"AUC":<8s} {"Src":<4s}')
print('  ' + '-' * 48)
for rk, (_, r) in enumerate(master_df.iterrows(), 1):
    print(f'  {rk:<4d} P{r["phase"]:<3d} {r["model"]:<28s} {r["auc"]:.4f}  {r["src"]}')

results['master'] = master

# Plots
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('Phase 5: Advanced Techniques — ogbg-molhiv', fontsize=13, fontweight='bold')

ax = axes[0, 0]
abl_names = [a['config'] for a in ablation]
abl_aucs = [a['auc'] for a in ablation]
colors = ['#2ecc71' if a['delta'] == 0 else '#e74c3c' if a['delta'] < -0.005 else '#f39c12' for a in ablation]
ax.barh(range(len(ablation)), abl_aucs, color=colors)
ax.set_yticks(range(len(ablation))); ax.set_yticklabels(abl_names, fontsize=7)
ax.set_xlabel('Test AUC'); ax.set_title('Ablation Study')
ax.axvline(x=ens_auc, color='green', linestyle='--', alpha=0.7)
for i, v in enumerate(abl_aucs):
    ax.text(v + 0.001, i, f'{v:.4f}', va='center', fontsize=7)

ax = axes[0, 1]
stack_names = ['Wtd Avg\n(0.3/0.7)', 'LogReg\nStacking', 'Focal\nGIN+CB']
stack_aucs = [ens_auc, lr_auc, fl_ens_auc]
ax.bar(stack_names, stack_aucs, color=['#3498db', '#2ecc71', '#e67e22'])
ax.set_ylabel('Test AUC'); ax.set_title('Ensemble Methods')
for i, v in enumerate(stack_aucs):
    ax.text(i, v + 0.002, f'{v:.4f}', ha='center', fontsize=8)

ax = axes[1, 0]
err_labels = ['Both\nCorrect', 'Both\nWrong', 'GIN\nOnly', 'CB\nOnly']
err_sizes = [n_br, n_bw, n_gow, n_cow]
err_colors = ['#2ecc71', '#e74c3c', '#f39c12', '#3498db']
ax.pie(err_sizes, labels=err_labels, colors=err_colors, autopct='%1.1f%%', startangle=90)
ax.set_title(f'Error Overlap (Jaccard={jaccard:.3f})')

ax = axes[1, 1]
top = master_df.head(8)
m_names = [f'P{int(r["phase"])} {r["model"][:25]}' for _, r in top.iterrows()]
m_aucs = top['auc'].values
ax.barh(range(len(m_names)), m_aucs, color='#3498db')
ax.set_yticks(range(len(m_names))); ax.set_yticklabels(m_names, fontsize=7)
ax.set_xlabel('Test AUC'); ax.set_title('Master Leaderboard'); ax.invert_yaxis()
for i, v in enumerate(m_aucs):
    ax.text(v + 0.001, i, f'{v:.4f}', va='center', fontsize=7)

plt.tight_layout()
plt.savefig(RESULTS_DIR / 'phase5_advanced_techniques.png', dpi=150, bbox_inches='tight')
plt.close()
print('\n  Saved: results/phase5_advanced_techniques.png')

with open(RESULTS_DIR / 'phase5_anthony_results.json', 'w') as f:
    json.dump(results, f, indent=2, default=str)
print('  Saved: results/phase5_anthony_results.json')

best_p5 = max(ens_auc, lr_auc, fl_ens_auc)
print(f'\n  PHASE 5 CHAMPION: {best_p5:.4f} AUC')
print(f'  OGB GIN-VN baseline: 0.7707 — we beat it by +{best_p5-0.7707:.4f}')
print('\nDone.')
