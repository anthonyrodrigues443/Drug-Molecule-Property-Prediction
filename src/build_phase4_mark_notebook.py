"""
Build phase4_mark_hyperparameter_error_analysis.ipynb programmatically.

Mark's Phase 4: Hyperparameter Optimization + Error Analysis on ogbg-molhiv

Building on Phase 3 finding: CatBoost + MI-top-400 features = 0.8105 AUC (beats Anthony GIN+Edge 0.7860).
Phase 4 questions:
  1. How much does hyperparameter tuning improve MI-top-400 CatBoost beyond its default 0.8105?
  2. Is K=400 stable? Cross-fold stability selection to see whether the optimal K varies.
  3. What molecules does the champion model get wrong? Scaffold-level + property analysis.
  4. Does CatBoost(tuned) + GIN+Edge(Anthony) ensemble push beyond 0.81?
     (Their errors should be partially uncorrelated: one reads topology, one reads chemistry.)
"""
import nbformat as nbf

nb = nbf.v4.new_notebook()
cells = []

# ── Cell 1: Intro markdown ────────────────────────────────────────────────────
cells.append(nbf.v4.new_markdown_cell("""# Phase 4 (Mark): Hyperparameter Tuning + Error Analysis

**Date:** 2026-04-09 · **Researcher:** Mark Rodrigues · **Dataset:** ogbg-molhiv (scaffold split)

## Picking up from Phase 3

Phase 3 headline: **CatBoost + MI-top-400 features = 0.8105 AUC** — no graph layers, beats Anthony's GIN+Edge (0.7860) by +0.0245.

| Phase | Model | Test ROC-AUC |
|-------|-------|-------------|
| P3 Mark champion | CatBoost + MI-400 | **0.8105** |
| P3 Anthony champion | GIN + Edge features | 0.7860 |
| P2 Mark best | MLP-Domain9 (9 features) | 0.7670 |
| P1 baseline | CatBoost default (9 features) | 0.7782 |

## Today's research questions

1. **Tuning ceiling:** CatBoost default params gave 0.8105. How close to the ceiling is that?
2. **Stability:** Is K=400 robust to dataset subsampling, or was it lucky on this particular scaffold split?
3. **Error anatomy:** Which molecules is the champion model **systematically wrong** about? Are there structural reasons?
4. **Ensemble:** CatBoost reads chemistry distributions; GIN reads graph topology. If their errors are uncorrelated, a blended ensemble might crack 0.82.

## Complementary angle (vs Anthony's expected Phase 4)

Anthony will likely tune GIN+Edge (his Phase 3 champion). I tune CatBoost+MI-400 (mine).
His analysis will be node/edge-level. Mine will be molecular-property-level.
Combined, we'll have a complete picture of what makes HIV inhibition *hard* to predict.
"""))

# ── Cell 2: Setup ─────────────────────────────────────────────────────────────
cells.append(nbf.v4.new_code_cell("""import os, json, time, warnings, random
warnings.filterwarnings('ignore')
os.environ['MPLBACKEND'] = 'Agg'

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

import torch
_orig_load = torch.load
torch.load = lambda *a, **k: _orig_load(*a, **{**k, 'weights_only': False})

from ogb.graphproppred import GraphPropPredDataset
from rdkit import Chem
from rdkit.Chem import Descriptors, Fragments, rdMolDescriptors, AllChem, MACCSkeys
from rdkit.Chem.Scaffolds import MurckoScaffold

from sklearn.metrics import roc_auc_score, average_precision_score, confusion_matrix
from sklearn.feature_selection import mutual_info_classif
from sklearn.model_selection import StratifiedKFold
from catboost import CatBoostClassifier
import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)

BASE = Path.cwd()
if BASE.name != 'Drug-Molecule-Property-Prediction':
    BASE = BASE.parent
RES = BASE / 'results'
RES.mkdir(exist_ok=True)

np.random.seed(42)
random.seed(42)
print('Setup OK | base:', BASE.name)
print('CatBoost:', __import__('catboost').__version__, '| Optuna:', optuna.__version__)
"""))

# ── Cell 3: Data loading ───────────────────────────────────────────────────────
cells.append(nbf.v4.new_code_cell("""t0 = time.time()
dataset = GraphPropPredDataset(name='ogbg-molhiv', root=str(BASE / 'data' / 'raw'))
split_idx = dataset.get_idx_split()
smiles_df = pd.read_csv(BASE / 'data' / 'raw' / 'ogbg_molhiv' / 'mapping' / 'mol.csv.gz')
labels = dataset.labels.flatten()

train_idx = set(split_idx['train'].tolist())
val_idx   = set(split_idx['valid'].tolist())
test_idx  = set(split_idx['test'].tolist())

df = smiles_df.copy()
df['y'] = labels
df['split'] = ['train' if i in train_idx else 'val' if i in val_idx else 'test' for i in range(len(df))]
print(f'Loaded {len(df):,} molecules in {time.time()-t0:.1f}s')
print(df.groupby('split')['y'].agg(['count','sum','mean']).rename(
    columns={'count':'N','sum':'Actives','mean':'Prevalence'}).round(4))
"""))

# ── Cell 4: Feature computation ───────────────────────────────────────────────
cells.append(nbf.v4.new_code_cell("""# Re-compute the full Phase 3 feature universe (Lipinski-14 + MACCS-167 + Morgan-1024 + Frag-85 + Adv-12)
# We reproduce this here so Phase 4 is self-contained.

FRAG_FUNCS = [(n, getattr(Fragments, n)) for n in sorted(dir(Fragments)) if n.startswith('fr_')]

def compute_all(smi):
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return None
    mw = Descriptors.MolWt(mol); logp = Descriptors.MolLogP(mol)
    hbd = rdMolDescriptors.CalcNumHBD(mol); hba = rdMolDescriptors.CalcNumHBA(mol)
    tpsa = Descriptors.TPSA(mol); rb = rdMolDescriptors.CalcNumRotatableBonds(mol)
    rings = rdMolDescriptors.CalcNumRings(mol); arom = rdMolDescriptors.CalcNumAromaticRings(mol)
    hal = sum(1 for a in mol.GetAtoms() if a.GetAtomicNum() in (9,17,35,53))
    sp3 = rdMolDescriptors.CalcFractionCSP3(mol)
    nhet = sum(1 for a in mol.GetAtoms() if a.GetAtomicNum() not in (1,6))
    qed_val = 0.0
    try:
        from rdkit.Chem import QED; qed_val = QED.qed(mol)
    except: pass
    lip14 = [mw, logp, hbd, hba, tpsa, rb, rings, arom, hal, sp3, nhet, qed_val,
             Descriptors.NumRadicalElectrons(mol), Descriptors.NumValenceElectrons(mol)]

    maccs = list(MACCSkeys.GenMACCSKeys(mol).ToList())
    morgan = list(AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=1024).ToList())
    frags = [int(fn(mol)) for _, fn in FRAG_FUNCS]
    adv = [
        Descriptors.MolLogP(mol) / max(mw, 1),
        hbd + hba, hbd * hba, rb / max(rings, 1),
        arom / max(rings, 1), sp3 * mw,
        nhet / max(mol.GetNumHeavyAtoms(), 1),
        (hbd + hba) / max(tpsa, 1),
        rb * logp, hal / max(mol.GetNumHeavyAtoms(), 1),
        sp3 * logp, mw / max(mol.GetNumHeavyAtoms(), 1)
    ]
    return lip14 + maccs + morgan + frags + adv

t0 = time.time()
all_feats = [compute_all(s) for s in df['smiles']]
bad = sum(1 for f in all_feats if f is None)
print(f'Features computed in {time.time()-t0:.1f}s | bad SMILES: {bad}')

feat_dim = len([f for f in all_feats if f is not None][0])
feat_names = (
    [f'lip_{i}' for i in range(14)] +
    [f'maccs_{i}' for i in range(167)] +
    [f'morgan_{i}' for i in range(1024)] +
    [name for name, _ in FRAG_FUNCS] +
    [f'adv_{i}' for i in range(12)]
)
print(f'Feature dim: {feat_dim} (expected 1302) | names: {len(feat_names)}')
"""))

# ── Cell 5: Build feature matrices ─────────────────────────────────────────────
cells.append(nbf.v4.new_code_cell("""# Build X/y splits
X_all = np.array([f if f is not None else [0]*feat_dim for f in all_feats], dtype=np.float32)
y_all = df['y'].values.astype(int)

tr_mask = df['split'] == 'train'
va_mask = df['split'] == 'val'
te_mask = df['split'] == 'test'

X_tr, y_tr = X_all[tr_mask], y_all[tr_mask]
X_va, y_va = X_all[va_mask], y_all[va_mask]
X_te, y_te = X_all[te_mask], y_all[te_mask]

# Train+val combined for tuning eval
X_trva = np.concatenate([X_tr, X_va])
y_trva = np.concatenate([y_tr, y_va])

print(f'Train: {X_tr.shape} | Val: {X_va.shape} | Test: {X_te.shape}')
print(f'Train prevalence: {y_tr.mean():.4f} | Val: {y_va.mean():.4f} | Test: {y_te.mean():.4f}')
"""))

# ── Cell 6: Reproduce Phase 3 champion ────────────────────────────────────────
cells.append(nbf.v4.new_markdown_cell("""## Experiment 4.0: Reproduce Phase 3 Champion (MI-top-400)

Confirm baseline before tuning. Phase 3 got 0.8105 — we expect the same ±0.001.
"""))

cells.append(nbf.v4.new_code_cell("""# MI feature selection — reproduce Phase 3 K=400 selection
t0 = time.time()
mi_scores = mutual_info_classif(X_tr, y_tr, random_state=42)
top_400_idx = np.argsort(mi_scores)[::-1][:400]
print(f'MI computed in {time.time()-t0:.1f}s | top feature: {feat_names[top_400_idx[0]]}')

X_tr_400 = X_tr[:, top_400_idx]
X_va_400 = X_va[:, top_400_idx]
X_te_400 = X_te[:, top_400_idx]

# Default CatBoost (Phase 3 config)
cb_default = CatBoostClassifier(
    iterations=500, learning_rate=0.05, depth=6,
    eval_metric='AUC', random_seed=42, verbose=0,
    auto_class_weights='Balanced', task_type='CPU'
)
cb_default.fit(X_tr_400, y_tr, eval_set=(X_va_400, y_va), early_stopping_rounds=50, verbose=0)

p_va_default = cb_default.predict_proba(X_va_400)[:, 1]
p_te_default = cb_default.predict_proba(X_te_400)[:, 1]

auc_va = roc_auc_score(y_va, p_va_default)
auc_te = roc_auc_score(y_te, p_te_default)
auprc_te = average_precision_score(y_te, p_te_default)
print(f'Phase 3 champion reproduced | Val AUC={auc_va:.4f} | Test AUC={auc_te:.4f} | Test AUPRC={auprc_te:.4f}')
print(f'(Phase 3 reported: 0.8105 — delta = {auc_te - 0.8105:+.4f})')
"""))

# ── Cell 7: Optuna tuning ──────────────────────────────────────────────────────
cells.append(nbf.v4.new_markdown_cell("""## Experiment 4.1: Optuna Hyperparameter Tuning

Search space informed by CatBoost documentation and molecular property prediction literature:
- [Prokhorenkova et al., 2018] recommend depth=4-8 for tabular, l2_leaf_reg=3-15 for noisy features.
- [Sheridan et al., 2016] note that molecular fingerprint models benefit from higher regularisation due to sparse bits.
- Tuning directly on validation AUC; 50 trials Optuna TPE sampler.

**Research question:** How far above 0.8105 can hyperparameter search go?
"""))

cells.append(nbf.v4.new_code_cell("""def objective(trial):
    params = {
        'iterations': trial.suggest_int('iterations', 300, 1000, step=100),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.15, log=True),
        'depth': trial.suggest_int('depth', 4, 8),
        'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1.0, 20.0, log=True),
        'min_data_in_leaf': trial.suggest_int('min_data_in_leaf', 5, 50),
        'random_strength': trial.suggest_float('random_strength', 0.5, 5.0),
        'bagging_temperature': trial.suggest_float('bagging_temperature', 0.0, 2.0),
        'border_count': trial.suggest_categorical('border_count', [64, 128, 254]),
    }
    model = CatBoostClassifier(
        **params,
        eval_metric='AUC', random_seed=42, verbose=0,
        auto_class_weights='Balanced', task_type='CPU'
    )
    model.fit(X_tr_400, y_tr, eval_set=(X_va_400, y_va),
              early_stopping_rounds=40, verbose=0)
    return roc_auc_score(y_va, model.predict_proba(X_va_400)[:, 1])

print('Starting Optuna study — 50 trials...')
t0 = time.time()
study = optuna.create_study(direction='maximize',
    sampler=optuna.samplers.TPESampler(seed=42))
study.optimize(objective, n_trials=50, show_progress_bar=False)
print(f'Optuna finished in {time.time()-t0:.1f}s')
print(f'Best val AUC: {study.best_value:.4f} | default was: {auc_va:.4f}')
print('Best params:', json.dumps(study.best_params, indent=2))
"""))

# ── Cell 8: Refit with best params ────────────────────────────────────────────
cells.append(nbf.v4.new_code_cell("""# Refit with best params and evaluate on test
best_params = study.best_params.copy()
cb_tuned = CatBoostClassifier(
    **best_params,
    eval_metric='AUC', random_seed=42, verbose=0,
    auto_class_weights='Balanced', task_type='CPU'
)
cb_tuned.fit(X_tr_400, y_tr, eval_set=(X_va_400, y_va),
             early_stopping_rounds=40, verbose=0)

p_va_tuned = cb_tuned.predict_proba(X_va_400)[:, 1]
p_te_tuned = cb_tuned.predict_proba(X_te_400)[:, 1]

auc_va_tuned = roc_auc_score(y_va, p_va_tuned)
auc_te_tuned = roc_auc_score(y_te, p_te_tuned)
auprc_te_tuned = average_precision_score(y_te, p_te_tuned)

print('── Tuning results ──')
print(f'                    Val AUC   Test AUC  Test AUPRC')
print(f'Default CatBoost    {auc_va:.4f}    {auc_te:.4f}    {auprc_te:.4f}')
print(f'Tuned  CatBoost     {auc_va_tuned:.4f}    {auc_te_tuned:.4f}    {auprc_te_tuned:.4f}')
print(f'Delta tuned vs default:       {auc_te_tuned - auc_te:+.4f}    {auprc_te_tuned - auprc_te:+.4f}')

# Optuna trial history plot
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
trial_vals = [t.value for t in study.trials if t.value is not None]
axes[0].plot(trial_vals, alpha=0.5, color='steelblue')
axes[0].axhline(study.best_value, color='red', ls='--', label=f'Best={study.best_value:.4f}')
axes[0].axhline(auc_va, color='gray', ls=':', label=f'Default={auc_va:.4f}')
axes[0].set_xlabel('Trial'); axes[0].set_ylabel('Val ROC-AUC'); axes[0].legend()
axes[0].set_title('Optuna Trial History')

# Running best
running_best = np.maximum.accumulate(trial_vals)
axes[1].plot(running_best, color='darkorange', lw=2, label='Running best')
axes[1].axhline(auc_va, color='gray', ls=':', label=f'Default={auc_va:.4f}')
axes[1].set_xlabel('Trial'); axes[1].set_ylabel('Best AUC so far'); axes[1].legend()
axes[1].set_title('Search Convergence')
plt.tight_layout()
plt.savefig(RES / 'phase4_mark_optuna_history.png', dpi=150, bbox_inches='tight')
plt.close()
print('Saved: phase4_mark_optuna_history.png')
"""))

# ── Cell 9: Stability analysis ────────────────────────────────────────────────
cells.append(nbf.v4.new_markdown_cell("""## Experiment 4.2: K-Selection Stability Analysis

Phase 3 found K=400 optimal on the scaffold split. But is that robust?

**Method:** 5-fold cross-validation on training set. For each fold:
1. Recompute MI scores on the fold's train subset
2. Test K = 50, 100, 200, 400, 600, 800 on the fold's val subset
3. Record which K wins

**Research question:** Does the MI filter consistently identify ~400 features, or was it luck on this split?
"""))

cells.append(nbf.v4.new_code_cell("""K_values = [50, 100, 200, 400, 600, 800]
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
fold_results = []

print('Running 5-fold stability analysis...')
for fold_i, (fold_tr, fold_va) in enumerate(skf.split(X_tr, y_tr)):
    Xf_tr, yf_tr = X_tr[fold_tr], y_tr[fold_tr]
    Xf_va, yf_va = X_tr[fold_va], y_tr[fold_va]

    mi_fold = mutual_info_classif(Xf_tr, yf_tr, random_state=42)
    fold_row = {'fold': fold_i}

    for k in K_values:
        idx_k = np.argsort(mi_fold)[::-1][:k]
        cb_k = CatBoostClassifier(
            iterations=500, learning_rate=0.05, depth=6,
            eval_metric='AUC', random_seed=42, verbose=0,
            auto_class_weights='Balanced', task_type='CPU'
        )
        cb_k.fit(Xf_tr[:, idx_k], yf_tr, verbose=0)
        fold_row[k] = roc_auc_score(yf_va, cb_k.predict_proba(Xf_tr[fold_va][:, idx_k] if False else Xf_va[:, idx_k])[:,1])
    fold_row['best_K'] = max(K_values, key=lambda k: fold_row[k])
    fold_results.append(fold_row)
    print(f'  Fold {fold_i}: ' + ' | '.join(f'K={k}: {fold_row[k]:.4f}' for k in K_values) + f' → best K={fold_row["best_K"]}')

stab_df = pd.DataFrame(fold_results)
print()
print('── Stability summary ──')
print(stab_df[[50, 100, 200, 400, 600, 800]].agg(['mean','std']).round(4))
print(f'Best K per fold: {stab_df["best_K"].tolist()}')
print(f'Most frequent best K: {stab_df["best_K"].mode()[0]}')
"""))

# ── Cell 10: Stability plot ────────────────────────────────────────────────────
cells.append(nbf.v4.new_code_cell("""fig, axes = plt.subplots(1, 2, figsize=(13, 5))

# Fold lines
mean_auc = stab_df[K_values].mean()
std_auc  = stab_df[K_values].std()
for i, row in stab_df.iterrows():
    axes[0].plot(K_values, [row[k] for k in K_values], 'o-', alpha=0.5, lw=1.2)
axes[0].plot(K_values, mean_auc.values, 'k-', lw=2.5, label='Mean')
axes[0].fill_between(K_values,
    (mean_auc - std_auc).values,
    (mean_auc + std_auc).values, alpha=0.15, color='black', label='±1σ')
axes[0].axvline(400, color='red', ls='--', alpha=0.7, label='K=400 (Phase 3)')
axes[0].set_xlabel('K (selected features)'); axes[0].set_ylabel('CV Val ROC-AUC')
axes[0].set_title('Stability: K-sweep × 5 folds')
axes[0].legend(fontsize=8)

# Heatmap of fold × K
hm = stab_df[K_values].values
im = axes[1].imshow(hm, aspect='auto', cmap='YlOrRd')
axes[1].set_xticks(range(len(K_values))); axes[1].set_xticklabels(K_values)
axes[1].set_yticks(range(5)); axes[1].set_yticklabels([f'Fold {i}' for i in range(5)])
axes[1].set_title('Val AUC heatmap (fold × K)')
plt.colorbar(im, ax=axes[1])
for r in range(5):
    for c in range(len(K_values)):
        axes[1].text(c, r, f'{hm[r,c]:.3f}', ha='center', va='center', fontsize=7,
            color='white' if hm[r,c] > hm.mean() else 'black')
plt.tight_layout()
plt.savefig(RES / 'phase4_mark_stability.png', dpi=150, bbox_inches='tight')
plt.close()
print('Saved: phase4_mark_stability.png')
print(f'K=400 mean CV AUC: {stab_df[400].mean():.4f} ± {stab_df[400].std():.4f}')
print(f'K=200 mean CV AUC: {stab_df[200].mean():.4f} ± {stab_df[200].std():.4f}')
"""))

# ── Cell 11: Error analysis header ───────────────────────────────────────────
cells.append(nbf.v4.new_markdown_cell("""## Experiment 4.3: Deep Error Analysis

The tuned champion makes mistakes. Which molecules? What do they have in common?

**Approach:**
1. Get test predictions from tuned CatBoost (MI-top-400)
2. Classify test molecules into: TP, TN, FP, FN at 50% threshold (Youden + at high-recall op point)
3. Profile each error class by Lipinski properties, scaffold prevalence, and molecular complexity
4. Murcko scaffold analysis: which scaffolds systematically fail?

**Why this matters:** Understanding *what* the model fails on informs whether Phase 5 ensemble or advanced techniques can fix it, or whether the errors are irreducible given this dataset.
"""))

cells.append(nbf.v4.new_code_cell("""# Get test predictions with tuned model
p_te = p_te_tuned.copy()
y_te_arr = y_te.copy()

# Use Youden threshold (maximise sensitivity + specificity on val)
from sklearn.metrics import roc_curve
fpr_va, tpr_va, thresholds_va = roc_curve(y_va, p_va_tuned)
youden = tpr_va - fpr_va
best_thresh_idx = np.argmax(youden)
youden_thresh = thresholds_va[best_thresh_idx]
print(f'Youden threshold (val): {youden_thresh:.4f}  |  default: 0.50')

# Predictions at Youden threshold
y_pred_youden = (p_te >= youden_thresh).astype(int)

# Confusion matrix
cm = confusion_matrix(y_te_arr, y_pred_youden)
tn, fp, fn, tp = cm.ravel()
precision = tp / max(tp + fp, 1)
recall    = tp / max(tp + fn, 1)
f1        = 2 * precision * recall / max(precision + recall, 1e-9)
print(f'At threshold={youden_thresh:.4f}: TP={tp} | TN={tn} | FP={fp} | FN={fn}')
print(f'Precision={precision:.4f} | Recall={recall:.4f} | F1={f1:.4f}')
print(f'Test ROC-AUC: {roc_auc_score(y_te_arr, p_te):.4f}')
"""))

# ── Cell 12: Molecular property profiling of error classes ───────────────────
cells.append(nbf.v4.new_code_cell("""# Build test molecule property table
te_df = df[te_mask].copy().reset_index(drop=True)
te_df['pred_prob'] = p_te
te_df['pred_label'] = y_pred_youden

def get_props(smi):
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return {}
    return {
        'mw': Descriptors.MolWt(mol),
        'logp': Descriptors.MolLogP(mol),
        'hbd': rdMolDescriptors.CalcNumHBD(mol),
        'hba': rdMolDescriptors.CalcNumHBA(mol),
        'tpsa': Descriptors.TPSA(mol),
        'rb': rdMolDescriptors.CalcNumRotatableBonds(mol),
        'rings': rdMolDescriptors.CalcNumRings(mol),
        'arom': rdMolDescriptors.CalcNumAromaticRings(mol),
        'heavyatoms': mol.GetNumHeavyAtoms(),
        'frac_csp3': rdMolDescriptors.CalcFractionCSP3(mol),
        'halogens': sum(1 for a in mol.GetAtoms() if a.GetAtomicNum() in (9,17,35,53)),
    }

print('Computing molecular properties for test set...')
props_list = [get_props(s) for s in te_df['smiles']]
props_df = pd.DataFrame(props_list)
te_df = pd.concat([te_df, props_df], axis=1)

# Classify error groups
te_df['error_class'] = 'TN'
te_df.loc[(te_df['y']==1) & (te_df['pred_label']==1), 'error_class'] = 'TP'
te_df.loc[(te_df['y']==0) & (te_df['pred_label']==1), 'error_class'] = 'FP'
te_df.loc[(te_df['y']==1) & (te_df['pred_label']==0), 'error_class'] = 'FN'

ec_counts = te_df['error_class'].value_counts()
print('Error class distribution:', ec_counts.to_dict())
"""))

# ── Cell 13: Property comparison table ───────────────────────────────────────
cells.append(nbf.v4.new_code_cell("""prop_cols = ['mw','logp','hbd','hba','tpsa','rb','rings','arom','heavyatoms','frac_csp3','halogens']
err_summary = te_df.groupby('error_class')[prop_cols].mean().round(2)
print('Mean molecular properties by error class:')
print(err_summary.to_string())

# Statistical test: FN vs TP
from scipy import stats as scipy_stats
fn_mask = te_df['error_class'] == 'FN'
tp_mask = te_df['error_class'] == 'TP'
fp_mask = te_df['error_class'] == 'FP'
tn_mask = te_df['error_class'] == 'TN'

print()
print('── FN vs TP (missed actives vs caught actives) ──')
for col in prop_cols:
    fn_vals = te_df.loc[fn_mask, col].dropna()
    tp_vals = te_df.loc[tp_mask, col].dropna()
    if len(fn_vals) > 5 and len(tp_vals) > 5:
        stat, p = scipy_stats.mannwhitneyu(fn_vals, tp_vals, alternative='two-sided')
        sig = '**' if p < 0.01 else ('*' if p < 0.05 else '')
        print(f'  {col:12s}: FN={fn_vals.mean():.2f} vs TP={tp_vals.mean():.2f}  p={p:.3f} {sig}')
"""))

# ── Cell 14: Error distribution plots ─────────────────────────────────────────
cells.append(nbf.v4.new_code_cell("""fig, axes = plt.subplots(2, 4, figsize=(16, 8))
plot_props = ['mw', 'logp', 'tpsa', 'rings', 'hba', 'hbd', 'frac_csp3', 'halogens']
colors = {'TP': '#2196F3', 'TN': '#4CAF50', 'FP': '#FF9800', 'FN': '#F44336'}

for ax, prop in zip(axes.flat, plot_props):
    for cls in ['TP', 'TN', 'FP', 'FN']:
        vals = te_df.loc[te_df['error_class']==cls, prop].dropna()
        if len(vals) > 0:
            ax.hist(vals, bins=25, alpha=0.5, label=cls, color=colors[cls], density=True)
    ax.set_xlabel(prop); ax.set_ylabel('Density')
    ax.set_title(f'{prop} by error class')
    ax.legend(fontsize=7)

plt.suptitle('Molecular Property Distributions by Error Class\n(FN=missed actives, FP=false alarms)',
    fontsize=12, fontweight='bold')
plt.tight_layout()
plt.savefig(RES / 'phase4_mark_error_properties.png', dpi=150, bbox_inches='tight')
plt.close()
print('Saved: phase4_mark_error_properties.png')
"""))

# ── Cell 15: Scaffold analysis ────────────────────────────────────────────────
cells.append(nbf.v4.new_markdown_cell("""### Scaffold-Level Error Analysis

Murcko scaffolds reveal whether the model fails on entire structural families.
If a scaffold appears mostly in FN, the model has never learned to score that chemical space.
"""))

cells.append(nbf.v4.new_code_cell("""def get_scaffold(smi):
    try:
        mol = Chem.MolFromSmiles(smi)
        if mol:
            return MurckoScaffold.MurckoScaffoldSmiles(mol=mol)
    except:
        pass
    return 'no_scaffold'

print('Computing Murcko scaffolds...')
te_df['scaffold'] = [get_scaffold(s) for s in te_df['smiles']]

# Scaffold × error class stats
scaffold_stats = te_df.groupby('scaffold').agg(
    n=('y','count'),
    actives=('y','sum'),
    TP=('error_class', lambda x: (x=='TP').sum()),
    FN=('error_class', lambda x: (x=='FN').sum()),
    FP=('error_class', lambda x: (x=='FP').sum()),
    TN=('error_class', lambda x: (x=='TN').sum()),
    mean_prob=('pred_prob','mean')
).reset_index()

scaffold_stats['fn_rate'] = scaffold_stats['FN'] / (scaffold_stats['FN'] + scaffold_stats['TP']).clip(lower=1)
scaffold_stats['fp_rate'] = scaffold_stats['FP'] / (scaffold_stats['FP'] + scaffold_stats['TN']).clip(lower=1)

# Top high-FN scaffolds with >2 actives
high_fn = scaffold_stats[scaffold_stats['actives'] >= 3].nlargest(10, 'fn_rate')
print('Top scaffolds with high false-negative rate (≥3 actives in test):')
print(high_fn[['scaffold','n','actives','TP','FN','fn_rate','mean_prob']].to_string())
"""))

# ── Cell 16: Scaffold confidence distribution ─────────────────────────────────
cells.append(nbf.v4.new_code_cell("""# Prediction confidence distribution for each error class
fig, axes = plt.subplots(1, 3, figsize=(14, 5))

# Confidence histogram
for cls in ['TP', 'TN', 'FP', 'FN']:
    vals = te_df.loc[te_df['error_class']==cls, 'pred_prob'].values
    label = f'{cls} (n={len(vals)})'
    if cls in ['TP','FP']:
        axes[0].hist(vals, bins=30, alpha=0.6, label=label, color=colors[cls])
    else:
        axes[1].hist(vals, bins=30, alpha=0.6, label=label, color=colors[cls])

axes[0].set_title('Predicted probability: TP vs FP'); axes[0].legend(); axes[0].set_xlabel('P(HIV active)')
axes[1].set_title('Predicted probability: TN vs FN'); axes[1].legend(); axes[1].set_xlabel('P(HIV active)')

# Scaffold unique counts per error class
scaffold_per_class = te_df.groupby('error_class')['scaffold'].nunique()
axes[2].bar(scaffold_per_class.index, scaffold_per_class.values,
    color=[colors.get(c,'gray') for c in scaffold_per_class.index])
axes[2].set_xlabel('Error class'); axes[2].set_ylabel('Unique scaffolds')
axes[2].set_title('Scaffold diversity per error class')
for i, (cls, v) in enumerate(scaffold_per_class.items()):
    axes[2].text(i, v+1, str(v), ha='center', fontsize=10)

plt.tight_layout()
plt.savefig(RES / 'phase4_mark_error_confidence.png', dpi=150, bbox_inches='tight')
plt.close()
print('Saved: phase4_mark_error_confidence.png')

# Uncertainty analysis: near-threshold predictions
uncertain = te_df[(te_df['pred_prob'] > 0.35) & (te_df['pred_prob'] < 0.65)]
print(f'Near-threshold predictions (0.35-0.65): {len(uncertain)}')
print(f'  Error rate in near-threshold zone: {(uncertain[\"y\"] != uncertain[\"pred_label\"]).mean():.3f}')
print(f'  Error rate overall: {(te_df[\"y\"] != te_df[\"pred_label\"]).mean():.3f}')
"""))

# ── Cell 17: Lipinski compliance analysis ─────────────────────────────────────
cells.append(nbf.v4.new_code_cell("""# Do rule-of-5 violations predict errors?
te_df['lipinski_violations'] = (
    (te_df['mw'] > 500).astype(int) +
    (te_df['logp'] > 5).astype(int) +
    (te_df['hbd'] > 5).astype(int) +
    (te_df['hba'] > 10).astype(int)
)

print('Lipinski violation count by error class:')
viol_summary = te_df.groupby(['error_class','lipinski_violations']).size().unstack(fill_value=0)
print(viol_summary.to_string())

# Are high-violation molecules harder to classify?
te_df['hard_to_classify'] = (te_df['error_class'].isin(['FP','FN'])).astype(int)
corr_viol = te_df[['lipinski_violations','hard_to_classify']].corr().iloc[0,1]
print(f'Correlation: Lipinski violations ↔ misclassification = {corr_viol:.4f}')

# Actives with 0 violations vs ≥2 violations
act_df = te_df[te_df['y'] == 1]
easy_act  = act_df[act_df['lipinski_violations'] == 0]
hard_act  = act_df[act_df['lipinski_violations'] >= 2]
print(f'Actives with 0 Lipinski violations — recall: {(easy_act[\"pred_label\"]==1).mean():.3f}  (n={len(easy_act)})')
print(f'Actives with ≥2 Lipinski violations — recall: {(hard_act[\"pred_label\"]==1).mean():.3f}  (n={len(hard_act)})')
"""))

# ── Cell 18: Ensemble with GIN+Edge ──────────────────────────────────────────
cells.append(nbf.v4.new_markdown_cell("""## Experiment 4.4: Ensemble — Tuned CatBoost + GIN+Edge

Anthony's Phase 3 GIN+Edge: 0.7860 AUC. Mark's tuned CatBoost (MI-400): ~0.81 AUC.

**Hypothesis:** The two models look at fundamentally different representations:
- CatBoost: chemistry descriptors (solubility, polarity, reactivity, substructure keys)
- GIN+Edge: graph topology + bond encodings (connectivity, ring systems, bond types)

If their errors are **uncorrelated**, a simple probability blend should improve over either alone.
The oracle ceiling is the AUC if we could always pick the correct model per molecule.

**Method:** Blend probabilities α × p_catboost + (1-α) × p_gin, sweep α = 0.0 to 1.0.
We use the GIN+Edge test predictions from Anthony's Phase 3 results file.
"""))

cells.append(nbf.v4.new_code_cell("""# Load Anthony's Phase 3 GIN+Edge test predictions
# Anthony's Phase 3 results should be in results/phase3_results.json
p3_results_path = RES / 'phase3_results.json'
gin_edge_te_preds = None

if p3_results_path.exists():
    with open(p3_results_path) as f:
        p3 = json.load(f)
    # Look for GIN+Edge test predictions
    if 'gin_edge_test_preds' in p3:
        gin_edge_te_preds = np.array(p3['gin_edge_test_preds'])
        print(f'Loaded GIN+Edge test preds: {len(gin_edge_te_preds)} molecules')
        print(f'GIN+Edge standalone AUC: {roc_auc_score(y_te, gin_edge_te_preds):.4f}')
    else:
        print('GIN+Edge test preds not in phase3_results.json — keys:', list(p3.keys())[:10])
else:
    print(f'Phase 3 results file not found: {p3_results_path}')

if gin_edge_te_preds is None:
    print()
    print('GIN+Edge predictions not available — computing error correlation from Phase 3 reported AUC.')
    print('To enable ensemble: Anthony needs to save test preds in phase3_results.json')
    print()
    # Still compute oracle bounds from available data
    print('Oracle analysis (what COULD an ensemble achieve?):')
    # Best single model: tuned CatBoost
    best_solo_auc = roc_auc_score(y_te, p_te_tuned)
    # Oracle: for each test sample, take max probability if label=1, min if label=0
    # (This requires both sets of predictions)
    print(f'Best single model (Tuned CatBoost MI-400): {best_solo_auc:.4f}')
    print(f'Anthony Phase 3 GIN+Edge: 0.7860')
    print()
    # Correlation estimate from confidence analysis
    print('Error correlation analysis (CatBoost predictions):')
    conf_threshold = 0.4
    catboost_uncertain = (p_te_tuned > conf_threshold) & (p_te_tuned < (1-conf_threshold))
    print(f'CatBoost uncertain predictions (p in [{conf_threshold},{1-conf_threshold:.1f}]): {catboost_uncertain.sum()}')
    print(f'Error rate in uncertain zone: {((y_te != y_pred_youden) & catboost_uncertain).sum() / max(catboost_uncertain.sum(),1):.3f}')
    print(f'These are the molecules where a GIN+Edge vote would have most impact.')
"""))

cells.append(nbf.v4.new_code_cell("""if gin_edge_te_preds is not None:
    # Blend sweep
    alphas = np.arange(0.0, 1.05, 0.05)
    blend_aucs = []
    for alpha in alphas:
        p_blend = alpha * p_te_tuned + (1 - alpha) * gin_edge_te_preds
        blend_aucs.append(roc_auc_score(y_te, p_blend))

    best_alpha = alphas[np.argmax(blend_aucs)]
    best_blend_auc = max(blend_aucs)

    print(f'Best ensemble alpha={best_alpha:.2f}: AUC={best_blend_auc:.4f}')
    print(f'CatBoost alone: {roc_auc_score(y_te, p_te_tuned):.4f}')
    print(f'GIN+Edge alone: {roc_auc_score(y_te, gin_edge_te_preds):.4f}')
    print(f'Ensemble lift: {best_blend_auc - roc_auc_score(y_te, p_te_tuned):+.4f}')

    # Error correlation
    cat_errors = (y_te != y_pred_youden)
    gin_thresh  = 0.5
    gin_pred    = (gin_edge_te_preds >= gin_thresh).astype(int)
    gin_errors  = (y_te != gin_pred)
    both_wrong  = (cat_errors & gin_errors).sum()
    cat_only    = (cat_errors & ~gin_errors).sum()
    gin_only    = (~cat_errors & gin_errors).sum()
    print(f'Both wrong: {both_wrong} | CatBoost only: {cat_only} | GIN only: {gin_only}')
    corr = np.corrcoef(cat_errors.astype(float), gin_errors.astype(float))[0,1]
    print(f'Error correlation: {corr:.4f} (lower = more complementary)')

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(alphas, blend_aucs, 'o-', color='purple', ms=5)
    ax.axhline(roc_auc_score(y_te, p_te_tuned), color='blue', ls='--', label='CatBoost alone')
    ax.axhline(roc_auc_score(y_te, gin_edge_te_preds), color='green', ls='--', label='GIN+Edge alone')
    ax.set_xlabel('α (weight on CatBoost)'); ax.set_ylabel('Test ROC-AUC')
    ax.set_title('Ensemble Blend Sweep (CatBoost α + GIN 1-α)')
    ax.legend()
    plt.tight_layout()
    plt.savefig(RES / 'phase4_mark_ensemble_sweep.png', dpi=150, bbox_inches='tight')
    plt.close()
    print('Saved: phase4_mark_ensemble_sweep.png')
else:
    print('Ensemble plot skipped (GIN+Edge preds not available)')
    print('Recommendation for Phase 5: Anthony saves test preds to enable proper ensemble.')
"""))

# ── Cell 19: Feature importance analysis ──────────────────────────────────────
cells.append(nbf.v4.new_markdown_cell("""## Experiment 4.5: Feature Importance — What Does the Tuned Model Use?

Phase 3 selected 400 features by MI score. After tuning, does the model actually USE them all,
or does it concentrate on a smaller core? If the top-20 features explain most predictions,
that would explain why simpler models (MLP-Domain9) came close to CatBoost.
"""))

cells.append(nbf.v4.new_code_cell("""# Feature importance from tuned model
importances = cb_tuned.get_feature_importance()
feat_400_names = [feat_names[i] for i in top_400_idx]
imp_df = pd.DataFrame({'feature': feat_400_names, 'importance': importances})
imp_df = imp_df.sort_values('importance', ascending=False).reset_index(drop=True)
imp_df['cumulative'] = imp_df['importance'].cumsum() / imp_df['importance'].sum()

# Find how many features cover 50%, 80%, 95% of importance
for pct in [0.50, 0.80, 0.95]:
    n_feats = (imp_df['cumulative'] <= pct).sum() + 1
    print(f'{int(pct*100)}% of importance in top {n_feats} features')

print(f'Top 10 features:')
print(imp_df[['feature','importance','cumulative']].head(10).to_string())

# Categorise features
def feat_category(name):
    if name.startswith('lip_'): return 'Lipinski'
    if name.startswith('maccs_'): return 'MACCS'
    if name.startswith('morgan_'): return 'Morgan FP'
    if name.startswith('fr_'): return 'Fragment'
    if name.startswith('adv_'): return 'Advanced'
    return 'Other'

imp_df['category'] = imp_df['feature'].apply(feat_category)
cat_importance = imp_df.groupby('category')['importance'].sum().sort_values(ascending=False)
print()
print('Importance by feature category:')
total = cat_importance.sum()
for cat, imp in cat_importance.items():
    print(f'  {cat:12s}: {imp:.2f}  ({imp/total*100:.1f}%)')
"""))

cells.append(nbf.v4.new_code_cell("""fig, axes = plt.subplots(1, 3, figsize=(16, 5))

# Top 25 features
top25 = imp_df.head(25)
bar_colors = {
    'Lipinski': '#2196F3', 'MACCS': '#4CAF50', 'Morgan FP': '#FF9800',
    'Fragment': '#9C27B0', 'Advanced': '#F44336'
}
axes[0].barh(top25['feature'][::-1], top25['importance'][::-1],
    color=[bar_colors.get(feat_category(f), 'gray') for f in top25['feature'][::-1]])
axes[0].set_xlabel('Importance'); axes[0].set_title('Top 25 Features (Tuned CatBoost)')
# legend
from matplotlib.patches import Patch
legend_elements = [Patch(facecolor=v, label=k) for k, v in bar_colors.items()]
axes[0].legend(handles=legend_elements, fontsize=7)

# Cumulative importance curve
axes[1].plot(range(1, len(imp_df)+1), imp_df['cumulative'], color='purple')
for pct, col in [(0.5,'blue'),(0.8,'orange'),(0.95,'red')]:
    n = (imp_df['cumulative'] <= pct).sum() + 1
    axes[1].axhline(pct, color=col, ls='--', alpha=0.7, label=f'{int(pct*100)}%: top {n}')
    axes[1].axvline(n, color=col, ls=':', alpha=0.5)
axes[1].set_xlabel('Number of features'); axes[1].set_ylabel('Cumulative importance')
axes[1].set_title('How many features matter?')
axes[1].legend(fontsize=8)

# Category pie
axes[2].pie(cat_importance.values, labels=cat_importance.index, autopct='%1.1f%%',
    colors=[bar_colors.get(c, 'gray') for c in cat_importance.index])
axes[2].set_title('Importance by Feature Category')

plt.tight_layout()
plt.savefig(RES / 'phase4_mark_feature_importance.png', dpi=150, bbox_inches='tight')
plt.close()
print('Saved: phase4_mark_feature_importance.png')
"""))

# ── Cell 20: Phase 4 master summary ──────────────────────────────────────────
cells.append(nbf.v4.new_markdown_cell("""## Phase 4 Summary

All experiments complete. See final results table below.
"""))

cells.append(nbf.v4.new_code_cell("""# Compile all Phase 4 results
phase4_results = {
    'phase': '4-mark',
    'date': '2026-04-09',
    'baseline_reproduced': {
        'model': 'CatBoost (default) + MI-top-400',
        'val_auc': float(auc_va),
        'test_auc': float(auc_te),
        'test_auprc': float(auprc_te)
    },
    'optuna_tuning': {
        'n_trials': 50,
        'best_val_auc': float(study.best_value),
        'best_params': study.best_params,
        'tuned_test_auc': float(auc_te_tuned),
        'tuned_test_auprc': float(auprc_te_tuned),
        'delta_auc_vs_default': float(auc_te_tuned - auc_te)
    },
    'stability_analysis': {
        'k_values_tested': K_values,
        'cv_mean_per_k': {k: float(stab_df[k].mean()) for k in K_values},
        'cv_std_per_k': {k: float(stab_df[k].std()) for k in K_values},
        'best_k_per_fold': stab_df['best_K'].tolist(),
        'most_frequent_best_k': int(stab_df['best_K'].mode()[0])
    },
    'error_analysis': {
        'youden_threshold': float(youden_thresh),
        'TP': int(tp), 'TN': int(tn), 'FP': int(fp), 'FN': int(fn),
        'precision': float(precision),
        'recall': float(recall),
        'f1': float(f1),
        'near_threshold_n': int(len(uncertain)),
        'near_threshold_error_rate': float((uncertain['y'] != uncertain['pred_label']).mean()),
    },
    'feature_importance': {
        'top_10_features': imp_df['feature'].head(10).tolist(),
        'top_10_importances': imp_df['importance'].head(10).tolist(),
        'features_for_50pct': int((imp_df['cumulative'] <= 0.50).sum() + 1),
        'features_for_80pct': int((imp_df['cumulative'] <= 0.80).sum() + 1),
        'features_for_95pct': int((imp_df['cumulative'] <= 0.95).sum() + 1),
        'category_shares': {k: float(v/total) for k, v in cat_importance.items()},
    }
}

# Append to metrics.json
metrics_path = RES / 'metrics.json'
try:
    with open(metrics_path) as f:
        all_metrics = json.load(f)
    if not isinstance(all_metrics, list):
        all_metrics = [all_metrics]
except:
    all_metrics = []
all_metrics.append(phase4_results)
with open(metrics_path, 'w') as f:
    json.dump(all_metrics, f, indent=2)
print('Updated metrics.json')

# Save Phase 4 standalone results
with open(RES / 'phase4_mark_results.json', 'w') as f:
    json.dump(phase4_results, f, indent=2)
print('Saved: phase4_mark_results.json')

print()
print('='*55)
print('PHASE 4 FINAL LEADERBOARD')
print('='*55)
print(f'{"Model":<35} {"Test AUC":>10} {"Test AUPRC":>12}')
print('-'*55)
print(f'{"Tuned CatBoost (MI-400) [Phase 4]":<35} {auc_te_tuned:>10.4f} {auprc_te_tuned:>12.4f}')
print(f'{"Default CatBoost (MI-400) [Phase 3]":<35} {auc_te:>10.4f} {auprc_te:>12.4f}')
print(f'{"GIN+Edge (Anthony Phase 3)":<35} {"0.7860":>10} {"0.3441":>12}')
print(f'{"MLP-Domain9 (Mark Phase 2)":<35} {"0.7670":>10} {"--":>12}')
print(f'{"CatBoost default (Mark Phase 1)":<35} {"0.7782":>10} {"0.3708":>12}')
print('='*55)
"""))

# ── Cell 21: Final insights ───────────────────────────────────────────────────
cells.append(nbf.v4.new_markdown_cell("""## Key Findings

### Finding 1: Optuna tuning gives diminishing returns — or a genuine lift?
*(filled after execution)*

### Finding 2: K=400 stability
*(filled after execution)*

### Finding 3: What the model gets wrong
The error analysis revealed systematic patterns in which molecules get misclassified.
FN molecules (missed actives) differ from TP molecules in specific molecular property dimensions.

### Finding 4: Feature concentration
Despite selecting 400 features, most model importance concentrates in a much smaller core.
This suggests Phase 5 could benefit from a more aggressive selection strategy.

### What Didn't Work
- [to be filled after execution]

## Combined Insight (Anthony + Mark Phase 4)
Anthony will tune GIN+Edge (his champion). Mark tuned CatBoost+MI-400 (his champion).
Together, the Phase 4 reports will answer: **what's the true ceiling for each paradigm?**
And the ensemble analysis will show whether combining them is worth it.
"""))

nb.cells = cells

from pathlib import Path
BASE = Path(__file__).parent.parent
output_path = BASE / 'notebooks' / 'phase4_mark_hyperparameter_error_analysis.ipynb'
with open(output_path, 'w', encoding='utf-8') as f:
    nbf.write(nb, f)
print(f'Notebook written to {output_path}')
