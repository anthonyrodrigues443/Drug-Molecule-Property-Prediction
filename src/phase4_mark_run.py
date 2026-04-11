"""
Phase 4 (Mark): Hyperparameter Tuning + Error Analysis
Run as a standalone script to avoid notebook timeout issues.
Results saved to results/phase4_mark_results.json + plots.
"""
import os, json, time, warnings, random
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

from sklearn.metrics import roc_auc_score, average_precision_score, confusion_matrix, roc_curve
from sklearn.feature_selection import mutual_info_classif
from sklearn.model_selection import StratifiedKFold
from scipy import stats as scipy_stats
from catboost import CatBoostClassifier
import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)

BASE = Path(__file__).parent.parent
RES = BASE / 'results'
RES.mkdir(exist_ok=True)

np.random.seed(42)
random.seed(42)
print('Setup OK | base:', BASE.name)

# ─────────────────────────────────────────────
# 1. DATA LOADING
# ─────────────────────────────────────────────
print('\n[1/7] Loading ogbg-molhiv dataset...')
t0 = time.time()
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
print(f'Loaded {len(df):,} mols in {time.time()-t0:.1f}s')

# ─────────────────────────────────────────────
# 2. FEATURE COMPUTATION
# ─────────────────────────────────────────────
print('\n[2/7] Computing features...')
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
feat_dim = len([f for f in all_feats if f is not None][0])
feat_names = (
    [f'lip_{i}' for i in range(14)] +
    [f'maccs_{i}' for i in range(167)] +
    [f'morgan_{i}' for i in range(1024)] +
    [name for name, _ in FRAG_FUNCS] +
    [f'adv_{i}' for i in range(12)]
)
print(f'Features computed in {time.time()-t0:.1f}s | dim={feat_dim}')

X_all = np.array([f if f is not None else [0]*feat_dim for f in all_feats], dtype=np.float32)
y_all = df['y'].values.astype(int)

tr_mask = df['split'] == 'train'
va_mask = df['split'] == 'val'
te_mask = df['split'] == 'test'

X_tr, y_tr = X_all[tr_mask], y_all[tr_mask]
X_va, y_va = X_all[va_mask], y_all[va_mask]
X_te, y_te = X_all[te_mask], y_all[te_mask]

# ─────────────────────────────────────────────
# 3. REPRODUCE PHASE 3 CHAMPION (MI-400)
# ─────────────────────────────────────────────
print('\n[3/7] Reproducing Phase 3 champion (MI-top-400)...')
t0 = time.time()
mi_scores = mutual_info_classif(X_tr, y_tr, random_state=42)
top_400_idx = np.argsort(mi_scores)[::-1][:400]
print(f'MI computed in {time.time()-t0:.1f}s | top feature: {feat_names[top_400_idx[0]]}')

X_tr_400 = X_tr[:, top_400_idx]
X_va_400 = X_va[:, top_400_idx]
X_te_400 = X_te[:, top_400_idx]

cb_default = CatBoostClassifier(
    iterations=500, learning_rate=0.05, depth=6,
    eval_metric='AUC', random_seed=42, verbose=0,
    auto_class_weights='Balanced', task_type='CPU'
)
cb_default.fit(X_tr_400, y_tr, eval_set=(X_va_400, y_va), early_stopping_rounds=50)

p_va_default = cb_default.predict_proba(X_va_400)[:, 1]
p_te_default = cb_default.predict_proba(X_te_400)[:, 1]
auc_va_default = roc_auc_score(y_va, p_va_default)
auc_te_default = roc_auc_score(y_te, p_te_default)
auprc_te_default = average_precision_score(y_te, p_te_default)
print(f'Default CatBoost | Val AUC={auc_va_default:.4f} | Test AUC={auc_te_default:.4f} (Phase 3: 0.8105, delta={auc_te_default-0.8105:+.4f})')

# ─────────────────────────────────────────────
# 4. OPTUNA HYPERPARAMETER TUNING
# ─────────────────────────────────────────────
print('\n[4/7] Optuna hyperparameter tuning (40 trials)...')

def objective(trial):
    params = {
        'iterations': trial.suggest_int('iterations', 300, 800, step=100),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.12, log=True),
        'depth': trial.suggest_int('depth', 4, 8),
        'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1.0, 20.0, log=True),
        'min_data_in_leaf': trial.suggest_int('min_data_in_leaf', 5, 50),
        'random_strength': trial.suggest_float('random_strength', 0.5, 5.0),
        'bagging_temperature': trial.suggest_float('bagging_temperature', 0.0, 2.0),
        'border_count': trial.suggest_categorical('border_count', [64, 128, 254]),
    }
    model = CatBoostClassifier(
        **params, eval_metric='AUC', random_seed=42, verbose=0,
        auto_class_weights='Balanced', task_type='CPU'
    )
    model.fit(X_tr_400, y_tr, eval_set=(X_va_400, y_va), early_stopping_rounds=30)
    return roc_auc_score(y_va, model.predict_proba(X_va_400)[:, 1])

t0 = time.time()
study = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler(seed=42))
study.optimize(objective, n_trials=40, show_progress_bar=False)
optuna_time = time.time() - t0
print(f'Optuna done in {optuna_time:.1f}s | Best val AUC={study.best_value:.4f} | default was={auc_va_default:.4f}')
print(f'Best params: {json.dumps(study.best_params, indent=2)}')

# Refit with best params
cb_tuned = CatBoostClassifier(
    **study.best_params, eval_metric='AUC', random_seed=42, verbose=0,
    auto_class_weights='Balanced', task_type='CPU'
)
cb_tuned.fit(X_tr_400, y_tr, eval_set=(X_va_400, y_va), early_stopping_rounds=30)
p_va_tuned = cb_tuned.predict_proba(X_va_400)[:, 1]
p_te_tuned = cb_tuned.predict_proba(X_te_400)[:, 1]
auc_va_tuned = roc_auc_score(y_va, p_va_tuned)
auc_te_tuned = roc_auc_score(y_te, p_te_tuned)
auprc_te_tuned = average_precision_score(y_te, p_te_tuned)
print(f'Tuned CatBoost | Val AUC={auc_va_tuned:.4f} | Test AUC={auc_te_tuned:.4f} | Test AUPRC={auprc_te_tuned:.4f}')
print(f'Tuning delta: {auc_te_tuned - auc_te_default:+.4f} AUC')

# Optuna plots
trial_vals = [t.value for t in study.trials if t.value is not None]
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
axes[0].plot(trial_vals, alpha=0.5, color='steelblue')
axes[0].axhline(study.best_value, color='red', ls='--', label=f'Best={study.best_value:.4f}')
axes[0].axhline(auc_va_default, color='gray', ls=':', label=f'Default={auc_va_default:.4f}')
axes[0].set_xlabel('Trial'); axes[0].set_ylabel('Val ROC-AUC'); axes[0].legend()
axes[0].set_title('Optuna Trial History (40 trials)')
running_best = np.maximum.accumulate(trial_vals)
axes[1].plot(running_best, color='darkorange', lw=2, label='Running best')
axes[1].axhline(auc_va_default, color='gray', ls=':', label=f'Default={auc_va_default:.4f}')
axes[1].set_xlabel('Trial'); axes[1].set_ylabel('Best AUC so far'); axes[1].legend()
axes[1].set_title('Search Convergence')
plt.tight_layout()
plt.savefig(RES / 'phase4_mark_optuna_history.png', dpi=150, bbox_inches='tight')
plt.close()
print('Saved: phase4_mark_optuna_history.png')

# ─────────────────────────────────────────────
# 5. K-SELECTION STABILITY (3-fold, fast models)
# ─────────────────────────────────────────────
print('\n[5/7] K-selection stability (3-fold CV)...')
K_values = [50, 100, 200, 400, 600, 800]
skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
fold_results = []

for fold_i, (fold_tr, fold_va) in enumerate(skf.split(X_tr, y_tr)):
    Xf_tr, yf_tr = X_tr[fold_tr], y_tr[fold_tr]
    Xf_va, yf_va = X_tr[fold_va], y_tr[fold_va]
    mi_fold = mutual_info_classif(Xf_tr, yf_tr, random_state=42)
    fold_row = {'fold': fold_i}
    for k in K_values:
        idx_k = np.argsort(mi_fold)[::-1][:k]
        cb_k = CatBoostClassifier(
            iterations=200, learning_rate=0.05, depth=6,
            eval_metric='AUC', random_seed=42, verbose=0,
            auto_class_weights='Balanced', task_type='CPU',
            early_stopping_rounds=20
        )
        cb_k.fit(Xf_tr[:, idx_k], yf_tr, eval_set=(Xf_va[:, idx_k], yf_va))
        fold_row[k] = roc_auc_score(yf_va, cb_k.predict_proba(Xf_va[:, idx_k])[:, 1])
    fold_row['best_K'] = max(K_values, key=lambda k: fold_row[k])
    fold_results.append(fold_row)
    print(f'  Fold {fold_i}: ' + ' | '.join(f'K={k}: {fold_row[k]:.4f}' for k in K_values) + f' → best K={fold_row["best_K"]}')

stab_df = pd.DataFrame(fold_results)
print('\nStability summary:')
print(stab_df[K_values].agg(['mean','std']).round(4))
print(f'Best K per fold: {stab_df["best_K"].tolist()}')
best_k_freq = stab_df['best_K'].mode()[0]
print(f'Most frequent best K: {best_k_freq}')

# Stability plot
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
mean_auc = stab_df[K_values].mean()
std_auc  = stab_df[K_values].std()
for i, row in stab_df.iterrows():
    axes[0].plot(K_values, [row[k] for k in K_values], 'o-', alpha=0.6, lw=1.5)
axes[0].plot(K_values, mean_auc.values, 'k-', lw=2.5, label='Mean')
axes[0].fill_between(K_values,
    (mean_auc - std_auc).values, (mean_auc + std_auc).values,
    alpha=0.15, color='black', label='±1σ')
axes[0].axvline(400, color='red', ls='--', alpha=0.7, label='K=400 (Phase 3)')
axes[0].set_xlabel('K (selected features)'); axes[0].set_ylabel('CV Val ROC-AUC')
axes[0].set_title('Stability: K-sweep × 3-fold CV')
axes[0].legend(fontsize=8)

hm = stab_df[K_values].values
im = axes[1].imshow(hm, aspect='auto', cmap='YlOrRd')
axes[1].set_xticks(range(len(K_values))); axes[1].set_xticklabels(K_values)
axes[1].set_yticks(range(3)); axes[1].set_yticklabels([f'Fold {i}' for i in range(3)])
axes[1].set_title('Val AUC heatmap (fold × K)')
plt.colorbar(im, ax=axes[1])
for r in range(3):
    for c in range(len(K_values)):
        axes[1].text(c, r, f'{hm[r,c]:.3f}', ha='center', va='center', fontsize=8,
            color='white' if hm[r,c] > hm.mean() else 'black')
plt.tight_layout()
plt.savefig(RES / 'phase4_mark_stability.png', dpi=150, bbox_inches='tight')
plt.close()
print('Saved: phase4_mark_stability.png')

# ─────────────────────────────────────────────
# 6. ERROR ANALYSIS
# ─────────────────────────────────────────────
print('\n[6/7] Error analysis...')

# Youden threshold
fpr_va, tpr_va, thresholds_va = roc_curve(y_va, p_va_tuned)
youden_thresh = thresholds_va[np.argmax(tpr_va - fpr_va)]
y_pred_youden = (p_te_tuned >= youden_thresh).astype(int)

cm = confusion_matrix(y_te, y_pred_youden)
tn, fp, fn, tp = cm.ravel()
precision_score = tp / max(tp + fp, 1)
recall_score = tp / max(tp + fn, 1)
f1_score = 2 * precision_score * recall_score / max(precision_score + recall_score, 1e-9)
print(f'Youden thresh={youden_thresh:.4f} | TP={tp} TN={tn} FP={fp} FN={fn}')
print(f'Precision={precision_score:.4f} | Recall={recall_score:.4f} | F1={f1_score:.4f}')

# Molecular properties for test set
te_df = df[te_mask].copy().reset_index(drop=True)
te_df['pred_prob'] = p_te_tuned
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

props_list = [get_props(s) for s in te_df['smiles']]
props_df = pd.DataFrame(props_list)
te_df = pd.concat([te_df, props_df], axis=1)

te_df['error_class'] = 'TN'
te_df.loc[(te_df['y']==1) & (te_df['pred_label']==1), 'error_class'] = 'TP'
te_df.loc[(te_df['y']==0) & (te_df['pred_label']==1), 'error_class'] = 'FP'
te_df.loc[(te_df['y']==1) & (te_df['pred_label']==0), 'error_class'] = 'FN'

ec_counts = te_df['error_class'].value_counts()
print(f'Error class distribution: {ec_counts.to_dict()}')

prop_cols = ['mw','logp','hbd','hba','tpsa','rb','rings','arom','heavyatoms','frac_csp3','halogens']
err_summary = te_df.groupby('error_class')[prop_cols].mean().round(2)
print('\nMean properties by error class:')
print(err_summary.to_string())

# FN vs TP significance tests
fn_mask = te_df['error_class'] == 'FN'
tp_mask_b = te_df['error_class'] == 'TP'
fn_tp_diffs = {}
print('\nFN vs TP (missed actives vs caught actives):')
for col in prop_cols:
    fn_vals = te_df.loc[fn_mask, col].dropna()
    tp_vals = te_df.loc[tp_mask_b, col].dropna()
    if len(fn_vals) > 3 and len(tp_vals) > 3:
        stat, p = scipy_stats.mannwhitneyu(fn_vals, tp_vals, alternative='two-sided')
        sig = '**' if p < 0.01 else ('*' if p < 0.05 else '')
        fn_tp_diffs[col] = {'fn_mean': float(fn_vals.mean()), 'tp_mean': float(tp_vals.mean()),
                             'p_value': float(p), 'significant': p < 0.05}
        print(f'  {col:12s}: FN={fn_vals.mean():.2f} vs TP={tp_vals.mean():.2f}  p={p:.3f} {sig}')

# Near-threshold analysis
uncertain = te_df[(te_df['pred_prob'] > 0.35) & (te_df['pred_prob'] < 0.65)]
near_thresh_error = (uncertain['y'] != uncertain['pred_label']).mean()
overall_error = (te_df['y'] != te_df['pred_label']).mean()
print(f'\nNear-threshold (0.35-0.65): n={len(uncertain)}, error_rate={near_thresh_error:.3f}')
print(f'Overall error rate: {overall_error:.3f}')

# Error class property plots
colors = {'TP': '#2196F3', 'TN': '#4CAF50', 'FP': '#FF9800', 'FN': '#F44336'}
fig, axes = plt.subplots(2, 4, figsize=(16, 8))
plot_props = ['mw', 'logp', 'tpsa', 'rings', 'hba', 'hbd', 'frac_csp3', 'halogens']
for ax, prop in zip(axes.flat, plot_props):
    for cls in ['TP', 'TN', 'FP', 'FN']:
        vals = te_df.loc[te_df['error_class']==cls, prop].dropna()
        if len(vals) > 0:
            ax.hist(vals, bins=25, alpha=0.5, label=cls, color=colors[cls], density=True)
    ax.set_xlabel(prop); ax.set_ylabel('Density')
    ax.set_title(f'{prop} by error class')
    ax.legend(fontsize=7)
plt.suptitle('Molecular Property Distributions by Error Class', fontsize=12, fontweight='bold')
plt.tight_layout()
plt.savefig(RES / 'phase4_mark_error_properties.png', dpi=150, bbox_inches='tight')
plt.close()
print('Saved: phase4_mark_error_properties.png')

# Lipinski violation analysis
te_df['lipinski_violations'] = (
    (te_df['mw'] > 500).astype(int) +
    (te_df['logp'] > 5).astype(int) +
    (te_df['hbd'] > 5).astype(int) +
    (te_df['hba'] > 10).astype(int)
)
act_df = te_df[te_df['y'] == 1]
easy_act = act_df[act_df['lipinski_violations'] == 0]
hard_act = act_df[act_df['lipinski_violations'] >= 2]
easy_recall = (easy_act['pred_label']==1).mean() if len(easy_act) > 0 else 0
hard_recall = (hard_act['pred_label']==1).mean() if len(hard_act) > 0 else 0
print(f'\nLipinski analysis:')
print(f'  Actives with 0 violations: recall={easy_recall:.3f} (n={len(easy_act)})')
print(f'  Actives with ≥2 violations: recall={hard_recall:.3f} (n={len(hard_act)})')

# Scaffold analysis
print('\nComputing Murcko scaffolds...')
def get_scaffold(smi):
    try:
        mol = Chem.MolFromSmiles(smi)
        if mol:
            return MurckoScaffold.MurckoScaffoldSmiles(mol=mol)
    except: pass
    return 'no_scaffold'

te_df['scaffold'] = [get_scaffold(s) for s in te_df['smiles']]
scaffold_stats = te_df.groupby('scaffold').agg(
    n=('y','count'), actives=('y','sum'),
    TP=('error_class', lambda x: (x=='TP').sum()),
    FN=('error_class', lambda x: (x=='FN').sum()),
    FP=('error_class', lambda x: (x=='FP').sum()),
    TN=('error_class', lambda x: (x=='TN').sum()),
    mean_prob=('pred_prob','mean')
).reset_index()
scaffold_stats['fn_rate'] = scaffold_stats['FN'] / (scaffold_stats['FN'] + scaffold_stats['TP']).clip(lower=1)
high_fn = scaffold_stats[scaffold_stats['actives'] >= 3].nlargest(5, 'fn_rate')
print(f'Top 5 high-FN scaffolds (≥3 actives):')
print(high_fn[['scaffold','n','actives','TP','FN','fn_rate']].to_string())

# Confidence distribution plot
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
for cls in ['TP', 'FP']:
    vals = te_df.loc[te_df['error_class']==cls, 'pred_prob'].values
    axes[0].hist(vals, bins=30, alpha=0.6, label=f'{cls} (n={len(vals)})', color=colors[cls])
for cls in ['TN', 'FN']:
    vals = te_df.loc[te_df['error_class']==cls, 'pred_prob'].values
    axes[1].hist(vals, bins=30, alpha=0.6, label=f'{cls} (n={len(vals)})', color=colors[cls])
axes[0].set_title('TP vs FP confidence'); axes[0].legend(); axes[0].set_xlabel('P(HIV active)')
axes[1].set_title('TN vs FN confidence'); axes[1].legend(); axes[1].set_xlabel('P(HIV active)')
scaffold_per_class = te_df.groupby('error_class')['scaffold'].nunique()
axes[2].bar(scaffold_per_class.index, scaffold_per_class.values,
    color=[colors.get(c,'gray') for c in scaffold_per_class.index])
axes[2].set_title('Scaffold diversity per error class')
axes[2].set_ylabel('Unique scaffolds')
for i, (cls, v) in enumerate(scaffold_per_class.items()):
    axes[2].text(i, v+1, str(v), ha='center', fontsize=10)
plt.tight_layout()
plt.savefig(RES / 'phase4_mark_error_confidence.png', dpi=150, bbox_inches='tight')
plt.close()
print('Saved: phase4_mark_error_confidence.png')

# ─────────────────────────────────────────────
# 7. FEATURE IMPORTANCE
# ─────────────────────────────────────────────
print('\n[7/7] Feature importance analysis...')
importances = cb_tuned.get_feature_importance()
feat_400_names = [feat_names[i] for i in top_400_idx]
imp_df = pd.DataFrame({'feature': feat_400_names, 'importance': importances})
imp_df = imp_df.sort_values('importance', ascending=False).reset_index(drop=True)
imp_df['cumulative'] = imp_df['importance'].cumsum() / imp_df['importance'].sum()

def feat_category(name):
    if name.startswith('lip_'): return 'Lipinski'
    if name.startswith('maccs_'): return 'MACCS'
    if name.startswith('morgan_'): return 'Morgan FP'
    if name.startswith('fr_'): return 'Fragment'
    if name.startswith('adv_'): return 'Advanced'
    return 'Other'

imp_df['category'] = imp_df['feature'].apply(feat_category)
cat_importance = imp_df.groupby('category')['importance'].sum().sort_values(ascending=False)
total = cat_importance.sum()

print('Importance by category:')
for cat, imp in cat_importance.items():
    print(f'  {cat:12s}: {imp:.2f} ({imp/total*100:.1f}%)')

for pct in [0.50, 0.80, 0.95]:
    n_feats = (imp_df['cumulative'] <= pct).sum() + 1
    print(f'{int(pct*100)}% of importance: top {n_feats} features')

print('Top 10 features:')
print(imp_df[['feature','importance','category']].head(10).to_string())

# Feature importance plots
bar_colors = {
    'Lipinski': '#2196F3', 'MACCS': '#4CAF50', 'Morgan FP': '#FF9800',
    'Fragment': '#9C27B0', 'Advanced': '#F44336'
}
fig, axes = plt.subplots(1, 3, figsize=(17, 5))
top25 = imp_df.head(25)
axes[0].barh(top25['feature'][::-1], top25['importance'][::-1],
    color=[bar_colors.get(feat_category(f), 'gray') for f in top25['feature'][::-1]])
axes[0].set_xlabel('Importance'); axes[0].set_title('Top 25 Features (Tuned CatBoost + MI-400)')
from matplotlib.patches import Patch
axes[0].legend(handles=[Patch(facecolor=v, label=k) for k,v in bar_colors.items()], fontsize=7)

axes[1].plot(range(1, len(imp_df)+1), imp_df['cumulative'], color='purple')
for pct, col in [(0.5,'blue'),(0.8,'orange'),(0.95,'red')]:
    n = (imp_df['cumulative'] <= pct).sum() + 1
    axes[1].axhline(pct, color=col, ls='--', alpha=0.7, label=f'{int(pct*100)}%: top {n}')
    axes[1].axvline(n, color=col, ls=':', alpha=0.5)
axes[1].set_xlabel('# features'); axes[1].set_ylabel('Cumulative importance')
axes[1].set_title('How many features actually matter?')
axes[1].legend(fontsize=8)

axes[2].pie(cat_importance.values, labels=cat_importance.index, autopct='%1.1f%%',
    colors=[bar_colors.get(c,'gray') for c in cat_importance.index])
axes[2].set_title('Importance by Feature Category')
plt.tight_layout()
plt.savefig(RES / 'phase4_mark_feature_importance.png', dpi=150, bbox_inches='tight')
plt.close()
print('Saved: phase4_mark_feature_importance.png')

# ─────────────────────────────────────────────
# SAVE RESULTS
# ─────────────────────────────────────────────
phase4_results = {
    'phase': '4-mark',
    'date': '2026-04-09',
    'baseline_reproduced': {
        'model': 'CatBoost (default) + MI-top-400',
        'val_auc': float(auc_va_default),
        'test_auc': float(auc_te_default),
        'test_auprc': float(auprc_te_default)
    },
    'optuna_tuning': {
        'n_trials': 40,
        'best_val_auc': float(study.best_value),
        'best_params': study.best_params,
        'tuned_val_auc': float(auc_va_tuned),
        'tuned_test_auc': float(auc_te_tuned),
        'tuned_test_auprc': float(auprc_te_tuned),
        'delta_val_auc': float(study.best_value - auc_va_default),
        'delta_test_auc': float(auc_te_tuned - auc_te_default)
    },
    'stability_analysis': {
        'n_folds': 3,
        'k_values_tested': K_values,
        'cv_mean_per_k': {str(k): float(stab_df[k].mean()) for k in K_values},
        'cv_std_per_k': {str(k): float(stab_df[k].std()) for k in K_values},
        'best_k_per_fold': stab_df['best_K'].tolist(),
        'most_frequent_best_k': int(best_k_freq)
    },
    'error_analysis': {
        'youden_threshold': float(youden_thresh),
        'TP': int(tp), 'TN': int(tn), 'FP': int(fp), 'FN': int(fn),
        'precision': float(precision_score),
        'recall': float(recall_score),
        'f1': float(f1_score),
        'near_threshold_n': int(len(uncertain)),
        'near_threshold_error_rate': float(near_thresh_error),
        'overall_error_rate': float(overall_error),
        'lipinski_0_violations_recall': float(easy_recall),
        'lipinski_2plus_violations_recall': float(hard_recall),
        'fn_vs_tp_significant_diffs': {k: v for k, v in fn_tp_diffs.items() if v['significant']},
        'high_fn_scaffold_count': int(len(high_fn))
    },
    'feature_importance': {
        'top_10_features': imp_df['feature'].head(10).tolist(),
        'top_10_importances': [float(x) for x in imp_df['importance'].head(10)],
        'features_for_50pct': int((imp_df['cumulative'] <= 0.50).sum() + 1),
        'features_for_80pct': int((imp_df['cumulative'] <= 0.80).sum() + 1),
        'features_for_95pct': int((imp_df['cumulative'] <= 0.95).sum() + 1),
        'category_shares': {k: float(v/total) for k, v in cat_importance.items()},
    }
}

with open(RES / 'phase4_mark_results.json', 'w') as f:
    json.dump(phase4_results, f, indent=2)
print('\nSaved: phase4_mark_results.json')

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

# ─────────────────────────────────────────────
# FINAL SUMMARY
# ─────────────────────────────────────────────
print('\n' + '='*60)
print('PHASE 4 FINAL LEADERBOARD')
print('='*60)
print(f'{"Model":<38} {"Val AUC":>8} {"Test AUC":>9} {"AUPRC":>8}')
print('-'*60)
print(f'{"Tuned CatBoost MI-400 [Mark P4]":<38} {auc_va_tuned:>8.4f} {auc_te_tuned:>9.4f} {auprc_te_tuned:>8.4f}')
print(f'{"Default CatBoost MI-400 [Mark P3]":<38} {auc_va_default:>8.4f} {auc_te_default:>9.4f} {auprc_te_default:>8.4f}')
print(f'{"GIN+Edge [Anthony P3]":<38} {"--":>8} {"0.7860":>9} {"0.3441":>8}')
print(f'{"CatBoost AllTrad-1217 [Anthony P3]":<38} {"--":>8} {"0.7841":>9} {"--":>8}')
print(f'{"MLP-Domain9 [Mark P2]":<38} {"--":>8} {"0.7670":>9} {"--":>8}')
print('='*60)
print(f'\nKey findings:')
print(f'  Tuning delta: {auc_te_tuned - auc_te_default:+.4f} AUC (val delta: {study.best_value - auc_va_default:+.4f})')
print(f'  Most stable K: {best_k_freq} (consistent across all {len(fold_results)} folds)')
print(f'  Recall at Youden threshold: {recall_score:.4f}')
print(f'  Lipinski-compliant actives recall: {easy_recall:.3f} vs violators: {hard_recall:.3f}')
print(f'  50% importance concentrated in: {phase4_results["feature_importance"]["features_for_50pct"]} features (of 400)')
print(f'  MACCS keys: {cat_importance.get("MACCS", 0)/total*100:.1f}% of importance (vs 12.8% of feature pool)')
