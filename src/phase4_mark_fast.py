"""
Phase 4 (Mark) — Fast version: uses pre-computed MI scores from Phase 3 context.
Stability test uses bootstrap subsampling on already-computed MI to avoid re-running MI per fold.
"""
import os, json, time, warnings, random
warnings.filterwarnings('ignore')
os.environ['MPLBACKEND'] = 'Agg'

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from scipy import stats as scipy_stats

import torch
_orig_load = torch.load
torch.load = lambda *a, **k: _orig_load(*a, **{**k, 'weights_only': False})

from ogb.graphproppred import GraphPropPredDataset
from rdkit import Chem
from rdkit.Chem import Descriptors, Fragments, rdMolDescriptors, AllChem, MACCSkeys
from rdkit.Chem.Scaffolds import MurckoScaffold

from sklearn.metrics import roc_auc_score, average_precision_score, confusion_matrix, roc_curve
from sklearn.feature_selection import mutual_info_classif
from catboost import CatBoostClassifier
import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)

BASE = Path(__file__).parent.parent
RES = BASE / 'results'
RES.mkdir(exist_ok=True)
np.random.seed(42); random.seed(42)
print('=== Phase 4 (Mark) — Fast Run ===')

# ─── Load data ───────────────────────────────────────────────────────────────
print('[1] Loading data...')
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
print(f'  Loaded {len(df):,} mols in {time.time()-t0:.1f}s')

# ─── Feature computation ──────────────────────────────────────────────────────
print('[2] Computing features...')
FRAG_FUNCS = [(n, getattr(Fragments, n)) for n in sorted(dir(Fragments)) if n.startswith('fr_')]

def compute_all(smi):
    mol = Chem.MolFromSmiles(smi)
    if mol is None: return None
    mw = Descriptors.MolWt(mol); logp = Descriptors.MolLogP(mol)
    hbd = rdMolDescriptors.CalcNumHBD(mol); hba = rdMolDescriptors.CalcNumHBA(mol)
    tpsa = Descriptors.TPSA(mol); rb = rdMolDescriptors.CalcNumRotatableBonds(mol)
    rings = rdMolDescriptors.CalcNumRings(mol); arom = rdMolDescriptors.CalcNumAromaticRings(mol)
    hal = sum(1 for a in mol.GetAtoms() if a.GetAtomicNum() in (9,17,35,53))
    sp3 = rdMolDescriptors.CalcFractionCSP3(mol)
    nhet = sum(1 for a in mol.GetAtoms() if a.GetAtomicNum() not in (1,6))
    try:
        from rdkit.Chem import QED; qed_val = QED.qed(mol)
    except: qed_val = 0.0
    lip14 = [mw,logp,hbd,hba,tpsa,rb,rings,arom,hal,sp3,nhet,qed_val,
             Descriptors.NumRadicalElectrons(mol), Descriptors.NumValenceElectrons(mol)]
    maccs = list(MACCSkeys.GenMACCSKeys(mol).ToList())
    morgan = list(AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=1024).ToList())
    frags = [int(fn(mol)) for _, fn in FRAG_FUNCS]
    adv = [logp/max(mw,1), hbd+hba, hbd*hba, rb/max(rings,1), arom/max(rings,1),
           sp3*mw, nhet/max(mol.GetNumHeavyAtoms(),1), (hbd+hba)/max(tpsa,1),
           rb*logp, hal/max(mol.GetNumHeavyAtoms(),1), sp3*logp, mw/max(mol.GetNumHeavyAtoms(),1)]
    return lip14 + maccs + morgan + frags + adv

t0 = time.time()
all_feats = [compute_all(s) for s in df['smiles']]
feat_dim = len([f for f in all_feats if f is not None][0])
feat_names = ([f'lip_{i}' for i in range(14)] + [f'maccs_{i}' for i in range(167)] +
              [f'morgan_{i}' for i in range(1024)] + [name for name, _ in FRAG_FUNCS] +
              [f'adv_{i}' for i in range(12)])
X_all = np.array([f if f is not None else [0]*feat_dim for f in all_feats], dtype=np.float32)
y_all = df['y'].values.astype(int)
print(f'  Features computed in {time.time()-t0:.1f}s | dim={feat_dim}')

tr_mask = df['split']=='train'; va_mask = df['split']=='val'; te_mask = df['split']=='test'
X_tr, y_tr = X_all[tr_mask], y_all[tr_mask]
X_va, y_va = X_all[va_mask], y_all[va_mask]
X_te, y_te = X_all[te_mask], y_all[te_mask]

# ─── MI selection ────────────────────────────────────────────────────────────
print('[3] MI feature selection...')
t0 = time.time()
mi_scores = mutual_info_classif(X_tr, y_tr, random_state=42)
top_400_idx = np.argsort(mi_scores)[::-1][:400]
print(f'  MI done in {time.time()-t0:.1f}s | top feature: {feat_names[top_400_idx[0]]}')
X_tr_400 = X_tr[:, top_400_idx]
X_va_400 = X_va[:, top_400_idx]
X_te_400 = X_te[:, top_400_idx]

# ─── Reproduce Phase 3 ───────────────────────────────────────────────────────
print('[4] Reproduce Phase 3 champion...')
cb_default = CatBoostClassifier(
    iterations=500, learning_rate=0.05, depth=6, eval_metric='AUC',
    random_seed=42, verbose=0, auto_class_weights='Balanced', task_type='CPU')
cb_default.fit(X_tr_400, y_tr, eval_set=(X_va_400, y_va), early_stopping_rounds=50)
p_va_def = cb_default.predict_proba(X_va_400)[:,1]
p_te_def = cb_default.predict_proba(X_te_400)[:,1]
auc_va_def = roc_auc_score(y_va, p_va_def)
auc_te_def = roc_auc_score(y_te, p_te_def)
auprc_te_def = average_precision_score(y_te, p_te_def)
print(f'  Default | Val={auc_va_def:.4f} | Test={auc_te_def:.4f} | Phase3 reported: 0.8105 delta={auc_te_def-0.8105:+.4f}')

# ─── Optuna tuning ───────────────────────────────────────────────────────────
print('[5] Optuna tuning (40 trials)...')
def objective(trial):
    params = dict(
        iterations=trial.suggest_int('iterations', 300, 800, step=100),
        learning_rate=trial.suggest_float('learning_rate', 0.01, 0.12, log=True),
        depth=trial.suggest_int('depth', 4, 8),
        l2_leaf_reg=trial.suggest_float('l2_leaf_reg', 1.0, 20.0, log=True),
        min_data_in_leaf=trial.suggest_int('min_data_in_leaf', 5, 50),
        random_strength=trial.suggest_float('random_strength', 0.5, 5.0),
        bagging_temperature=trial.suggest_float('bagging_temperature', 0.0, 2.0),
        border_count=trial.suggest_categorical('border_count', [64, 128, 254]),
    )
    m = CatBoostClassifier(**params, eval_metric='AUC', random_seed=42, verbose=0,
                           auto_class_weights='Balanced', task_type='CPU')
    m.fit(X_tr_400, y_tr, eval_set=(X_va_400, y_va), early_stopping_rounds=30)
    return roc_auc_score(y_va, m.predict_proba(X_va_400)[:,1])

t0 = time.time()
study = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler(seed=42))
study.optimize(objective, n_trials=40)
print(f'  Optuna done in {time.time()-t0:.1f}s | Best val={study.best_value:.4f} (default={auc_va_def:.4f})')
print(f'  Best params: {study.best_params}')

cb_tuned = CatBoostClassifier(**study.best_params, eval_metric='AUC', random_seed=42, verbose=0,
                              auto_class_weights='Balanced', task_type='CPU')
cb_tuned.fit(X_tr_400, y_tr, eval_set=(X_va_400, y_va), early_stopping_rounds=30)
p_va_tun = cb_tuned.predict_proba(X_va_400)[:,1]
p_te_tun = cb_tuned.predict_proba(X_te_400)[:,1]
auc_va_tun = roc_auc_score(y_va, p_va_tun)
auc_te_tun = roc_auc_score(y_te, p_te_tun)
auprc_te_tun = average_precision_score(y_te, p_te_tun)
print(f'  Tuned  | Val={auc_va_tun:.4f} | Test={auc_te_tun:.4f} | delta_test={auc_te_tun-auc_te_def:+.4f}')
print(f'  Val↔Test gap: default={auc_va_def-auc_te_def:+.4f} | tuned={auc_va_tun-auc_te_tun:+.4f}')

# Optuna plot
trial_vals = [t.value for t in study.trials if t.value is not None]
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
ax1.plot(trial_vals, alpha=0.5, color='steelblue')
ax1.axhline(study.best_value, color='red', ls='--', label=f'Best={study.best_value:.4f}')
ax1.axhline(auc_va_def, color='gray', ls=':', label=f'Default={auc_va_def:.4f}')
ax1.set_xlabel('Trial'); ax1.set_ylabel('Val AUC'); ax1.legend(); ax1.set_title('Optuna Trials')
ax2.plot(np.maximum.accumulate(trial_vals), color='darkorange', lw=2, label='Running best')
ax2.axhline(auc_va_def, color='gray', ls=':', label=f'Default={auc_va_def:.4f}')
ax2.set_xlabel('Trial'); ax2.legend(); ax2.set_title('Search Convergence')
plt.tight_layout()
plt.savefig(RES / 'phase4_mark_optuna_history.png', dpi=150, bbox_inches='tight'); plt.close()
print('  Saved: phase4_mark_optuna_history.png')

# ─── Stability: K-sweep on val using global MI ranking ───────────────────────
# Test K stability using global MI scores (already computed).
# Bootstrap the TRAIN split to see if optimal K is consistent.
print('[6] K-stability: 5 bootstrap × K-sweep (global MI, no MI recompute)...')
K_values = [50, 100, 200, 400, 600, 800]
rng = np.random.RandomState(42)
n_tr = len(X_tr)
boot_results = []
for b in range(5):
    idx_b = rng.choice(n_tr, size=int(0.8*n_tr), replace=False)
    boot_row = {'boot': b}
    for k in K_values:
        # Use global MI ranking (already computed, no recompute needed)
        idx_k = top_400_idx[:k] if k <= 400 else np.argsort(mi_scores)[::-1][:k]
        cb_b = CatBoostClassifier(iterations=150, learning_rate=0.07, depth=6,
                                  eval_metric='AUC', random_seed=42, verbose=0,
                                  auto_class_weights='Balanced', task_type='CPU')
        cb_b.fit(X_tr[idx_b][:, idx_k], y_tr[idx_b],
                 eval_set=(X_va[:, idx_k], y_va), early_stopping_rounds=15)
        boot_row[k] = roc_auc_score(y_va, cb_b.predict_proba(X_va[:, idx_k])[:,1])
    boot_row['best_K'] = max(K_values, key=lambda k: boot_row[k])
    boot_results.append(boot_row)
    print(f'  Boot {b}: ' + ' | '.join(f'K={k}:{boot_row[k]:.4f}' for k in K_values) +
          f' → best K={boot_row["best_K"]}')

stab_df = pd.DataFrame(boot_results)
best_k_freq = stab_df['best_K'].mode()[0]
print(f'  Most frequent best K: {best_k_freq} | best-K per boot: {stab_df["best_K"].tolist()}')
print(f'  K=400 mean={stab_df[400].mean():.4f} ± {stab_df[400].std():.4f}')
print(f'  K=200 mean={stab_df[200].mean():.4f} ± {stab_df[200].std():.4f}')

# Stability plot
mean_a = stab_df[K_values].mean(); std_a = stab_df[K_values].std()
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
for i, row in stab_df.iterrows():
    ax1.plot(K_values, [row[k] for k in K_values], 'o-', alpha=0.5, lw=1.5)
ax1.plot(K_values, mean_a.values, 'k-', lw=2.5, label='Mean')
ax1.fill_between(K_values, (mean_a-std_a).values, (mean_a+std_a).values, alpha=0.15, color='black', label='±1σ')
ax1.axvline(400, color='red', ls='--', alpha=0.7, label='K=400 (Phase 3)')
ax1.set_xlabel('K'); ax1.set_ylabel('Val ROC-AUC'); ax1.set_title('K-stability (5 × 80% bootstrap)'); ax1.legend(fontsize=8)
hm = stab_df[K_values].values
im = ax2.imshow(hm, aspect='auto', cmap='YlOrRd')
ax2.set_xticks(range(len(K_values))); ax2.set_xticklabels(K_values)
ax2.set_yticks(range(5)); ax2.set_yticklabels([f'Boot {i}' for i in range(5)])
ax2.set_title('Val AUC heatmap (bootstrap × K)')
plt.colorbar(im, ax=ax2)
for r in range(5):
    for c in range(len(K_values)):
        ax2.text(c, r, f'{hm[r,c]:.3f}', ha='center', va='center', fontsize=8,
                 color='white' if hm[r,c]>hm.mean() else 'black')
plt.tight_layout()
plt.savefig(RES / 'phase4_mark_stability.png', dpi=150, bbox_inches='tight'); plt.close()
print('  Saved: phase4_mark_stability.png')

# ─── Error analysis ──────────────────────────────────────────────────────────
print('[7] Error analysis...')
fpr_va, tpr_va, thresh_va = roc_curve(y_va, p_va_tun)
youden_thresh = thresh_va[np.argmax(tpr_va - fpr_va)]
y_pred = (p_te_tun >= youden_thresh).astype(int)
tn, fp, fn, tp = confusion_matrix(y_te, y_pred).ravel()
prec = tp/max(tp+fp,1); rec = tp/max(tp+fn,1)
f1 = 2*prec*rec/max(prec+rec,1e-9)
print(f'  Youden={youden_thresh:.4f} | TP={tp} TN={tn} FP={fp} FN={fn}')
print(f'  Prec={prec:.4f} Rec={rec:.4f} F1={f1:.4f}')

te_df = df[te_mask].copy().reset_index(drop=True)
te_df['pred_prob'] = p_te_tun; te_df['pred_label'] = y_pred

def get_props(smi):
    mol = Chem.MolFromSmiles(smi)
    if mol is None: return {}
    return {'mw': Descriptors.MolWt(mol), 'logp': Descriptors.MolLogP(mol),
            'hbd': rdMolDescriptors.CalcNumHBD(mol), 'hba': rdMolDescriptors.CalcNumHBA(mol),
            'tpsa': Descriptors.TPSA(mol), 'rb': rdMolDescriptors.CalcNumRotatableBonds(mol),
            'rings': rdMolDescriptors.CalcNumRings(mol), 'arom': rdMolDescriptors.CalcNumAromaticRings(mol),
            'heavyatoms': mol.GetNumHeavyAtoms(), 'frac_csp3': rdMolDescriptors.CalcFractionCSP3(mol),
            'halogens': sum(1 for a in mol.GetAtoms() if a.GetAtomicNum() in (9,17,35,53))}

props_df = pd.DataFrame([get_props(s) for s in te_df['smiles']])
te_df = pd.concat([te_df, props_df], axis=1)
te_df['error_class'] = 'TN'
te_df.loc[(te_df['y']==1) & (te_df['pred_label']==1), 'error_class'] = 'TP'
te_df.loc[(te_df['y']==0) & (te_df['pred_label']==1), 'error_class'] = 'FP'
te_df.loc[(te_df['y']==1) & (te_df['pred_label']==0), 'error_class'] = 'FN'
print(f'  Error classes: {te_df["error_class"].value_counts().to_dict()}')

prop_cols = ['mw','logp','hbd','hba','tpsa','rb','rings','arom','heavyatoms','frac_csp3','halogens']
print('  Mean properties by error class:')
print(te_df.groupby('error_class')[prop_cols].mean().round(2).to_string())

fn_tp_diffs = {}
fn_m = te_df['error_class']=='FN'; tp_m = te_df['error_class']=='TP'
print('\n  FN vs TP (missed actives vs caught):')
for col in prop_cols:
    fn_v = te_df.loc[fn_m, col].dropna(); tp_v = te_df.loc[tp_m, col].dropna()
    if len(fn_v) > 3 and len(tp_v) > 3:
        _, p = scipy_stats.mannwhitneyu(fn_v, tp_v, alternative='two-sided')
        sig = '**' if p < 0.01 else ('*' if p < 0.05 else '')
        fn_tp_diffs[col] = {'fn_mean': float(fn_v.mean()), 'tp_mean': float(tp_v.mean()),
                            'p_value': float(p), 'significant': p < 0.05}
        if p < 0.05:
            print(f'    {col:12s}: FN={fn_v.mean():.2f} vs TP={tp_v.mean():.2f}  p={p:.3f} {sig}')

# Lipinski violations
te_df['lip_viol'] = ((te_df['mw']>500).astype(int) + (te_df['logp']>5).astype(int) +
                     (te_df['hbd']>5).astype(int) + (te_df['hba']>10).astype(int))
act_df = te_df[te_df['y']==1]
easy = act_df[act_df['lip_viol']==0]; hard = act_df[act_df['lip_viol']>=2]
easy_rec = (easy['pred_label']==1).mean() if len(easy) else 0
hard_rec = (hard['pred_label']==1).mean() if len(hard) else 0
print(f'  Lipinski 0-viol actives: recall={easy_rec:.3f} (n={len(easy)})')
print(f'  Lipinski ≥2-viol actives: recall={hard_rec:.3f} (n={len(hard)})')

# Near-threshold
uncertain = te_df[(te_df['pred_prob']>0.35) & (te_df['pred_prob']<0.65)]
near_thresh_err = (uncertain['y'] != uncertain['pred_label']).mean() if len(uncertain) else 0
print(f'  Near-threshold zone (0.35-0.65): n={len(uncertain)}, error_rate={near_thresh_err:.3f}')

# Scaffold analysis
def get_scaffold(smi):
    try:
        mol = Chem.MolFromSmiles(smi)
        if mol: return MurckoScaffold.MurckoScaffoldSmiles(mol=mol)
    except: pass
    return 'no_scaffold'

te_df['scaffold'] = [get_scaffold(s) for s in te_df['smiles']]
sc = te_df.groupby('scaffold').agg(
    n=('y','count'), actives=('y','sum'),
    FN=('error_class', lambda x: (x=='FN').sum()),
    TP=('error_class', lambda x: (x=='TP').sum()),
).reset_index()
sc['fn_rate'] = sc['FN'] / (sc['FN']+sc['TP']).clip(lower=1)
high_fn = sc[sc['actives']>=3].nlargest(5, 'fn_rate')
print(f'  Top 5 high-FN scaffolds (≥3 actives):')
print(high_fn[['scaffold','n','actives','TP','FN','fn_rate']].to_string())

# Error plots
colors = {'TP':'#2196F3','TN':'#4CAF50','FP':'#FF9800','FN':'#F44336'}
fig, axes = plt.subplots(2, 4, figsize=(16, 8))
for ax, prop in zip(axes.flat, ['mw','logp','tpsa','rings','hba','hbd','frac_csp3','halogens']):
    for cls in ['TP','TN','FP','FN']:
        v = te_df.loc[te_df['error_class']==cls, prop].dropna()
        if len(v): ax.hist(v, bins=25, alpha=0.5, label=cls, color=colors[cls], density=True)
    ax.set_xlabel(prop); ax.set_title(f'{prop}'); ax.legend(fontsize=7)
plt.suptitle('Molecular Properties by Error Class', fontsize=12, fontweight='bold')
plt.tight_layout()
plt.savefig(RES / 'phase4_mark_error_properties.png', dpi=150, bbox_inches='tight'); plt.close()

fig, axes = plt.subplots(1, 3, figsize=(15, 5))
for cls in ['TP','FP']:
    v = te_df.loc[te_df['error_class']==cls, 'pred_prob'].values
    axes[0].hist(v, bins=30, alpha=0.6, label=f'{cls}(n={len(v)})', color=colors[cls])
for cls in ['TN','FN']:
    v = te_df.loc[te_df['error_class']==cls, 'pred_prob'].values
    axes[1].hist(v, bins=30, alpha=0.6, label=f'{cls}(n={len(v)})', color=colors[cls])
axes[0].set_title('TP vs FP'); axes[0].legend(); axes[0].set_xlabel('P(active)')
axes[1].set_title('TN vs FN'); axes[1].legend(); axes[1].set_xlabel('P(active)')
spc = te_df.groupby('error_class')['scaffold'].nunique()
axes[2].bar(spc.index, spc.values, color=[colors.get(c,'gray') for c in spc.index])
axes[2].set_title('Scaffold diversity per class')
for i,(c,v) in enumerate(spc.items()): axes[2].text(i, v+1, str(v), ha='center')
plt.tight_layout()
plt.savefig(RES / 'phase4_mark_error_confidence.png', dpi=150, bbox_inches='tight'); plt.close()
print('  Saved error analysis plots')

# ─── Feature importance ───────────────────────────────────────────────────────
print('[8] Feature importance...')
importances = cb_tuned.get_feature_importance()
feat_400_names = [feat_names[i] for i in top_400_idx]
imp_df = pd.DataFrame({'feature': feat_400_names, 'importance': importances})
imp_df = imp_df.sort_values('importance', ascending=False).reset_index(drop=True)
imp_df['cumulative'] = imp_df['importance'].cumsum() / imp_df['importance'].sum()

def feat_cat(name):
    if name.startswith('lip_'): return 'Lipinski'
    if name.startswith('maccs_'): return 'MACCS'
    if name.startswith('morgan_'): return 'Morgan FP'
    if name.startswith('fr_'): return 'Fragment'
    if name.startswith('adv_'): return 'Advanced'
    return 'Other'

imp_df['category'] = imp_df['feature'].apply(feat_cat)
cat_imp = imp_df.groupby('category')['importance'].sum().sort_values(ascending=False)
total = cat_imp.sum()
print('  Category importance:')
for c, v in cat_imp.items(): print(f'    {c:12s}: {v/total*100:.1f}%')
n50 = (imp_df['cumulative']<=0.50).sum()+1
n80 = (imp_df['cumulative']<=0.80).sum()+1
n95 = (imp_df['cumulative']<=0.95).sum()+1
print(f'  50% importance: top {n50} | 80%: top {n80} | 95%: top {n95}')
print(f'  Top 10 features: {imp_df["feature"].head(10).tolist()}')

bar_colors = {'Lipinski':'#2196F3','MACCS':'#4CAF50','Morgan FP':'#FF9800','Fragment':'#9C27B0','Advanced':'#F44336'}
fig, axes = plt.subplots(1, 3, figsize=(17, 5))
top25 = imp_df.head(25)
axes[0].barh(top25['feature'][::-1], top25['importance'][::-1],
             color=[bar_colors.get(feat_cat(f),'gray') for f in top25['feature'][::-1]])
axes[0].set_xlabel('Importance'); axes[0].set_title('Top 25 Features')
from matplotlib.patches import Patch
axes[0].legend(handles=[Patch(facecolor=v, label=k) for k,v in bar_colors.items()], fontsize=7)
axes[1].plot(range(1,len(imp_df)+1), imp_df['cumulative'], color='purple')
for pct, col in [(0.5,'blue'),(0.8,'orange'),(0.95,'red')]:
    n = (imp_df['cumulative']<=pct).sum()+1
    axes[1].axhline(pct, color=col, ls='--', alpha=0.7, label=f'{int(pct*100)}%: top {n}')
axes[1].set_xlabel('# features'); axes[1].set_title('How many features matter?'); axes[1].legend(fontsize=8)
axes[2].pie(cat_imp.values, labels=cat_imp.index, autopct='%1.1f%%',
            colors=[bar_colors.get(c,'gray') for c in cat_imp.index])
axes[2].set_title('Importance by Category')
plt.tight_layout()
plt.savefig(RES / 'phase4_mark_feature_importance.png', dpi=150, bbox_inches='tight'); plt.close()
print('  Saved: phase4_mark_feature_importance.png')

# ─── Save results ─────────────────────────────────────────────────────────────
phase4 = {
    'phase': '4-mark', 'date': '2026-04-09',
    'baseline': {'val_auc': float(auc_va_def), 'test_auc': float(auc_te_def), 'test_auprc': float(auprc_te_def)},
    'optuna': {
        'n_trials': 40, 'best_val_auc': float(study.best_value),
        'best_params': study.best_params,
        'tuned_val_auc': float(auc_va_tun), 'tuned_test_auc': float(auc_te_tun),
        'tuned_test_auprc': float(auprc_te_tun),
        'delta_val': float(study.best_value-auc_va_def),
        'delta_test': float(auc_te_tun-auc_te_def),
        'val_test_gap_default': float(auc_va_def-auc_te_def),
        'val_test_gap_tuned': float(auc_va_tun-auc_te_tun),
    },
    'stability': {
        'method': 'bootstrap_80pct_5x',
        'k_values': K_values,
        'mean_per_k': {str(k): float(stab_df[k].mean()) for k in K_values},
        'std_per_k': {str(k): float(stab_df[k].std()) for k in K_values},
        'best_k_per_boot': stab_df['best_K'].tolist(),
        'most_frequent_k': int(best_k_freq),
    },
    'error_analysis': {
        'youden_threshold': float(youden_thresh),
        'TP': int(tp), 'TN': int(tn), 'FP': int(fp), 'FN': int(fn),
        'precision': float(prec), 'recall': float(rec), 'f1': float(f1),
        'near_threshold_error_rate': float(near_thresh_err),
        'lipinski_0viol_recall': float(easy_rec),
        'lipinski_2plus_recall': float(hard_rec),
        'fn_tp_diffs': {k: v for k,v in fn_tp_diffs.items() if v['significant']},
    },
    'feature_importance': {
        'top10': imp_df['feature'].head(10).tolist(),
        'n_for_50pct': n50, 'n_for_80pct': n80, 'n_for_95pct': n95,
        'category_pct': {k: float(v/total) for k,v in cat_imp.items()},
    }
}
with open(RES / 'phase4_mark_results.json', 'w') as f:
    json.dump(phase4, f, indent=2)
print('Saved: phase4_mark_results.json')

# Append to metrics.json
metrics_path = RES / 'metrics.json'
try:
    with open(metrics_path) as f: all_m = json.load(f)
    if not isinstance(all_m, list): all_m = [all_m]
except: all_m = []
all_m.append(phase4)
with open(metrics_path, 'w') as f: json.dump(all_m, f, indent=2)

# ─── Final summary ───────────────────────────────────────────────────────────
print('\n' + '='*62)
print('PHASE 4 FINAL LEADERBOARD')
print('='*62)
print(f'{"Model":<38} {"ValAUC":>8} {"TestAUC":>8} {"AUPRC":>7}')
print('-'*62)
print(f'{"Tuned CatBoost MI-400 [Mark P4]":<38} {auc_va_tun:>8.4f} {auc_te_tun:>8.4f} {auprc_te_tun:>7.4f}')
print(f'{"Default CatBoost MI-400 [Mark P3]":<38} {auc_va_def:>8.4f} {auc_te_def:>8.4f} {auprc_te_def:>7.4f}')
print(f'{"GIN+Edge [Anthony P3]":<38} {"--":>8} {"0.7860":>8} {"0.3441":>7}')
print(f'{"MLP-Domain9 [Mark P2]":<38} {"--":>8} {"0.7670":>8} {"--":>7}')
print('='*62)
print(f'\nKey insights:')
print(f'  Tuning val lift:    {study.best_value-auc_va_def:+.4f}')
print(f'  Tuning test lift:   {auc_te_tun-auc_te_def:+.4f}')
print(f'  Val↔Test gap grew:  default={auc_va_def-auc_te_def:+.4f} → tuned={auc_va_tun-auc_te_tun:+.4f}')
print(f'  K stability: most frequent best K = {best_k_freq}')
print(f'  50% importance in {n50} features | 80% in {n80} features')
print(f'  MACCS: {cat_imp.get("MACCS",0)/total*100:.1f}% of importance (12.8% of pool)')
print(f'  Lipinski-compliant recall={easy_rec:.3f} vs violators={hard_rec:.3f}')
print(f'\nDone!')
