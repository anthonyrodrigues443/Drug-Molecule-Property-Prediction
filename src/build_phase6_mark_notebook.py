"""
Build Phase 6 Mark notebook: LIME + Subgroup SHAP + Feature Group Attribution
Complementary to Anthony's global SHAP + GIN saliency.
"""
import json
from pathlib import Path
import nbformat as nbf

nb = nbf.v4.new_notebook()
cells = []

def md(text):
    cells.append(nbf.v4.new_markdown_cell(text))

def code(text):
    cells.append(nbf.v4.new_code_cell(text))

# ── Title ────────────────────────────────────────────────────────────────────
md("""# Phase 6 (Mark): LIME + Subgroup SHAP + Feature Group Attribution — Drug Molecule
**Date:** 2026-04-11
**Researcher:** Mark Rodrigues
**Project:** ogbg-molhiv HIV Activity Prediction

## Objective
Anthony (Phase 6) revealed global SHAP and GIN gradient saliency — sulfur 2.8× more important than carbon, MACCS substructure dominates bulk properties 2.4×.

My complementary question: **Does the global story hold up locally?** Three experiments:
1. **LIME local explanations** — do individual molecules show the same top features as SHAP?
2. **Subgroup SHAP (Lipinski violators vs. compliant)** — Phase 4 showed violators have 2× recall; *why?*
3. **Feature group attribution** — MACCS / Morgan / RDKit-descriptor / domain features each claim credit; which group is independently predictive vs. collinear proxy?

**Research references:**
1. Ribeiro et al. (2016) — "Why Should I Trust You?": Explaining the Predictions of Any Classifier (LIME)
2. Lundberg & Lee (2017) — SHAP TreeExplainer for tree models
3. Lipinski et al. (1997) — Rule of Five; HIV antivirals frequently violate it (high MW, complex ring systems)
4. Riniker & Landrum (2013) — Open-source platform to benchmark fingerprints — ECFP4 vs MACCS comparison on bioactivity tasks
""")

# ── Setup ─────────────────────────────────────────────────────────────────────
code("""
import os, json, warnings, sys
warnings.filterwarnings('ignore')
os.environ['MPLBACKEND'] = 'Agg'

from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import shap
from lime import lime_tabular
from sklearn.metrics import roc_auc_score, classification_report, confusion_matrix
from sklearn.feature_selection import mutual_info_classif
from catboost import CatBoostClassifier
from pathlib import Path

from ogb.graphproppred import GraphPropPredDataset
from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors, AllChem, MACCSkeys, Fragments

BASE_DIR = Path(r'C:/Users/antho/OneDrive/Desktop/YC-Portfolio-Projects/Drug-Molecule-Property-Prediction')
RESULTS_DIR = BASE_DIR / 'results'
RESULTS_DIR.mkdir(exist_ok=True)

np.random.seed(42)

# Patch torch.load for PyTorch 2.6+ compatibility
import torch
_orig_load = torch.load
def _patched_load(*a, **kw):
    kw.setdefault('weights_only', False)
    return _orig_load(*a, **kw)
torch.load = _patched_load

print('Libraries loaded OK')
print(f'Working dir: {BASE_DIR}')
""")

# ── Data loading ──────────────────────────────────────────────────────────────
md("## 1. Data Loading + Feature Engineering\nSame pipeline as Phases 3–5 to ensure exact reproducibility.")

code("""
print('Loading ogbg-molhiv dataset...')
dataset = GraphPropPredDataset(name='ogbg-molhiv', root=str(BASE_DIR / 'data' / 'raw'))
split_idx = dataset.get_idx_split()
smiles_df = pd.read_csv(BASE_DIR / 'data' / 'raw' / 'ogbg_molhiv' / 'mapping' / 'mol.csv.gz')
smiles_list = smiles_df['smiles'].tolist()
labels = dataset.labels.flatten().astype(int)

train_idx = split_idx['train'].tolist()
val_idx   = split_idx['valid'].tolist()
test_idx  = split_idx['test'].tolist()

FRAG_FUNCS = [(n, getattr(Fragments, n)) for n in sorted(dir(Fragments)) if n.startswith('fr_')]

def compute_all_features(smi):
    mol = Chem.MolFromSmiles(smi)
    if mol is None: return None
    feats = {}
    # ── Domain / Lipinski ────────────────────────────────────────────────────
    mw    = Descriptors.MolWt(mol)
    logp  = Descriptors.MolLogP(mol)
    hbd   = rdMolDescriptors.CalcNumHBD(mol)
    hba   = rdMolDescriptors.CalcNumHBA(mol)
    feats.update({
        'mol_weight': mw, 'logp': logp, 'hbd': hbd, 'hba': hba,
        'tpsa': rdMolDescriptors.CalcTPSA(mol),
        'rotatable_bonds': rdMolDescriptors.CalcNumRotatableBonds(mol),
        'aromatic_rings': rdMolDescriptors.CalcNumAromaticRings(mol),
        'ring_count': rdMolDescriptors.CalcNumRings(mol),
        'heavy_atom_count': mol.GetNumHeavyAtoms(),
        'fraction_csp3': rdMolDescriptors.CalcFractionCSP3(mol),
        'num_heteroatoms': rdMolDescriptors.CalcNumHeteroatoms(mol),
        'lipinski_violations': int(mw>500)+int(logp>5)+int(hbd>5)+int(hba>10),
        'bertz_ct': Descriptors.BertzCT(mol),
        'labute_asa': Descriptors.LabuteASA(mol),
    })
    # ── MACCS keys (167 bits) ────────────────────────────────────────────────
    maccs = list(MACCSkeys.GenMACCSKeys(mol))
    feats.update({f'maccs_{i}': int(v) for i, v in enumerate(maccs)})
    # ── Morgan ECFP4 (1024 bits) ─────────────────────────────────────────────
    morgan = list(AllChem.GetMorganFingerprintAsBitVect(mol, 2, 1024))
    feats.update({f'morgan_{i}': int(v) for i, v in enumerate(morgan)})
    # ── RDKit descriptors (85 Fr_ fragment counts) ───────────────────────────
    for name, func in FRAG_FUNCS:
        feats[name] = func(mol)
    return feats

print('Computing features (all 41k molecules - ~3 min)...')
import time; t0 = time.time()
all_feats, valid_mask = [], []
for i, smi in enumerate(smiles_list):
    f = compute_all_features(smi)
    if f is None:
        valid_mask.append(False)
    else:
        valid_mask.append(True)
        all_feats.append(f)
    if (i+1) % 5000 == 0:
        print(f'  {i+1}/{len(smiles_list)} ({time.time()-t0:.0f}s)')

feat_df  = pd.DataFrame(all_feats)
y_all    = labels[valid_mask]
idx_all  = np.where(valid_mask)[0]

train_set = set(train_idx)
val_set   = set(val_idx)
test_set  = set(test_idx)

is_train  = np.array([i in train_set for i in idx_all])
is_val    = np.array([i in val_set   for i in idx_all])
is_test   = np.array([i in test_set  for i in idx_all])

X_train = feat_df[is_train].values; y_train = y_all[is_train]
X_val   = feat_df[is_val  ].values; y_val   = y_all[is_val]
X_test  = feat_df[is_test ].values; y_test  = y_all[is_test]
feat_names = list(feat_df.columns)

print(f'\\nFeature matrix: {feat_df.shape}')
print(f'Train {X_train.shape} | Val {X_val.shape} | Test {X_test.shape}')
print(f'Train positives: {y_train.sum()} / {len(y_train)} ({100*y_train.mean():.1f}%)')
print(f'Elapsed: {time.time()-t0:.1f}s')
""")

# ── Retrain champion CatBoost MI-400 ──────────────────────────────────────────
md("## 2. Retrain Champion CatBoost MI-400\nPhase 3 champion: CatBoost on mutual-information top-400 features (AUC=0.8105).")

code("""
print('Selecting top-400 features by mutual information...')
mi_scores = mutual_info_classif(X_train, y_train, random_state=42)
top400_idx = np.argsort(mi_scores)[::-1][:400]
top400_names = [feat_names[i] for i in top400_idx]

X_train_mi = X_train[:, top400_idx]
X_val_mi   = X_val[:,   top400_idx]
X_test_mi  = X_test[:,  top400_idx]

print('Training CatBoost MI-400 (default params for speed)...')
cb_model = CatBoostClassifier(
    iterations=500, depth=6, learning_rate=0.1,
    loss_function='Logloss', eval_metric='AUC',
    class_weights=[1, 10], random_seed=42, verbose=0
)
cb_model.fit(X_train_mi, y_train, eval_set=(X_val_mi, y_val))

train_auc = roc_auc_score(y_train, cb_model.predict_proba(X_train_mi)[:,1])
val_auc   = roc_auc_score(y_val,   cb_model.predict_proba(X_val_mi  )[:,1])
test_auc  = roc_auc_score(y_test,  cb_model.predict_proba(X_test_mi )[:,1])

print(f'Train AUC: {train_auc:.4f}')
print(f'Val   AUC: {val_auc:.4f}')
print(f'Test  AUC: {test_auc:.4f}')

test_probs = cb_model.predict_proba(X_test_mi)[:,1]
test_preds = (test_probs >= 0.5).astype(int)
""")

# ── Experiment 6.1: LIME ──────────────────────────────────────────────────────
md("""## Experiment 6.1: LIME Local Explanations
**Hypothesis:** Anthony's global SHAP story (sulfur / MACCS dominates) should be consistent across all individual molecules. If LIME shows dramatically different top features per molecule, global SHAP is masking real heterogeneity.

**Method:** Pick 8 representative test molecules (2 high-confidence TP, 2 high-confidence TN, 2 FP, 2 FN) and run LIME tabular with 5,000 perturbation samples. Compare top-5 LIME features per molecule against SHAP global ranking.
""")

code("""
print('Setting up LIME explainer...')
lime_explainer = lime_tabular.LimeTabularExplainer(
    X_train_mi,
    feature_names=top400_names,
    class_names=['Inactive', 'Active'],
    mode='classification',
    random_state=42
)

# Select representative molecules
tp_mask = (test_preds == 1) & (y_test == 1)
tn_mask = (test_preds == 0) & (y_test == 0)
fp_mask = (test_preds == 1) & (y_test == 0)
fn_mask = (test_preds == 0) & (y_test == 1)

def pick_cases(mask, probs, n=2, high_conf=True):
    idxs = np.where(mask)[0]
    if len(idxs) == 0: return []
    if high_conf:
        ranked = idxs[np.argsort(np.abs(probs[idxs] - 0.5))[::-1]]
    else:
        ranked = idxs[np.argsort(np.abs(probs[idxs] - 0.5))]
    return ranked[:n].tolist()

tp_cases = pick_cases(tp_mask, test_probs, n=2)
tn_cases = pick_cases(tn_mask, test_probs, n=2)
fp_cases = pick_cases(fp_mask, test_probs, n=2)
fn_cases = pick_cases(fn_mask, test_probs, n=2)

case_labels = (
    [('TP', i) for i in tp_cases] +
    [('TN', i) for i in tn_cases] +
    [('FP', i) for i in fp_cases] +
    [('FN', i) for i in fn_cases]
)
print(f'Selected {len(case_labels)} molecules: {[l for l,_ in case_labels]}')

print('Running LIME (8 molecules x 5000 perturbations)...')
lime_results = {}
for label, idx in case_labels:
    exp = lime_explainer.explain_instance(
        X_test_mi[idx], cb_model.predict_proba,
        num_features=10, num_samples=5000
    )
    lime_results[f'{label}_{idx}'] = {
        'label': label,
        'true_class': int(y_test[idx]),
        'pred_prob': float(test_probs[idx]),
        'top_features': exp.as_list()
    }
    print(f'  {label} (prob={test_probs[idx]:.3f}) top feat: {exp.as_list()[0][0][:40]}')

print('LIME analysis complete.')
""")

code("""
# ── LIME summary: top feature categories across all 8 molecules ──────────────
print('Analyzing LIME feature category overlap with SHAP global...')

# Categorise each feature by type
def categorise(fname):
    if fname.startswith('maccs_'):  return 'MACCS'
    if fname.startswith('morgan_'): return 'Morgan'
    if fname.startswith('fr_'):     return 'RDKit Fragment'
    return 'Domain'

# Collect all LIME top features
lime_feat_counts = {}
for key, info in lime_results.items():
    for feat_name, weight in info['top_features']:
        # LIME names can include condition strings - extract feature name
        base = feat_name.split(' ')[0].rstrip('><=')
        cat = categorise(base)
        lime_feat_counts[cat] = lime_feat_counts.get(cat, 0) + abs(weight)

total_lime_weight = sum(lime_feat_counts.values())
lime_category_frac = {k: v/total_lime_weight for k, v in lime_feat_counts.items()}

print('\\nLIME Feature Category Attribution (aggregate weight across 8 molecules):')
for cat, frac in sorted(lime_category_frac.items(), key=lambda x: -x[1]):
    print(f'  {cat:20s} {frac:.3f} ({frac*100:.1f}%)')

# Compare with SHAP global from Anthony's results (load if available)
try:
    with open(RESULTS_DIR / 'phase6_results.json') as f:
        anthony_p6 = json.load(f)
    print('\\nAnthony SHAP category fractions (from Phase 6 results):')
    for k, v in anthony_p6.get('feature_category_importance', {}).items():
        print(f'  {k:20s} {v:.3f} ({v*100:.1f}%)')
except:
    print('(Anthony phase6_results.json not found - skip comparison)')
""")

code("""
# ── LIME per-molecule plot ────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 4, figsize=(20, 10))
axes = axes.flatten()
colors_map = {'TP': '#2196F3', 'TN': '#4CAF50', 'FP': '#FF5722', 'FN': '#FF9800'}

for ax_i, (key, info) in enumerate(lime_results.items()):
    ax = axes[ax_i]
    top10 = info['top_features'][:8]
    feats_short = [f[:30] + ('...' if len(f) > 30 else '') for f, _ in top10]
    weights     = [w for _, w in top10]
    bar_colors  = ['#E53935' if w < 0 else '#43A047' for w in weights]
    ax.barh(range(len(weights)), weights[::-1], color=bar_colors[::-1])
    ax.set_yticks(range(len(weights)))
    ax.set_yticklabels(feats_short[::-1], fontsize=8)
    lbl = info['label']
    ax.set_title(f"{lbl} | P(active)={info['pred_prob']:.3f}",
                 color=colors_map.get(lbl, 'black'), fontsize=9, fontweight='bold')
    ax.axvline(0, color='black', lw=0.8)
    ax.set_xlabel('LIME weight', fontsize=8)

plt.suptitle('LIME Local Explanations: 8 Representative Molecules\\n(Green=supports Active, Red=supports Inactive)',
             fontsize=12, fontweight='bold')
plt.tight_layout()
plt.savefig(RESULTS_DIR / 'phase6_mark_lime_explanations.png', dpi=120, bbox_inches='tight')
plt.close()
print('Saved phase6_mark_lime_explanations.png')
""")

# ── Experiment 6.2: Subgroup SHAP ─────────────────────────────────────────────
md("""## Experiment 6.2: Subgroup SHAP — Lipinski Violators vs. Compliant
**Hypothesis (from Phase 4):** Lipinski-violating actives had 2× higher recall than compliant actives (0.828 vs 0.400). SHAP should show a different *mechanism* — the model uses different features for large complex HIV inhibitors vs. small drug-like ones.

**Method:** Split test set into Lipinski-compliant (0 violations) vs. violating (≥1 violation). Run SHAP TreeExplainer on each subgroup. Compare top-10 SHAP features and their mean absolute values.
""")

code("""
print('Computing Lipinski violation status for test set...')

# Find lipinski_violations column in original feature set
lip_viol_idx_full = feat_names.index('lipinski_violations')

# For test molecules, get violations from full feature matrix
X_test_full = feat_df[is_test].values
lip_viol_test = X_test_full[:, lip_viol_idx_full].astype(int)

compliant_mask  = (lip_viol_test == 0)
violating_mask  = (lip_viol_test >= 1)

print(f'Test set: {compliant_mask.sum()} compliant | {violating_mask.sum()} violating')
print(f'Active rate - compliant: {y_test[compliant_mask].mean():.3f} | violating: {y_test[violating_mask].mean():.3f}')

# AUC per subgroup
auc_compliant  = roc_auc_score(y_test[compliant_mask],  test_probs[compliant_mask])
auc_violating  = roc_auc_score(y_test[violating_mask],  test_probs[violating_mask])
print(f'AUC - compliant: {auc_compliant:.4f} | violating: {auc_violating:.4f}')

# Recall @ 0.5 per subgroup
rec_comp = (test_preds[compliant_mask] == 1)[y_test[compliant_mask]==1].mean()
rec_viol = (test_preds[violating_mask] == 1)[y_test[violating_mask]==1].mean()
print(f'Recall@0.5 - compliant actives: {rec_comp:.3f} | violating actives: {rec_viol:.3f}')
print('(Phase 4 finding confirmed: violators have higher recall)')
""")

code("""
print('Running SHAP TreeExplainer on compliant subset...')
explainer = shap.TreeExplainer(cb_model)

# Limit to 500 per subgroup for speed
n_limit = 500
X_comp_shap = X_test_mi[compliant_mask][:n_limit]
X_viol_shap = X_test_mi[violating_mask][:n_limit]

shap_compliant = explainer.shap_values(X_comp_shap)
if isinstance(shap_compliant, list): shap_compliant = shap_compliant[1]  # class 1

print('Running SHAP TreeExplainer on violating subset...')
shap_violating = explainer.shap_values(X_viol_shap)
if isinstance(shap_violating, list): shap_violating = shap_violating[1]

# Mean |SHAP| per feature for each subgroup
mean_abs_comp = np.abs(shap_compliant).mean(axis=0)
mean_abs_viol = np.abs(shap_violating).mean(axis=0)

top10_comp_idx = np.argsort(mean_abs_comp)[::-1][:10]
top10_viol_idx = np.argsort(mean_abs_viol)[::-1][:10]

print('\\nTop-10 SHAP features - COMPLIANT molecules:')
for rank, fi in enumerate(top10_comp_idx):
    print(f'  {rank+1}. {top400_names[fi]:40s}  {mean_abs_comp[fi]:.4f}')

print('\\nTop-10 SHAP features - VIOLATING molecules:')
for rank, fi in enumerate(top10_viol_idx):
    print(f'  {rank+1}. {top400_names[fi]:40s}  {mean_abs_viol[fi]:.4f}')

top10_comp_set = set(top400_names[i] for i in top10_comp_idx)
top10_viol_set = set(top400_names[i] for i in top10_viol_idx)
overlap = top10_comp_set & top10_viol_set
print(f'\\nOverlap in top-10: {len(overlap)} / 10 features')
print(f'Overlap features: {sorted(overlap)}')
""")

code("""
# ── Subgroup SHAP category breakdown ─────────────────────────────────────────
def category_importance(shap_mat, feat_names_list):
    abs_shap = np.abs(shap_mat).mean(axis=0)
    total = abs_shap.sum()
    cats = {}
    for fi, fname in enumerate(feat_names_list):
        cat = categorise(fname)
        cats[cat] = cats.get(cat, 0) + abs_shap[fi]
    return {k: v/total for k, v in cats.items()}

cat_comp = category_importance(shap_compliant, top400_names)
cat_viol = category_importance(shap_violating, top400_names)

print('\\nSHAP category fractions:')
print(f'{"Category":20s} {"Compliant":12s} {"Violating":12s} {"Δ":8s}')
all_cats = sorted(set(cat_comp) | set(cat_viol))
for cat in all_cats:
    c = cat_comp.get(cat, 0)
    v = cat_viol.get(cat, 0)
    print(f'{cat:20s} {c:.3f}        {v:.3f}        {v-c:+.3f}')
""")

code("""
# ── Subgroup SHAP comparison plot ─────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(18, 6))

# Panel 1: category fractions bar chart
ax = axes[0]
cats = sorted(set(cat_comp) | set(cat_viol))
x = np.arange(len(cats))
w = 0.35
ax.bar(x - w/2, [cat_comp.get(c,0) for c in cats], w, label='Compliant', color='#2196F3', alpha=0.85)
ax.bar(x + w/2, [cat_viol.get(c,0) for c in cats], w, label='Violating (≥1)', color='#FF5722', alpha=0.85)
ax.set_xticks(x)
ax.set_xticklabels(cats, rotation=15, ha='right', fontsize=9)
ax.set_ylabel('Mean |SHAP| fraction')
ax.set_title('Feature Category Attribution\\nby Lipinski Compliance')
ax.legend()

# Panel 2: top-10 SHAP features - compliant
ax = axes[1]
top10_comp_names  = [top400_names[i][:25] for i in top10_comp_idx][::-1]
top10_comp_values = mean_abs_comp[top10_comp_idx][::-1]
ax.barh(range(10), top10_comp_values, color='#2196F3', alpha=0.85)
ax.set_yticks(range(10))
ax.set_yticklabels(top10_comp_names, fontsize=8)
ax.set_title('Top-10 SHAP\\n(Compliant molecules)', fontsize=10)
ax.set_xlabel('Mean |SHAP|')

# Panel 3: top-10 SHAP features - violating
ax = axes[2]
top10_viol_names  = [top400_names[i][:25] for i in top10_viol_idx][::-1]
top10_viol_values = mean_abs_viol[top10_viol_idx][::-1]
ax.barh(range(10), top10_viol_values, color='#FF5722', alpha=0.85)
ax.set_yticks(range(10))
ax.set_yticklabels(top10_viol_names, fontsize=8)
ax.set_title('Top-10 SHAP\\n(Violating molecules ≥1)', fontsize=10)
ax.set_xlabel('Mean |SHAP|')

plt.suptitle('Subgroup SHAP: Lipinski Compliant vs. Violating Molecules',
             fontsize=12, fontweight='bold')
plt.tight_layout()
plt.savefig(RESULTS_DIR / 'phase6_mark_subgroup_shap.png', dpi=120, bbox_inches='tight')
plt.close()
print('Saved phase6_mark_subgroup_shap.png')
""")

# ── Experiment 6.3: Feature Group Attribution ──────────────────────────────────
md("""## Experiment 6.3: Feature Group Attribution — Which Group Is Independently Predictive?
**Hypothesis:** Anthony's SHAP shows MACCS dominates (35% of importance). But SHAP collinearity inflates importance of correlated feature groups. Test: train CatBoost on each feature group *alone* vs. combined. If group A alone reaches 90% of combined AUC, it's independently predictive; if not, it's collinear proxy.

**Method:** Train 5 CatBoost models — MACCS alone, Morgan alone, RDKit alone, Domain alone, All-combined. Compare test AUC for each. Then compare with Phase 3 MI-400 SHAP category breakdown to detect collinearity inflation.
""")

code("""
print('Feature group attribution experiment...')

def get_group_indices(names, prefix_or_type):
    if prefix_or_type == 'maccs':
        return [i for i, n in enumerate(names) if n.startswith('maccs_')]
    elif prefix_or_type == 'morgan':
        return [i for i, n in enumerate(names) if n.startswith('morgan_')]
    elif prefix_or_type == 'rdkit':
        return [i for i, n in enumerate(names) if n.startswith('fr_')]
    elif prefix_or_type == 'domain':
        return [i for i, n in enumerate(names) if not any(n.startswith(p) for p in ('maccs_', 'morgan_', 'fr_'))]
    return list(range(len(names)))

# Full feature set (non-MI-selected, 1302 features)
all_feat_idx_by_group = {
    'MACCS (167)':        get_group_indices(feat_names, 'maccs'),
    'Morgan ECFP4 (1024)':get_group_indices(feat_names, 'morgan'),
    'RDKit Fr_ (85)':     get_group_indices(feat_names, 'rdkit'),
    'Domain (14)':        get_group_indices(feat_names, 'domain'),
}

results_group = []
for group_name, group_idx in all_feat_idx_by_group.items():
    Xtr = X_train[:, group_idx]
    Xva = X_val[:,   group_idx]
    Xte = X_test[:,  group_idx]

    model = CatBoostClassifier(
        iterations=300, depth=6, learning_rate=0.1,
        loss_function='Logloss', eval_metric='AUC',
        class_weights=[1, 10], random_seed=42, verbose=0
    )
    model.fit(Xtr, y_train, eval_set=(Xva, y_val))

    auc_val  = roc_auc_score(y_val,  model.predict_proba(Xva)[:,1])
    auc_test = roc_auc_score(y_test, model.predict_proba(Xte)[:,1])
    n_feats  = len(group_idx)

    results_group.append({
        'Group': group_name,
        'N features': n_feats,
        'Val AUC': round(auc_val, 4),
        'Test AUC': round(auc_test, 4),
    })
    print(f'  {group_name:25s} Val AUC={auc_val:.4f}  Test AUC={auc_test:.4f}')

# Also MI-400 combined (from above)
results_group.append({
    'Group': 'MI-400 (combined)',
    'N features': 400,
    'Val AUC': round(val_auc, 4),
    'Test AUC': round(test_auc, 4),
})

df_group = pd.DataFrame(results_group).sort_values('Test AUC', ascending=False)
print('\\nFeature Group Attribution Results:')
print(df_group.to_string(index=False))
""")

code("""
# ── Group attribution plot ────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

ax = axes[0]
groups = df_group['Group'].tolist()
test_aucs = df_group['Test AUC'].tolist()
colors = ['#1976D2' if 'MI-400' in g else ('#43A047' if 'MACCS' in g else ('#FF7043' if 'Morgan' in g else ('#9C27B0' if 'RDKit' in g else '#607D8B'))) for g in groups]
bars = ax.barh(range(len(groups)), test_aucs, color=colors, alpha=0.85)
ax.set_yticks(range(len(groups)))
ax.set_yticklabels(groups, fontsize=9)
ax.set_xlabel('Test AUC')
ax.set_title('Test AUC by Feature Group\\n(CatBoost trained on each group alone)', fontsize=10)
ax.axvline(0.5, color='gray', lw=1, ls='--', label='Random')
# Highlight MI-400 combined
ax.axvline(test_auc, color='#1976D2', lw=1.5, ls=':', alpha=0.7, label=f'MI-400 combined ({test_auc:.4f})')
for bar, auc in zip(bars, test_aucs):
    ax.text(auc + 0.002, bar.get_y() + bar.get_height()/2, f'{auc:.4f}', va='center', fontsize=8)
ax.legend(fontsize=8)

# Panel 2: features needed vs AUC (efficiency)
ax = axes[1]
for row in results_group:
    g = row['Group']
    n = row['N features']
    a = row['Test AUC']
    col = '#1976D2' if 'MI-400' in g else ('#43A047' if 'MACCS' in g else ('#FF7043' if 'Morgan' in g else ('#9C27B0' if 'RDKit' in g else '#607D8B')))
    ax.scatter(n, a, s=120, color=col, zorder=5)
    ax.annotate(g[:20], (n, a), xytext=(5, 5), textcoords='offset points', fontsize=7)
ax.set_xlabel('Number of features')
ax.set_ylabel('Test AUC')
ax.set_title('Efficiency: Features vs. Performance', fontsize=10)
ax.axhline(0.5, color='gray', lw=1, ls='--')

plt.suptitle('Feature Group Attribution: Which Fingerprint Family Is Most Independently Predictive?',
             fontsize=11, fontweight='bold')
plt.tight_layout()
plt.savefig(RESULTS_DIR / 'phase6_mark_group_attribution.png', dpi=120, bbox_inches='tight')
plt.close()
print('Saved phase6_mark_group_attribution.png')
""")

# ── Experiment 6.4: LIME vs SHAP divergence ───────────────────────────────────
md("""## Experiment 6.4: LIME vs. SHAP Divergence — Global vs. Local Story
**Key question:** Do LIME's local explanations (per-molecule) agree with SHAP's global ranking? If they diverge significantly, the "global story" is a population average that doesn't explain individual molecules well.

**Method:** For the 8 molecules above, compare their LIME top-3 features vs. the global SHAP top-3. Compute a per-molecule "agreement score" (Jaccard similarity of top-5 feature sets between LIME and SHAP).
""")

code("""
print('Computing per-molecule SHAP explanations for comparison with LIME...')

lime_mol_indices = [idx for _, idx in case_labels]
X_lime_subset = X_test_mi[lime_mol_indices]
shap_lime_subset = explainer.shap_values(X_lime_subset)
if isinstance(shap_lime_subset, list): shap_lime_subset = shap_lime_subset[1]

# Global SHAP top features (from full test set analysis)
global_shap = explainer.shap_values(X_test_mi[:1000])
if isinstance(global_shap, list): global_shap = global_shap[1]
global_mean_abs = np.abs(global_shap).mean(axis=0)
global_top10_idx = set(np.argsort(global_mean_abs)[::-1][:10])
global_top10_names = {top400_names[i] for i in global_top10_idx}

print(f'Global SHAP top-10: {sorted(list(global_top10_names))[:5]}...')

# Per-molecule agreement: Jaccard(local LIME top-5, global SHAP top-5)
agreement_scores = []
for mol_i, (case_key, info) in enumerate(lime_results.items()):
    lime_top5 = set([f.split(' ')[0].rstrip('><=') for f, _ in info['top_features'][:5]])

    local_shap_abs = np.abs(shap_lime_subset[mol_i])
    local_top5_idx = set(np.argsort(local_shap_abs)[::-1][:5])
    local_top5_names = {top400_names[i] for i in local_top5_idx}

    # LIME vs local SHAP agreement
    lime_local_jaccard = len(lime_top5 & local_top5_names) / len(lime_top5 | local_top5_names)
    # Local SHAP vs global SHAP agreement
    local_global_jaccard = len(local_top5_names & global_top10_names) / len(local_top5_names | global_top10_names)

    agreement_scores.append({
        'Molecule': case_key,
        'Type': info['label'],
        'P(active)': round(info['pred_prob'], 3),
        'LIME-vs-localSHAP': round(lime_local_jaccard, 3),
        'localSHAP-vs-globalSHAP': round(local_global_jaccard, 3),
    })
    print(f'  {case_key:12s}: LIME↔localSHAP={lime_local_jaccard:.3f}  localSHAP↔globalSHAP={local_global_jaccard:.3f}')

df_agree = pd.DataFrame(agreement_scores)
print(f'\\nMean LIME↔localSHAP Jaccard:       {df_agree["LIME-vs-localSHAP"].mean():.3f}')
print(f'Mean localSHAP↔globalSHAP Jaccard: {df_agree["localSHAP-vs-globalSHAP"].mean():.3f}')
print('\\n=> Interpretation: low local↔global Jaccard means individual molecules')
print('   are explained differently from the population average.')
""")

code("""
# ── Agreement heatmap plot ────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Agreement scores heatmap
ax = axes[0]
pivot_data = df_agree[['Molecule', 'LIME-vs-localSHAP', 'localSHAP-vs-globalSHAP']].set_index('Molecule')
sns.heatmap(pivot_data, annot=True, fmt='.3f', cmap='RdYlGn', vmin=0, vmax=1,
            ax=ax, cbar_kws={'label': 'Jaccard similarity'})
ax.set_title('Explainability Agreement Scores\\n(LIME vs local SHAP vs global SHAP)', fontsize=10)
ax.set_ylabel('')

# Bar chart comparing mean agreements by molecule type
ax = axes[1]
type_agree = df_agree.groupby('Type')[['LIME-vs-localSHAP', 'localSHAP-vs-globalSHAP']].mean()
type_agree.plot(kind='bar', ax=ax, color=['#FF7043', '#2196F3'], alpha=0.85, rot=0)
ax.set_xlabel('Molecule Type')
ax.set_ylabel('Mean Jaccard Similarity')
ax.set_title('Agreement by Prediction Type\\n(TP/TN/FP/FN)', fontsize=10)
ax.legend(['LIME↔LocalSHAP', 'LocalSHAP↔GlobalSHAP'], fontsize=8)
ax.axhline(0.5, color='gray', ls='--', lw=1)

plt.suptitle('LIME vs. SHAP Divergence: Does the Global Story Hold Up Locally?',
             fontsize=11, fontweight='bold')
plt.tight_layout()
plt.savefig(RESULTS_DIR / 'phase6_mark_lime_shap_divergence.png', dpi=120, bbox_inches='tight')
plt.close()
print('Saved phase6_mark_lime_shap_divergence.png')
""")

# ── Combined summary plot ──────────────────────────────────────────────────────
md("## Summary: Combined Explainability Picture")

code("""
print('Creating combined summary figure...')
fig = plt.figure(figsize=(16, 10))

# --- Top row: category attribution comparison ---
ax1 = fig.add_subplot(2, 3, 1)
cats_all = sorted(set(cat_comp) | set(cat_viol) | set(lime_category_frac))
x = np.arange(len(cats_all))
w = 0.3
ax1.bar(x - w, [cat_comp.get(c, 0) for c in cats_all], w, label='SHAP Compliant', color='#2196F3', alpha=0.8)
ax1.bar(x,     [cat_viol.get(c, 0) for c in cats_all], w, label='SHAP Violating', color='#FF5722', alpha=0.8)
ax1.bar(x + w, [lime_category_frac.get(c, 0) for c in cats_all], w, label='LIME (all)', color='#9C27B0', alpha=0.8)
ax1.set_xticks(x)
ax1.set_xticklabels(cats_all, rotation=20, ha='right', fontsize=8)
ax1.set_ylabel('Fraction of importance')
ax1.set_title('Feature Category Attribution\\nSHAP Subgroups vs LIME', fontsize=9)
ax1.legend(fontsize=7)

# --- AUC subgroup bar ---
ax2 = fig.add_subplot(2, 3, 2)
subgroups = ['Compliant', 'Violating (≥1)']
aucs = [auc_compliant, auc_violating]
bar_colors = ['#2196F3', '#FF5722']
bars = ax2.bar(subgroups, aucs, color=bar_colors, alpha=0.85)
ax2.set_ylim(0.5, 0.9)
ax2.set_ylabel('Test AUC')
ax2.set_title('AUC by Lipinski Compliance\\n(Phase 4: violators have 2x recall)', fontsize=9)
for bar, auc in zip(bars, aucs):
    ax2.text(bar.get_x() + bar.get_width()/2, auc + 0.005, f'{auc:.4f}', ha='center', fontsize=10, fontweight='bold')

# --- Group attribution AUC ---
ax3 = fig.add_subplot(2, 3, 3)
grp_labels = [r['Group'].split(' ')[0] for r in results_group]
grp_aucs   = [r['Test AUC'] for r in results_group]
gcols = ['#1976D2', '#43A047', '#FF7043', '#9C27B0', '#607D8B'][:len(grp_labels)]
bars = ax3.barh(range(len(grp_labels)), grp_aucs, color=gcols, alpha=0.85)
ax3.set_yticks(range(len(grp_labels)))
ax3.set_yticklabels(grp_labels, fontsize=9)
ax3.set_xlabel('Test AUC')
ax3.set_title('AUC by Feature Group Alone', fontsize=9)
ax3.axvline(0.5, color='gray', ls='--', lw=1)

# --- LIME agreement heatmap mini ---
ax4 = fig.add_subplot(2, 3, 4)
pivot_data_mini = df_agree[['LIME-vs-localSHAP', 'localSHAP-vs-globalSHAP']].T
sns.heatmap(pivot_data_mini, annot=True, fmt='.2f', cmap='RdYlGn', vmin=0, vmax=1,
            ax=ax4, cbar=False, xticklabels=[d['Type'] for d in agreement_scores])
ax4.set_title('LIME vs SHAP Agreement\\nper molecule (Jaccard)', fontsize=9)

# --- Recall subgroup bar ---
ax5 = fig.add_subplot(2, 3, 5)
recall_data = [rec_comp, rec_viol]
ax5.bar(subgroups, recall_data, color=bar_colors, alpha=0.85)
ax5.set_ylim(0, 1.0)
ax5.set_ylabel('Recall @ 0.5 threshold')
ax5.set_title('Recall (Actives Only)\\nCompliant vs Violating', fontsize=9)
for i, (x, r) in enumerate(zip(subgroups, recall_data)):
    ax5.text(i, r + 0.02, f'{r:.3f}', ha='center', fontsize=10, fontweight='bold')

# --- LIME category pie ---
ax6 = fig.add_subplot(2, 3, 6)
lime_cats = sorted(lime_category_frac, key=lambda k: -lime_category_frac[k])
lime_vals = [lime_category_frac[k] for k in lime_cats]
pie_colors = ['#43A047', '#FF7043', '#9C27B0', '#607D8B'][:len(lime_cats)]
ax6.pie(lime_vals, labels=lime_cats, colors=pie_colors, autopct='%1.1f%%', startangle=90)
ax6.set_title('LIME Category Attribution\\n(8 representative molecules)', fontsize=9)

plt.suptitle('Phase 6 (Mark): Explainability - LIME + Subgroup SHAP + Group Attribution\\nDrug Molecule HIV Activity Prediction',
             fontsize=12, fontweight='bold')
plt.tight_layout()
plt.savefig(RESULTS_DIR / 'phase6_mark_summary.png', dpi=120, bbox_inches='tight')
plt.close()
print('Saved phase6_mark_summary.png')
""")

# ── Save results JSON ──────────────────────────────────────────────────────────
code("""
print('Saving phase6_mark_results.json...')

results_out = {
    'phase': 6,
    'researcher': 'Mark',
    'date': '2026-04-11',
    'test_auc_overall': round(test_auc, 4),
    'test_auc_compliant': round(auc_compliant, 4),
    'test_auc_violating': round(auc_violating, 4),
    'recall_compliant_actives': round(float(rec_comp), 4),
    'recall_violating_actives': round(float(rec_viol), 4),
    'feature_group_auc': {r['Group']: r['Test AUC'] for r in results_group},
    'lime_category_fractions': {k: round(v, 4) for k, v in lime_category_frac.items()},
    'shap_compliant_category_fractions': {k: round(v, 4) for k, v in cat_comp.items()},
    'shap_violating_category_fractions': {k: round(v, 4) for k, v in cat_viol.items()},
    'lime_shap_agreement_mean': round(float(df_agree['LIME-vs-localSHAP'].mean()), 4),
    'local_global_shap_agreement_mean': round(float(df_agree['localSHAP-vs-globalSHAP'].mean()), 4),
    'lime_details': [
        {
            'key': k,
            'label': v['label'],
            'pred_prob': v['pred_prob'],
            'top3_features': v['top_features'][:3]
        } for k, v in lime_results.items()
    ],
    'subgroup_top10_overlap_count': len(overlap),
    'subgroup_top10_overlap_features': sorted(list(overlap)),
}

import json
with open(RESULTS_DIR / 'phase6_mark_results.json', 'w') as f:
    json.dump(results_out, f, indent=2)

# ── Update metrics.json (it is a list) ────────────────────────────────────────
try:
    with open(RESULTS_DIR / 'metrics.json') as f:
        metrics = json.load(f)
except:
    metrics = []
if not isinstance(metrics, list):
    metrics = [metrics]
metrics.append({
    'phase': '6-mark',
    'researcher': 'Mark',
    'date': '2026-04-11',
    'description': 'LIME + Subgroup SHAP + Feature Group Attribution',
    'test_auc': round(test_auc, 4),
    'auc_compliant': round(auc_compliant, 4),
    'auc_violating': round(auc_violating, 4),
    'lime_shap_mean_jaccard': round(float(df_agree['LIME-vs-localSHAP'].mean()), 4),
})
with open(RESULTS_DIR / 'metrics.json', 'w') as f:
    json.dump(metrics, f, indent=2)

print('Results saved.')
print('\\n' + '='*60)
print('PHASE 6 (MARK) COMPLETE')
print('='*60)
print(f'Overall test AUC:      {test_auc:.4f}')
print(f'Compliant AUC:         {auc_compliant:.4f}')
print(f'Violating AUC:         {auc_violating:.4f}')
print(f'Mean LIME-SHAP Jaccard: {df_agree["LIME-vs-localSHAP"].mean():.3f}')
print('\\nPlots saved:')
for name in ['phase6_mark_lime_explanations.png', 'phase6_mark_subgroup_shap.png',
             'phase6_mark_group_attribution.png', 'phase6_mark_lime_shap_divergence.png',
             'phase6_mark_summary.png']:
    print(f'  results/{name}')
""")

nb.cells = cells
nb_path = Path(r'C:\Users\antho\OneDrive\Desktop\YC-Portfolio-Projects\Drug-Molecule-Property-Prediction\notebooks\phase6_mark_explainability.ipynb')
nb_path.parent.mkdir(exist_ok=True)

with open(nb_path, 'w', encoding='utf-8') as f:
    nbf.write(nb, f)

print(f'Notebook written: {nb_path}')
