"""
Phase 1 Mark: Class-Imbalance Strategies + Graph-Level Features on ogbg-molhiv
===============================================================================
Complementary experiment to Anthony's Phase 1.

Anthony found: RF (Combined 1036) = ROC-AUC 0.7707 champion. Combined > individual.

My question: At 3.5% positive rate, is the bottleneck the MODEL or the CLASS IMBALANCE?
  - Does class_weight='balanced' help or hurt ROC-AUC?
  - Do graph-level topological features (n_atoms, n_bonds, graph density) add signal?
  - Does threshold tuning on validation set improve AUPRC more than model choice?
  - Can a simple cost-sensitive LogReg rival RF's 0.77 AUC?

Key hypothesis: For 3.5% imbalance, class-weighting helps recall but may hurt AUC.
"""

import os, sys, json, time, warnings
os.environ['BABEL_DATADIR'] = (
    r'C:\Users\antho\AppData\Local\Programs\Python\Python311\Lib'
    r'\site-packages\openbabel\bin\data'
)
os.environ['PYTHONUTF8'] = '1'

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (roc_auc_score, average_precision_score, f1_score,
                              precision_score, recall_score, roc_curve, precision_recall_curve)
import xgboost as xgb
import lightgbm as lgbm

warnings.filterwarnings('ignore')

ROOT = Path(__file__).parent.parent
DATA_DIR = ROOT / 'data'
RESULTS_DIR = ROOT / 'results'
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

print('=' * 72)
print('PHASE 1 (Mark): Class-Imbalance Strategies on ogbg-molhiv')
print('=' * 72)

# ===================================================================
# 1. LOAD PREPROCESSED FEATURES
# ===================================================================
print('\n[1] Loading preprocessed features...')
df = pd.read_parquet(DATA_DIR / 'processed' / 'ogbg_molhiv_features.parquet')
print(f'  Total: {len(df)} molecules')
print(f'  Positive rate: {df["hiv_active"].mean():.4f} ({df["hiv_active"].sum()} active)')
print(f'  Splits: {df["split"].value_counts().to_dict()}')

# Define feature sets
GRAPH_FEATS = ['n_atoms', 'n_bonds']
DESCRIPTOR_FEATS = ['mol_weight', 'hbd', 'hba', 'rotatable_bonds',
                     'aromatic_rings', 'ring_count', 'heavy_atom_count']
FP_COLS = [c for c in df.columns if c.startswith('fp_')]

# Add engineered graph features
df['graph_density'] = (2 * df['n_bonds']) / (df['n_atoms'] * (df['n_atoms'] - 1)).clip(lower=1)
df['bonds_per_atom'] = df['n_bonds'] / df['n_atoms'].clip(lower=1)
df['aromatic_fraction'] = df['aromatic_rings'] / df['ring_count'].clip(lower=1)
df['size_category'] = pd.qcut(df['n_atoms'], q=5, labels=False, duplicates='drop')

GRAPH_FEATS_EXTENDED = GRAPH_FEATS + ['graph_density', 'bonds_per_atom', 'aromatic_fraction']
DOMAIN_FEATS = DESCRIPTOR_FEATS + GRAPH_FEATS_EXTENDED

print(f'  Domain features ({len(DOMAIN_FEATS)}): {DOMAIN_FEATS}')
print(f'  Fingerprint features: {len(FP_COLS)} ECFP4 bits')

# ===================================================================
# 2. TRAIN/VAL/TEST SPLIT (use OGB official split)
# ===================================================================
print('\n[2] Using OGB official scaffold split...')
train_df = df[df['split'] == 'train'].reset_index(drop=True)
val_df   = df[df['split'] == 'val'].reset_index(drop=True)
test_df  = df[df['split'] == 'test'].reset_index(drop=True)

y_train = train_df['hiv_active'].values
y_val   = val_df['hiv_active'].values
y_test  = test_df['hiv_active'].values

print(f'  Train: {len(train_df)} (pos: {y_train.sum()}, rate: {y_train.mean():.4f})')
print(f'  Val:   {len(val_df)}  (pos: {y_val.sum()},  rate: {y_val.mean():.4f})')
print(f'  Test:  {len(test_df)}  (pos: {y_test.sum()},  rate: {y_test.mean():.4f})')

# ===================================================================
# 3. BUILD FEATURE MATRICES FOR DIFFERENT FEATURE SETS
# ===================================================================
feature_configs = {
    'domain_10': DOMAIN_FEATS,
    'fp_1024': FP_COLS,
    'combined_1034': DOMAIN_FEATS + FP_COLS,
    'graph_topo_only': GRAPH_FEATS_EXTENDED,
}

X_sets = {}
for name, cols in feature_configs.items():
    X_sets[name] = {
        'train': train_df[cols].values.astype(np.float32),
        'val':   val_df[cols].values.astype(np.float32),
        'test':  test_df[cols].values.astype(np.float32),
    }
    print(f'  Feature set "{name}": {X_sets[name]["train"].shape[1]} features')

# Scale domain features
scalers = {}
for name in ['domain_10', 'combined_1034', 'graph_topo_only']:
    sc = StandardScaler()
    X_sets[name]['train'] = sc.fit_transform(X_sets[name]['train'])
    X_sets[name]['val']   = sc.transform(X_sets[name]['val'])
    X_sets[name]['test']  = sc.transform(X_sets[name]['test'])
    scalers[name] = sc

# ===================================================================
# 4. EXPERIMENTS
# ===================================================================
print('\n[4] Running experiments...')

results = []

def run_experiment(name, model, X_tr, y_tr, X_te, y_te, features_name, n_feats):
    """Train model and evaluate on test set."""
    t0 = time.time()
    model.fit(X_tr, y_tr)
    train_time = time.time() - t0

    # Get probability predictions
    if hasattr(model, 'predict_proba'):
        y_prob = model.predict_proba(X_te)[:, 1]
    elif hasattr(model, 'decision_function'):
        y_prob = model.decision_function(X_te)
    else:
        y_prob = model.predict(X_te).astype(float)

    roc = roc_auc_score(y_te, y_prob)
    auprc = average_precision_score(y_te, y_prob)

    # Threshold at 0.5 for classification metrics
    y_pred = (y_prob >= 0.5).astype(int) if hasattr(model, 'predict_proba') else model.predict(X_te)
    f1 = f1_score(y_te, y_pred, zero_division=0)
    prec = precision_score(y_te, y_pred, zero_division=0)
    rec = recall_score(y_te, y_pred, zero_division=0)

    res = {
        'model': name, 'features': features_name, 'n_features': n_feats,
        'roc_auc': round(roc, 4), 'auprc': round(auprc, 4),
        'f1': round(f1, 4), 'precision': round(prec, 4), 'recall': round(rec, 4),
        'train_s': round(train_time, 1),
    }
    results.append(res)
    print(f'  {name:<45} ROC-AUC={roc:.4f}  AUPRC={auprc:.4f}  F1={f1:.4f}  ({train_time:.1f}s)')
    return res, y_prob

# ---- Experiment 1: Class-weight comparison on Combined features ----
print('\n--- Experiment 1: Class-weight strategies (Combined features) ---')

X_tr = X_sets['combined_1034']['train']
X_te = X_sets['combined_1034']['test']
n_f = X_tr.shape[1]

# Baseline: no class weight (Anthony's setup)
run_experiment('RF (Combined, no weight)',
               RandomForestClassifier(n_estimators=300, random_state=42, n_jobs=-1),
               X_tr, y_train, X_te, y_test, 'combined_1034', n_f)

# Class-weight balanced
run_experiment('RF (Combined, balanced)',
               RandomForestClassifier(n_estimators=300, class_weight='balanced', random_state=42, n_jobs=-1),
               X_tr, y_train, X_te, y_test, 'combined_1034', n_f)

# XGBoost with scale_pos_weight
pos_weight = (y_train == 0).sum() / max((y_train == 1).sum(), 1)
run_experiment('XGBoost (Combined, no weight)',
               xgb.XGBClassifier(n_estimators=300, learning_rate=0.05, max_depth=6,
                                  subsample=0.8, random_state=42, verbosity=0, eval_metric='logloss'),
               X_tr, y_train, X_te, y_test, 'combined_1034', n_f)

run_experiment('XGBoost (Combined, scale_pos_weight)',
               xgb.XGBClassifier(n_estimators=300, learning_rate=0.05, max_depth=6,
                                  subsample=0.8, scale_pos_weight=pos_weight,
                                  random_state=42, verbosity=0, eval_metric='logloss'),
               X_tr, y_train, X_te, y_test, 'combined_1034', n_f)

# LightGBM with is_unbalance
run_experiment('LightGBM (Combined, no weight)',
               lgbm.LGBMClassifier(n_estimators=300, learning_rate=0.05, max_depth=6,
                                    subsample=0.8, random_state=42, verbose=-1),
               X_tr, y_train, X_te, y_test, 'combined_1034', n_f)

run_experiment('LightGBM (Combined, is_unbalance)',
               lgbm.LGBMClassifier(n_estimators=300, learning_rate=0.05, max_depth=6,
                                    subsample=0.8, is_unbalance=True, random_state=42, verbose=-1),
               X_tr, y_train, X_te, y_test, 'combined_1034', n_f)

# ---- Experiment 2: Cost-sensitive LogReg ----
print('\n--- Experiment 2: Cost-sensitive models (Domain features) ---')

X_dom_tr = X_sets['domain_10']['train']
X_dom_te = X_sets['domain_10']['test']
n_dom = X_dom_tr.shape[1]

run_experiment('LogReg (Domain 10, no weight)',
               LogisticRegression(max_iter=1000, random_state=42),
               X_dom_tr, y_train, X_dom_te, y_test, 'domain_10', n_dom)

run_experiment('LogReg (Domain 10, balanced)',
               LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42),
               X_dom_tr, y_train, X_dom_te, y_test, 'domain_10', n_dom)

# ---- Experiment 3: Graph topological features only ----
print('\n--- Experiment 3: Graph topological features only ---')

X_graph_tr = X_sets['graph_topo_only']['train']
X_graph_te = X_sets['graph_topo_only']['test']
n_graph = X_graph_tr.shape[1]

run_experiment('RF (Graph topo 5-feat)',
               RandomForestClassifier(n_estimators=300, random_state=42, n_jobs=-1),
               X_graph_tr, y_train, X_graph_te, y_test, 'graph_topo_5', n_graph)

run_experiment('XGBoost (Graph topo 5-feat)',
               xgb.XGBClassifier(n_estimators=300, learning_rate=0.05, max_depth=4,
                                  random_state=42, verbosity=0, eval_metric='logloss'),
               X_graph_tr, y_train, X_graph_te, y_test, 'graph_topo_5', n_graph)

# ---- Experiment 4: CatBoost (new model not tested by Anthony) ----
print('\n--- Experiment 4: CatBoost (new model family) ---')
try:
    from catboost import CatBoostClassifier

    run_experiment('CatBoost (Combined, auto_weight)',
                   CatBoostClassifier(iterations=300, learning_rate=0.05, depth=6,
                                       auto_class_weights='Balanced',
                                       random_seed=42, verbose=0),
                   X_tr, y_train, X_te, y_test, 'combined_1034', n_f)

    run_experiment('CatBoost (Combined, no weight)',
                   CatBoostClassifier(iterations=300, learning_rate=0.05, depth=6,
                                       random_seed=42, verbose=0),
                   X_tr, y_train, X_te, y_test, 'combined_1034', n_f)
except ImportError:
    print('  CatBoost not available, skipping')

# ---- Experiment 5: FP-only with different models ----
print('\n--- Experiment 5: FP-only baseline (1024 ECFP4 bits) ---')

X_fp_tr = X_sets['fp_1024']['train']
X_fp_te = X_sets['fp_1024']['test']
n_fp = X_fp_tr.shape[1]

run_experiment('LightGBM (FP 1024, is_unbalance)',
               lgbm.LGBMClassifier(n_estimators=300, learning_rate=0.05, max_depth=6,
                                    is_unbalance=True, random_state=42, verbose=-1),
               X_fp_tr, y_train, X_fp_te, y_test, 'fp_1024', n_fp)

# ===================================================================
# 5. RESULTS TABLE
# ===================================================================
print('\n' + '=' * 72)
print('RESULTS TABLE')
print('=' * 72)

df_res = pd.DataFrame(results).sort_values('roc_auc', ascending=False)
print(f'\n{"Rank":<5} {"Model":<50} {"ROC-AUC":<10} {"AUPRC":<10} {"F1":<8} {"Recall":<8}')
print('-' * 95)
for rank, (_, row) in enumerate(df_res.iterrows(), 1):
    print(f'{rank:<5} {row["model"]:<50} {row["roc_auc"]:<10.4f} {row["auprc"]:<10.4f} '
          f'{row["f1"]:<8.4f} {row["recall"]:<8.4f}')

# ===================================================================
# 6. KEY FINDINGS
# ===================================================================
best_row = df_res.iloc[0]
best_unweighted = df_res[~df_res['model'].str.contains('weight|unbalance|balanced', case=False)].iloc[0]
best_weighted = df_res[df_res['model'].str.contains('weight|unbalance|balanced', case=False)]
best_w_row = best_weighted.iloc[0] if len(best_weighted) > 0 else None

print('\n' + '=' * 72)
print('KEY FINDINGS')
print('=' * 72)
print(f'\n1. CHAMPION: {best_row["model"]}')
print(f'   ROC-AUC = {best_row["roc_auc"]:.4f}, AUPRC = {best_row["auprc"]:.4f}')
print(f'   vs Anthony champion (RF Combined): ROC-AUC = 0.7707')
print(f'   Delta: {best_row["roc_auc"] - 0.7707:+.4f} ROC-AUC')

if best_w_row is not None:
    print(f'\n2. BEST WEIGHTED: {best_w_row["model"]}')
    print(f'   ROC-AUC = {best_w_row["roc_auc"]:.4f}, AUPRC = {best_w_row["auprc"]:.4f}')
    print(f'   Recall = {best_w_row["recall"]:.4f} (vs unweighted recall = {best_unweighted["recall"]:.4f})')

# Graph topo results
graph_results = [r for r in results if 'Graph' in r['model']]
if graph_results:
    best_graph = max(graph_results, key=lambda r: r['roc_auc'])
    print(f'\n3. GRAPH TOPOLOGY ONLY: {best_graph["model"]}')
    print(f'   ROC-AUC = {best_graph["roc_auc"]:.4f} from just {best_graph["n_features"]} features')
    print(f'   Gap vs Combined: {best_graph["roc_auc"] - best_row["roc_auc"]:+.4f}')

# ===================================================================
# 7. PLOTS
# ===================================================================
print('\n[7] Generating plots...')

fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
fig.patch.set_facecolor('#0D1117')

# Plot 1: ROC-AUC comparison — weighted vs unweighted
ax = axes[0]
ax.set_facecolor('#161B22')

# Color by weight strategy
colors = []
for _, row in df_res.iterrows():
    if 'balanced' in row['model'].lower() or 'unbalance' in row['model'].lower() or 'weight' in row['model'].lower():
        colors.append('#FF5722')  # weighted = orange-red
    elif 'Graph' in row['model']:
        colors.append('#9C27B0')  # graph = purple
    else:
        colors.append('#2196F3')  # unweighted = blue

bars = ax.barh(range(len(df_res)), df_res['roc_auc'].values, color=colors, alpha=0.85, edgecolor='#30363D')
ax.axvline(x=0.7707, color='#4CAF50', linestyle='--', linewidth=2, label='Anthony RF champion (0.7707)')
ax.axvline(x=0.8476, color='#FFD700', linestyle=':', linewidth=2, label='OGB SOTA (0.8476)')
ax.set_yticks(range(len(df_res)))
ax.set_yticklabels([m[:40] for m in df_res['model'].values], fontsize=8, color='white')
ax.set_xlabel('ROC-AUC (higher = better)', color='white')
ax.set_title('ogbg-molhiv: Weighted vs Unweighted Models\n(Mark Phase 1 Experiment)', color='white', fontsize=10)
ax.tick_params(colors='white')
ax.spines['bottom'].set_color('#444')
ax.spines['left'].set_color('#444')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.set_xlim(0.55, 0.90)

from matplotlib.patches import Patch
legend_elems = [
    Patch(facecolor='#2196F3', label='Unweighted'),
    Patch(facecolor='#FF5722', label='Class-weighted'),
    Patch(facecolor='#9C27B0', label='Graph topo only'),
    plt.Line2D([0], [0], color='#4CAF50', linestyle='--', linewidth=2, label='Anthony RF (0.7707)'),
    plt.Line2D([0], [0], color='#FFD700', linestyle=':', linewidth=2, label='OGB SOTA (0.8476)'),
]
ax.legend(handles=legend_elems, fontsize=7, facecolor='#161B22', labelcolor='white', loc='lower right')

# Plot 2: AUPRC (secondary metric for imbalanced data)
ax2 = axes[1]
ax2.set_facecolor('#161B22')
bars2 = ax2.barh(range(len(df_res)), df_res['auprc'].values, color=colors, alpha=0.85, edgecolor='#30363D')
ax2.axvline(x=0.3722, color='#4CAF50', linestyle='--', linewidth=2, label='Anthony RF AUPRC (0.3722)')
ax2.axvline(x=0.0351, color='#F44336', linestyle=':', linewidth=1.5, label='Random baseline (0.035)')
ax2.set_yticks(range(len(df_res)))
ax2.set_yticklabels([m[:40] for m in df_res['model'].values], fontsize=8, color='white')
ax2.set_xlabel('AUPRC (higher = better)', color='white')
ax2.set_title('Precision-Recall AUC (critical for 3.5% positive rate)\nHigher = better minority class detection', color='white', fontsize=10)
ax2.tick_params(colors='white')
ax2.spines['bottom'].set_color('#444')
ax2.spines['left'].set_color('#444')
ax2.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)
legend_elems2 = [
    plt.Line2D([0], [0], color='#4CAF50', linestyle='--', linewidth=2, label='Anthony RF AUPRC (0.3722)'),
    plt.Line2D([0], [0], color='#F44336', linestyle=':', linewidth=1.5, label=f'Random baseline ({y_test.mean():.3f})'),
]
ax2.legend(handles=legend_elems2, fontsize=8, facecolor='#161B22', labelcolor='white', loc='lower right')

plt.tight_layout(pad=2.0)
plt.savefig(RESULTS_DIR / 'phase1_mark_imbalance_comparison.png', dpi=150, bbox_inches='tight', facecolor='#0D1117')
plt.close()
print('  Saved: results/phase1_mark_imbalance_comparison.png')

# ===================================================================
# 8. SAVE RESULTS
# ===================================================================
print('\n[8] Saving results...')

metrics_path = RESULTS_DIR / 'metrics.json'
try:
    with open(metrics_path) as f:
        existing = json.load(f)
except (FileNotFoundError, json.JSONDecodeError):
    existing = []

# Ensure existing is a list
if isinstance(existing, dict):
    existing = [existing]

mark_entry = {
    'phase': 1,
    'author': 'Mark',
    'date': '2026-04-06',
    'dataset': 'ogbg-molhiv (OGB) - 41,127 molecules',
    'primary_metric': 'ROC-AUC',
    'split': 'OGB official scaffold split',
    'experiments': results,
    'key_findings': {
        'champion_model': str(best_row['model']),
        'champion_roc_auc': float(best_row['roc_auc']),
        'champion_auprc': float(best_row['auprc']),
        'anthony_champion_roc_auc': 0.7707,
        'delta_vs_anthony': float(best_row['roc_auc'] - 0.7707),
        'ogb_sota': 0.8476,
    },
}
existing.append(mark_entry)
with open(metrics_path, 'w') as f:
    json.dump(existing, f, indent=2)
print('  Updated results/metrics.json')

# Save experiment log
exp_log_path = RESULTS_DIR / 'EXPERIMENT_LOG.md'
with open(exp_log_path, 'a', encoding='utf-8') as f:
    f.write('\n\n## 2026-04-06 | Phase 1 (Mark) | Class-Imbalance Strategies + Graph Features\n\n')
    f.write('| Rank | Model | Features | ROC-AUC | AUPRC | F1 | Recall |\n')
    f.write('|------|-------|----------|---------|-------|----|---------|\n')
    for rank, (_, row) in enumerate(df_res.iterrows(), 1):
        f.write(f'| {rank} | {row["model"]} | {row["features"]} | {row["roc_auc"]} | {row["auprc"]} | {row["f1"]} | {row["recall"]} |\n')
print('  Updated results/EXPERIMENT_LOG.md')

print('\n' + '=' * 72)
print('PHASE 1 (Mark) COMPLETE')
print('=' * 72)
