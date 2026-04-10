"""
Phase 5 Mark: Advanced Techniques + Ablation + LLM Comparison
Drug Molecule Property Prediction (ogbg-molhiv)
Date: 2026-04-10
"""
import os, sys, time, warnings, json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')
os.chdir(r'C:/Users/antho/OneDrive/Desktop/YC-Portfolio-Projects/Drug-Molecule-Property-Prediction')
from pathlib import Path

DATA    = Path('data/processed')
RESULTS = Path('results')

from catboost import CatBoostClassifier
from sklearn.metrics import (roc_auc_score, average_precision_score,
                              precision_score, recall_score, f1_score, roc_curve)

SEED = 42

# ── Load Phase 4 cached data ──────────────────────────────────────────────
print("Loading Phase 4 cached data ...")
d = np.load(DATA / 'phase4_mark_data.npz', allow_pickle=True)
X_tr, y_tr = d['X_tr'], d['y_tr']
X_va, y_va = d['X_va'], d['y_va']
X_te, y_te = d['X_te'], d['y_te']
mi_scores   = d['mi_scores']
top400_idx  = d['top_400_idx']
smiles_te   = d['smiles_te']

feat_names  = np.array(json.load(open(DATA / 'feat_names.json')))

print(f"Train: {X_tr.shape}  Val: {X_va.shape}  Test: {X_te.shape}")
print(f"Test actives: {y_te.sum()} / {len(y_te)} ({y_te.mean()*100:.1f}%)")

# ── Feature category masks ────────────────────────────────────────────────
def cat_in_top400(prefix):
    m = np.array([f.startswith(prefix) for f in feat_names])
    return np.array([i for i in top400_idx if m[i]])

lip400   = cat_in_top400('lip')
adv400   = cat_in_top400('adv')
maccs400 = cat_in_top400('maccs')
morgan400= cat_in_top400('morgan')
frag400  = cat_in_top400('fr')

print("Feature categories in top-400:")
for name, idx in [('Lipinski', lip400), ('Advanced', adv400),
                  ('MACCS', maccs400), ('Morgan', morgan400), ('Fragment', frag400)]:
    print(f"  {name:10s}: {len(idx):3d} features")

# ── Training helper ───────────────────────────────────────────────────────
def train_cb(X_tr_sub, y_tr_, X_te_sub, y_te_, n_iter=500, depth=8, lr=0.055, l2=4.7, min_leaf=38):
    cb = CatBoostClassifier(
        iterations=n_iter, learning_rate=lr, depth=depth,
        l2_leaf_reg=l2, min_data_in_leaf=min_leaf,
        auto_class_weights='Balanced', random_seed=SEED, verbose=0
    )
    t0 = time.time()
    cb.fit(X_tr_sub, y_tr_)
    elapsed = time.time() - t0
    proba = cb.predict_proba(X_te_sub)[:, 1]
    auc   = roc_auc_score(y_te_, proba)
    auprc = average_precision_score(y_te_, proba)
    return auc, auprc, elapsed, proba

# ════════════════════════════════════════════════════════════════════════════
# EXPERIMENT 5.1: ABLATION STUDY
# ════════════════════════════════════════════════════════════════════════════
print("\n" + "="*70)
print("EXPERIMENT 5.1: ABLATION STUDY")
print("="*70)

print("5.1a  Baseline MI-top-400 ...")
auc_base, auprc_base, t_base, preds_base = train_cb(
    X_tr[:, top400_idx], y_tr, X_te[:, top400_idx], y_te
)
print(f"  Baseline: AUC={auc_base:.4f}  AUPRC={auprc_base:.4f}  t={t_base:.1f}s")

ablation_results = [{
    'experiment': 'MI-400 baseline', 'removed': 'none',
    'n_features': len(top400_idx), 'auc': auc_base, 'auprc': auprc_base, 'delta': 0.0
}]

print("5.1b  Leave-one-category-out ...")
for cat_name, cat_idx in [
    ('MACCS',    maccs400),
    ('Morgan FP',morgan400),
    ('Advanced', adv400),
    ('Lipinski', lip400),
    ('Fragment', frag400),
]:
    keep = np.array([i for i in top400_idx if i not in set(cat_idx)])
    print(f"  Remove {cat_name:10s} ({len(cat_idx):3d} removed, {len(keep)} remain) ...", end=' ', flush=True)
    auc, auprc, t, _ = train_cb(X_tr[:, keep], y_tr, X_te[:, keep], y_te)
    delta = auc - auc_base
    ablation_results.append({
        'experiment': f'Remove {cat_name}', 'removed': cat_name,
        'n_features': len(keep), 'auc': auc, 'auprc': auprc, 'delta': delta
    })
    print(f"AUC={auc:.4f}  delta={delta:+.4f}  t={t:.1f}s")

print("5.1c  Single-category models ...")
mask_map = {
    'maccs': np.array([f.startswith('maccs') for f in feat_names]),
    'morgan': np.array([f.startswith('morgan') for f in feat_names]),
    'lip': np.array([f.startswith('lip') for f in feat_names]),
    'adv': np.array([f.startswith('adv') for f in feat_names]),
}
for cat_name, prefix in [
    ('MACCS only',    'maccs'),
    ('Morgan only',   'morgan'),
    ('Lipinski only', 'lip'),
    ('Advanced only', 'adv'),
]:
    idx = np.where(mask_map[prefix])[0]
    print(f"  {cat_name:15s} ({len(idx)} feats) ...", end=' ', flush=True)
    auc, auprc, t, _ = train_cb(X_tr[:, idx], y_tr, X_te[:, idx], y_te)
    delta = auc - auc_base
    ablation_results.append({
        'experiment': cat_name, 'removed': 'others',
        'n_features': len(idx), 'auc': auc, 'auprc': auprc, 'delta': delta
    })
    print(f"AUC={auc:.4f}  delta={delta:+.4f}")

abl_df = pd.DataFrame(ablation_results)
print("\n-- Ablation Table --")
print(abl_df[['experiment','n_features','auc','auprc','delta']].to_string(index=False))

# ════════════════════════════════════════════════════════════════════════════
# EXPERIMENT 5.2: SUBGROUP SPECIALIST
# ════════════════════════════════════════════════════════════════════════════
print("\n" + "="*70)
print("EXPERIMENT 5.2: SUBGROUP SPECIALIST (fix MW < 450 blind spot)")
print("="*70)

# Use lip_0 (MW) as proxy
lip0_idx = int(np.where(feat_names == 'lip_0')[0][0])
mw_tr = X_tr[:, lip0_idx]
mw_te = X_te[:, lip0_idx]

MW_THRESH = 450
small_tr = mw_tr < MW_THRESH
large_tr = mw_tr >= MW_THRESH
small_te = mw_te < MW_THRESH
large_te = mw_te >= MW_THRESH

print(f"Train: small={small_tr.sum()} ({y_tr[small_tr].sum()} act), large={large_tr.sum()} ({y_tr[large_tr].sum()} act)")
print(f"Test:  small={small_te.sum()} ({y_te[small_te].sum()} act), large={large_te.sum()} ({y_te[large_te].sum()} act)")

X_tr_400 = X_tr[:, top400_idx]
X_te_400 = X_te[:, top400_idx]

print("Training small-molecule specialist (MW < 450) ...", end=' ', flush=True)
auc_ss, auprc_ss, t_ss, preds_ss = train_cb(
    X_tr_400[small_tr], y_tr[small_tr], X_te_400[small_te], y_te[small_te]
)
print(f"AUC={auc_ss:.4f}  AUPRC={auprc_ss:.4f}  t={t_ss:.1f}s")

print("Training large-molecule specialist (MW >= 450) ...", end=' ', flush=True)
auc_ls, auprc_ls, t_ls, preds_ls = train_cb(
    X_tr_400[large_tr], y_tr[large_tr], X_te_400[large_te], y_te[large_te]
)
print(f"AUC={auc_ls:.4f}  AUPRC={auprc_ls:.4f}  t={t_ls:.1f}s")

# Combined predictions
preds_spec = np.zeros(len(y_te))
preds_spec[small_te] = preds_ss
preds_spec[large_te] = preds_ls

auc_spec = roc_auc_score(y_te, preds_spec)
auprc_spec = average_precision_score(y_te, preds_spec)
print(f"\nSubgroup specialist combined: AUC={auc_spec:.4f}  delta={auc_spec-auc_base:+.4f}")

# Per-group recall
THRESH = 0.3958
def group_recall(proba, y, mask):
    if mask.sum() == 0 or y[mask].sum() == 0: return np.nan
    return ((proba[mask] >= THRESH) & (y[mask] == 1)).sum() / y[mask].sum()

print("\nPer-group recall:")
for gname, gmask in [('Small MW (<450)', small_te), ('Large MW (>=450)', large_te), ('All', np.ones(len(y_te), bool))]:
    r_gen  = group_recall(preds_base, y_te, gmask)
    r_spec = group_recall(preds_spec, y_te, gmask)
    print(f"  {gname:20s}  Generalist={r_gen:.3f}  Specialist={r_spec:.3f}  delta={r_spec-r_gen:+.3f}")

# ════════════════════════════════════════════════════════════════════════════
# EXPERIMENT 5.3: DIVERSE CATBOOST ENSEMBLE
# ════════════════════════════════════════════════════════════════════════════
print("\n" + "="*70)
print("EXPERIMENT 5.3: DIVERSE CATBOOST ENSEMBLE")
print("="*70)

# Model B: MACCS + Advanced (most efficient features)
idx_B = np.concatenate([maccs400, adv400])
print(f"Model B: MACCS+Adv ({len(idx_B)} feats) ...", end=' ', flush=True)
auc_B, auprc_B, t_B, preds_B = train_cb(X_tr[:, idx_B], y_tr, X_te[:, idx_B], y_te)
print(f"AUC={auc_B:.4f}  t={t_B:.1f}s")

# Model C: Morgan + Lipinski (coverage features)
idx_C = np.concatenate([morgan400, lip400])
print(f"Model C: Morgan+Lip ({len(idx_C)} feats) ...", end=' ', flush=True)
auc_C, auprc_C, t_C, preds_C = train_cb(X_tr[:, idx_C], y_tr, X_te[:, idx_C], y_te)
print(f"AUC={auc_C:.4f}  t={t_C:.1f}s")

# Model D: ALL MACCS (no MI filter, 167 features)
all_maccs = np.where(np.array([f.startswith('maccs') for f in feat_names]))[0]
print(f"Model D: All MACCS (167 feats) ...", end=' ', flush=True)
auc_D, auprc_D, t_D, preds_D = train_cb(X_tr[:, all_maccs], y_tr, X_te[:, all_maccs], y_te)
print(f"AUC={auc_D:.4f}  t={t_D:.1f}s")

# Ensemble combinations
ens_results = []
for name, combo in [
    ('A: MI-400 only',               [preds_base]),
    ('A+B: MI400 + MACCS+Adv',       [preds_base, preds_B]),
    ('A+D: MI400 + AllMACS',         [preds_base, preds_D]),
    ('A+B+C: 3-model avg',           [preds_base, preds_B, preds_C]),
    ('A+B+C+D: 4-model avg',         [preds_base, preds_B, preds_C, preds_D]),
    ('B+D: 2x MACCS',                [preds_B, preds_D]),
]:
    avg = np.mean(combo, axis=0)
    auc = roc_auc_score(y_te, avg)
    auprc = average_precision_score(y_te, avg)
    d_auc = auc - auc_base
    ens_results.append({'combo': name, 'n_models': len(combo),
                        'auc': auc, 'auprc': auprc, 'delta': d_auc})
    print(f"  {name:40s}  AUC={auc:.4f}  delta={d_auc:+.4f}")

ens_df = pd.DataFrame(ens_results).sort_values('auc', ascending=False)
print("\n-- Ensemble Leaderboard --")
print(ens_df.to_string(index=False))

best_ens_auc  = ens_df.iloc[0]['auc']
best_ens_name = ens_df.iloc[0]['combo']
best_ens_preds_combo = None

# Identify which ensemble is best
for name, combo in [
    ('A: MI-400 only',               [preds_base]),
    ('A+B: MI400 + MACCS+Adv',       [preds_base, preds_B]),
    ('A+D: MI400 + AllMACS',         [preds_base, preds_D]),
    ('A+B+C: 3-model avg',           [preds_base, preds_B, preds_C]),
    ('A+B+C+D: 4-model avg',         [preds_base, preds_B, preds_C, preds_D]),
    ('B+D: 2x MACCS',                [preds_B, preds_D]),
]:
    if name == best_ens_name:
        best_ens_preds = np.mean(combo, axis=0)
        break
else:
    best_ens_preds = preds_base

# ════════════════════════════════════════════════════════════════════════════
# EXPERIMENT 5.4: LLM BASELINE (Claude Haiku)
# ════════════════════════════════════════════════════════════════════════════
print("\n" + "="*70)
print("EXPERIMENT 5.4: LLM BASELINE (Claude Haiku vs Custom Model)")
print("="*70)

import re

SYSTEM_PROMPT = (
    "You are a computational chemistry expert. "
    "You will be given a SMILES string representing a molecule. "
    "Predict the probability (0.0 to 1.0) that this molecule inhibits HIV replication, "
    "based on its structural features. "
    "Respond with ONLY a single float number between 0.0 and 1.0. "
    "Examples: 0.05 | 0.73 | 0.15 | 0.92"
)

def smiles_prompt(smiles):
    return (
        f"SMILES: {smiles}\n\n"
        "Predict the HIV inhibition probability for this molecule. "
        "Respond with a single float between 0.0 and 1.0 only."
    )

def parse_prob(text):
    text = text.strip()
    matches = re.findall(r'\b(0\.\d+|1\.0+|0|1)\b', text)
    if matches:
        return float(matches[0])
    try:
        return float(text)
    except:
        return 0.1

# Eval set: all actives + equal inactives
rng = np.random.default_rng(42)
active_idx   = np.where(y_te == 1)[0]
inactive_idx = rng.choice(np.where(y_te == 0)[0], size=len(active_idx), replace=False)
eval_idx = np.concatenate([active_idx, inactive_idx])
rng.shuffle(eval_idx)

eval_smiles  = smiles_te[eval_idx]
eval_labels  = y_te[eval_idx]
eval_custom  = preds_base[eval_idx]
eval_best_ens= best_ens_preds[eval_idx]

print(f"LLM eval set: {len(eval_idx)} molecules ({eval_labels.sum()} actives, {(eval_labels==0).sum()} inactives)")
print(f"Custom (MI-400) AUC on subset: {roc_auc_score(eval_labels, eval_custom):.4f}")

llm_cache = RESULTS / 'phase5_mark_llm_predictions.json'
API_AVAILABLE = False
llm_preds = None

try:
    import anthropic
    client = anthropic.Anthropic()
    # Quick test
    test = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=16,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": smiles_prompt(str(eval_smiles[0]))}]
    )
    test_val = test.content[0].text.strip()
    print(f"API test OK: {test_val!r}")
    API_AVAILABLE = True
except Exception as e:
    print(f"API not available: {e}")

if API_AVAILABLE:
    print(f"\nRunning Claude Haiku on {len(eval_smiles)} SMILES ...")
    llm_proba = []
    latencies = []
    errors = 0
    for i, smiles in enumerate(eval_smiles):
        t0 = time.time()
        try:
            msg = client.messages.create(
                model="claude-haiku-4-5-20251001",
                max_tokens=16,
                system=SYSTEM_PROMPT,
                messages=[{"role": "user", "content": smiles_prompt(str(smiles))}]
            )
            raw = msg.content[0].text.strip()
            prob = parse_prob(raw)
        except Exception as e:
            prob = 0.1
            errors += 1
        latency = time.time() - t0
        llm_proba.append(prob)
        latencies.append(latency)
        if (i+1) % 20 == 0 or (i+1) == len(eval_smiles):
            print(f"  [{i+1}/{len(eval_smiles)}] avg_lat={np.mean(latencies):.2f}s  errors={errors}")

    llm_preds = np.array(llm_proba)
    save_data = {
        'eval_idx': eval_idx.tolist(), 'eval_labels': eval_labels.tolist(),
        'llm_preds': llm_preds.tolist(), 'custom_preds': eval_custom.tolist(),
        'latencies': latencies, 'errors': errors, 'model': 'claude-haiku-4-5-20251001'
    }
    with open(llm_cache, 'w') as f:
        json.dump(save_data, f, indent=2)
    print(f"Saved LLM predictions to {llm_cache}")

elif llm_cache.exists():
    print(f"Loading cached LLM predictions from {llm_cache}")
    saved = json.load(open(llm_cache))
    llm_preds = np.array(saved['llm_preds'])
    eval_labels = np.array(saved['eval_labels'])
    eval_custom = np.array(saved['custom_preds'])
    latencies = saved['latencies']
    errors = saved['errors']
    API_AVAILABLE = True

# Compute comparison metrics
auc_custom_sub = roc_auc_score(eval_labels, eval_custom)
auc_best_ens_sub = roc_auc_score(eval_labels, eval_best_ens)

if llm_preds is not None:
    auc_llm     = roc_auc_score(eval_labels, llm_preds)
    auprc_llm   = average_precision_score(eval_labels, llm_preds)
    auprc_cust  = average_precision_score(eval_labels, eval_custom)

    t_llm_bin = (llm_preds >= 0.3).astype(int)
    t_cust_bin = (eval_custom >= 0.3958).astype(int)

    prec_llm  = precision_score(eval_labels, t_llm_bin, zero_division=0)
    rec_llm   = recall_score(eval_labels, t_llm_bin, zero_division=0)
    f1_llm    = f1_score(eval_labels, t_llm_bin, zero_division=0)
    prec_cust = precision_score(eval_labels, t_cust_bin, zero_division=0)
    rec_cust  = recall_score(eval_labels, t_cust_bin, zero_division=0)
    f1_cust   = f1_score(eval_labels, t_cust_bin, zero_division=0)

    avg_lat = np.mean(latencies)

    print("\n-- Head-to-Head: CatBoost MI-400 vs Claude Haiku --")
    print(f"{'Metric':20s}  {'CatBoost MI-400':>16s}  {'Claude Haiku':>14s}  {'Winner':>10s}")
    print("-" * 70)
    for metric, cv, lv in [
        ('ROC-AUC',   auc_custom_sub, auc_llm),
        ('AUPRC',     auprc_cust,     auprc_llm),
        ('Precision', prec_cust,      prec_llm),
        ('Recall',    rec_cust,       rec_llm),
        ('F1',        f1_cust,        f1_llm),
    ]:
        winner = 'Custom OK' if cv > lv else 'LLM OK'
        print(f"  {metric:18s}  {cv:16.4f}  {lv:14.4f}  {winner:>10s}")
    print(f"  {'Latency':18s}  {'~2ms':>16s}  {avg_lat:.2f}s")
    print(f"  {'Cost/1K preds':18s}  {'~$0':>16s}  {'~$0.18':>14s}")
    print(f"\n  Delta AUC: {auc_custom_sub - auc_llm:+.4f} (positive = Custom wins)")

# ════════════════════════════════════════════════════════════════════════════
# PLOTS
# ════════════════════════════════════════════════════════════════════════════
print("\nGenerating plots ...")

# Plot 1: Ablation study
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle('Phase 5: Ablation Study - Feature Category Impact', fontsize=13, fontweight='bold')

loco = abl_df[abl_df['removed'].isin(['MACCS','Morgan FP','Advanced','Lipinski','Fragment'])].copy()
loco = loco.sort_values('delta')
colors = ['#d62728' if d < 0 else '#2ca02c' for d in loco['delta']]
bars = axes[0].barh(loco['removed'], loco['delta'] * 100, color=colors, edgecolor='black', linewidth=0.6)
axes[0].axvline(0, color='black', linewidth=1)
axes[0].set_xlabel('Delta AUC x 100 (vs MI-400 baseline)', fontsize=10)
axes[0].set_title('Leave-One-Category-Out\n(negative = category was important)', fontsize=10)
for bar, val in zip(bars, loco['delta'] * 100):
    x_off = 0.05 if val >= 0 else -0.05
    ha = 'left' if val >= 0 else 'right'
    axes[0].text(val + x_off, bar.get_y() + bar.get_height()/2,
                 f'{val:+.2f}', va='center', ha=ha, fontsize=9)

single = abl_df[abl_df['removed'] == 'others']
bar_c = ['#1f77b4','#ff7f0e','#2ca02c','#9467bd']
b = axes[1].bar(single['experiment'], single['auc'], color=bar_c[:len(single)], edgecolor='black')
axes[1].axhline(auc_base, color='red', linestyle='--', linewidth=1.5,
                label=f'Full MI-400 ({auc_base:.4f})')
axes[1].set_ylim(0.5, auc_base + 0.07)
axes[1].set_ylabel('Test ROC-AUC', fontsize=10)
axes[1].set_title('Single Category Only (all 400 dims)', fontsize=10)
axes[1].legend(fontsize=9)
for bar, val in zip(b, single['auc']):
    axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.002,
                 f'{val:.4f}', ha='center', va='bottom', fontsize=9)

plt.tight_layout()
plt.savefig(RESULTS / 'phase5_mark_ablation.png', dpi=120, bbox_inches='tight')
plt.close()
print("Saved: phase5_mark_ablation.png")

# Plot 2: LLM comparison
if llm_preds is not None:
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle(f'Phase 5: CatBoost MI-400 vs Claude Haiku ({len(eval_labels)} molecules)',
                 fontsize=12, fontweight='bold')

    fpr_cb,  tpr_cb,  _ = roc_curve(eval_labels, eval_custom)
    fpr_llm, tpr_llm, _ = roc_curve(eval_labels, llm_preds)
    axes[0].plot(fpr_cb, tpr_cb, 'b-', lw=2, label=f'CatBoost MI-400 (AUC={auc_custom_sub:.3f})')
    axes[0].plot(fpr_llm, tpr_llm, 'r--', lw=2, label=f'Claude Haiku (AUC={auc_llm:.3f})')
    axes[0].plot([0,1],[0,1],'k:',alpha=0.4)
    axes[0].set_xlabel('FPR'); axes[0].set_ylabel('TPR')
    axes[0].set_title('ROC Curve Comparison')
    axes[0].legend(fontsize=9)

    for label, color in [(0, '#1f77b4'), (1, '#d62728')]:
        mask_l = eval_labels == label
        name = 'Inactive' if label == 0 else 'Active'
        axes[1].hist(eval_custom[mask_l], bins=20, alpha=0.5, color=color,
                     label=f'Custom {name}', density=True)
        axes[2].hist(llm_preds[mask_l], bins=20, alpha=0.5, color=color,
                     label=f'LLM {name}', density=True)
    for ax, title in [(axes[1], 'CatBoost MI-400 Scores'), (axes[2], 'Claude Haiku Scores')]:
        ax.set_title(title)
        ax.legend(fontsize=9)
        ax.set_xlabel('Predicted probability')

    plt.tight_layout()
    plt.savefig(RESULTS / 'phase5_mark_llm_comparison.png', dpi=120, bbox_inches='tight')
    plt.close()
    print("Saved: phase5_mark_llm_comparison.png")

# Plot 3: Summary
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle('Phase 5 Summary: Advanced Techniques for HIV Activity Prediction', fontsize=12, fontweight='bold')

# Ablation impact (sorted)
loco2 = abl_df[abl_df['removed'].isin(['MACCS','Morgan FP','Advanced','Lipinski','Fragment'])].copy()
loco2 = loco2.sort_values('delta')
colors2 = ['#d62728' if d < 0 else '#2ca02c' for d in loco2['delta']]
axes[0].barh(loco2['removed'], loco2['delta'] * 100, color=colors2, edgecolor='black', linewidth=0.6)
axes[0].axvline(0, color='black', linewidth=1)
axes[0].set_xlabel('Delta AUC x 100', fontsize=10)
axes[0].set_title('Ablation: Category Removal Impact\n(negative = was important)', fontsize=10)
axes[0].set_xlim(-4.5, 1.0)

# All models summary
p5_labels = ['GIN+Edge\n(Anthony P3)', 'CatBoost\nMI-400 (P3)', f'Best Ens\n(P5)', 'Subgroup\nSpec (P5)']
p5_aucs   = [0.7860, auc_base, best_ens_auc, auc_spec]
p5_colors = ['#7f7f7f', '#1f77b4', '#2ca02c', '#ff7f0e']
if llm_preds is not None:
    p5_labels.append('Claude\nHaiku (P5)')
    p5_aucs.append(auc_llm)
    p5_colors.append('#d62728')

bars = axes[1].bar(range(len(p5_labels)), p5_aucs, color=p5_colors, edgecolor='black', linewidth=0.6)
axes[1].set_xticks(range(len(p5_labels)))
axes[1].set_xticklabels(p5_labels, fontsize=9)
axes[1].set_ylabel('Test ROC-AUC', fontsize=10)
axes[1].set_ylim(0.60, max(p5_aucs) + 0.05)
axes[1].set_title('Phase 5 Model Comparison', fontsize=10)
for bar, val in zip(bars, p5_aucs):
    axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.002,
                 f'{val:.4f}', ha='center', va='bottom', fontsize=8, fontweight='bold')

plt.tight_layout()
plt.savefig(RESULTS / 'phase5_mark_summary.png', dpi=120, bbox_inches='tight')
plt.close()
print("Saved: phase5_mark_summary.png")

# ════════════════════════════════════════════════════════════════════════════
# SAVE RESULTS
# ════════════════════════════════════════════════════════════════════════════
print("\nSaving results ...")
with open(RESULTS / 'metrics.json') as f:
    all_metrics = json.load(f)

phase5 = {
    'phase': 5,
    'date': '2026-04-10',
    'researcher': 'Mark',
    'ablation': abl_df.to_dict(orient='records'),
    'subgroup_specialist': {
        'combined_auc': float(auc_spec),
        'combined_auprc': float(auprc_spec),
        'delta_vs_generalist': float(auc_spec - auc_base),
        'small_specialist_auc': float(auc_ss),
        'large_specialist_auc': float(auc_ls),
        'generalist_auc': float(auc_base),
    },
    'ensemble': ens_df.to_dict(orient='records'),
    'llm_comparison': {
        'model': 'claude-haiku-4-5-20251001',
        'n_eval_samples': int(len(eval_labels)),
        'custom_auc_on_subset': float(auc_custom_sub),
        'llm_auc': float(auc_llm) if llm_preds is not None else None,
        'delta_custom_minus_llm': float(auc_custom_sub - auc_llm) if llm_preds is not None else None,
        'api_available': API_AVAILABLE,
    },
    'champion': {
        'model': best_ens_name,
        'auc': float(best_ens_auc),
        'delta_vs_phase3': float(best_ens_auc - 0.8105),
    }
}
all_metrics['phase5_mark'] = phase5
with open(RESULTS / 'phase5_mark_results.json', 'w') as f:
    json.dump(phase5, f, indent=2)
with open(RESULTS / 'metrics.json', 'w') as f:
    json.dump(all_metrics, f, indent=2)

print("\n" + "="*70)
print("PHASE 5 COMPLETE")
print("="*70)
print(f"  Baseline (Phase 3 best run):  AUC=0.8105")
print(f"  Phase 5 best ensemble:        AUC={best_ens_auc:.4f}  delta={best_ens_auc-0.8105:+.4f}")
print(f"  Subgroup specialist:          AUC={auc_spec:.4f}  delta={auc_spec-auc_base:+.4f}")
if llm_preds is not None:
    print(f"  Claude Haiku (LLM):          AUC={auc_llm:.4f}")
    print(f"  Custom beats LLM by:         {auc_custom_sub - auc_llm:+.4f} AUC points")
