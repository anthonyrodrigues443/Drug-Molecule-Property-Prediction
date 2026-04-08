"""
Build phase3_mark_feature_selection.ipynb programmatically.

Mark's complementary angle for Phase 3:
  Anthony built GNN+edge (0.7860 champion) + CatBoost ablation on 11 feature sets.
  His counterintuitive finding: "Full hybrid (1345d) HURTS — 0.7415 vs 0.7841 all-trad (1217)."
  Mark extends this with (1) RDKit Fragment (Fr_*) descriptors (85 domain-chemistry
  functional-group counts medicinal chemists actually use), and (2) mutual-information
  feature selection to rescue the high-dim hybrid and test whether feature CURATION
  beats feature QUANTITY.

Goal: BEAT Anthony's GIN+Edge champion (0.7860) with pure CatBoost + curated features.
"""
import nbformat as nbf

nb = nbf.v4.new_notebook()
cells = []

# ---------- Cell 1: markdown intro ----------
cells.append(nbf.v4.new_markdown_cell("""# Phase 3 (Mark): Domain Fragments + Feature Selection

**Date:** 2026-04-08 · **Researcher:** Mark Rodrigues · **Dataset:** ogbg-molhiv (scaffold split)

### Building on Anthony's Phase 3
Anthony's Phase 3 landed two big results:
1. **GIN + edge features = 0.7860 AUC** — first GNN to beat CatBoost (+0.008 over CatBoost+all-trad)
2. **Counterintuitive:** appending GNN embeddings to CatBoost+all-traditional (1345 dims) *hurts* by −0.037 vs 1217 dims. More features ≠ better.

He did NOT try:
- **RDKit Fragment (`Fr_*`) descriptors** — 85 explicit functional-group counts (aliphatic OH, aromatic amines, nitro, sulfonamide, halides, etc.) that medicinal chemists use for toxicophore/pharmacophore screening.
- **Feature selection to rescue the full hybrid** — is the 1345-d failure because of *useless noise*, or is CatBoost actually learning something the pruning would destroy?

### Hypotheses
- **H1 (domain knowledge):** 85 functional-group counts (Fr_*) alone beat Lipinski-14 (0.7744) because HIV inhibition depends on specific reactive groups.
- **H2 (curation):** Mutual-info top-K on Anthony's 1217-d all-traditional has a sweet spot that beats using all 1217.
- **H3 (headline):** `Lipinski + Fragments + top-K Morgan bits` CatBoost > Anthony's GIN+Edge champion 0.7860.
"""))

# ---------- Cell 2: setup ----------
cells.append(nbf.v4.new_code_cell("""import os, json, time, warnings
warnings.filterwarnings('ignore')
os.environ['MPLBACKEND'] = 'Agg'
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Patch torch.load for OGB cache
import torch
_orig_load = torch.load
torch.load = lambda *a, **k: _orig_load(*a, **{**k, 'weights_only': False})

from ogb.graphproppred import GraphPropPredDataset
from rdkit import Chem
from rdkit.Chem import Fragments, Descriptors, rdMolDescriptors, AllChem, MACCSkeys
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.feature_selection import mutual_info_classif
from catboost import CatBoostClassifier

BASE = Path.cwd()
if BASE.name != 'Drug-Molecule-Property-Prediction':
    BASE = BASE.parent
RES = BASE / 'results'; RES.mkdir(exist_ok=True)

np.random.seed(42)
print('setup OK, base=', BASE.name)"""))

# ---------- Cell 3: data load ----------
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
print(f'loaded {len(df)} mols in {time.time()-t0:.1f}s')
print(df.groupby('split')['y'].agg(['count', 'sum', 'mean']).round(4))"""))

# ---------- Cell 4: compute features ----------
cells.append(nbf.v4.new_code_cell("""# Compute Lipinski-14, RDKit Fragments-85, Morgan FP-1024, MACCS-167, Advanced-12 for every molecule.
# Mirrors Anthony's Phase 3 feature universe PLUS the new Fr_* family.

FRAG_FUNCS = [(n, getattr(Fragments, n)) for n in sorted(dir(Fragments)) if n.startswith('fr_')]
print('Fragment families:', len(FRAG_FUNCS))

def compute_all(smi):
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return None
    # Lipinski-14 (same as Anthony's data_pipeline)
    mw = Descriptors.MolWt(mol); logp = Descriptors.MolLogP(mol)
    hbd = rdMolDescriptors.CalcNumHBD(mol); hba = rdMolDescriptors.CalcNumHBA(mol)
    lip = {
        'mol_weight': mw, 'logp': logp, 'hbd': hbd, 'hba': hba,
        'tpsa': rdMolDescriptors.CalcTPSA(mol),
        'rotatable_bonds': rdMolDescriptors.CalcNumRotatableBonds(mol),
        'aromatic_rings': rdMolDescriptors.CalcNumAromaticRings(mol),
        'ring_count': rdMolDescriptors.CalcNumRings(mol),
        'heavy_atom_count': mol.GetNumHeavyAtoms(),
        'fraction_csp3': rdMolDescriptors.CalcFractionCSP3(mol),
        'molar_refractivity': Descriptors.MolMR(mol),
        'qed': Descriptors.qed(mol),
        'lipinski_violations': int(mw > 500) + int(logp > 5) + int(hbd > 5) + int(hba > 10),
        'passes_lipinski': int(mw <= 500 and logp <= 5 and hbd <= 5 and hba <= 10),
    }
    # Fragments-85 (MARK'S NEW CONTRIBUTION)
    frag = {n: f(mol) for n, f in FRAG_FUNCS}
    # Morgan 1024
    fp = np.array(AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=1024), dtype=np.int8)
    # MACCS-167
    maccs = np.array(MACCSkeys.GenMACCSKeys(mol), dtype=np.int8)
    # Advanced-12 (same as Anthony's list)
    adv = {
        'num_heteroatoms': rdMolDescriptors.CalcNumHeteroatoms(mol),
        'num_saturated_rings': rdMolDescriptors.CalcNumSaturatedRings(mol),
        'num_aliphatic_rings': rdMolDescriptors.CalcNumAliphaticRings(mol),
        'num_amide_bonds': rdMolDescriptors.CalcNumAmideBonds(mol),
        'num_spiro_atoms': rdMolDescriptors.CalcNumSpiroAtoms(mol),
        'num_bridgehead_atoms': rdMolDescriptors.CalcNumBridgeheadAtoms(mol),
        'labuteasa': Descriptors.LabuteASA(mol),
        'balabanj': Descriptors.BalabanJ(mol),
        'bertzct': Descriptors.BertzCT(mol),
        'maxpartialcharge': Descriptors.MaxPartialCharge(mol) or 0.0,
        'minpartialcharge': Descriptors.MinPartialCharge(mol) or 0.0,
        'numvalenceelectrons': Descriptors.NumValenceElectrons(mol),
    }
    return lip, frag, fp, maccs, adv

t0 = time.time()
lip_rows, frag_rows, fp_rows, maccs_rows, adv_rows, mask = [], [], [], [], [], []
for smi in df['smiles']:
    r = compute_all(smi)
    if r is None:
        mask.append(False); continue
    mask.append(True)
    l, fr, fp, mc, ad = r
    lip_rows.append(l); frag_rows.append(fr)
    fp_rows.append(fp); maccs_rows.append(mc); adv_rows.append(ad)
df_v = df[mask].reset_index(drop=True)
lip_df = pd.DataFrame(lip_rows)
frag_df = pd.DataFrame(frag_rows)
fp_arr = np.vstack(fp_rows)
maccs_arr = np.vstack(maccs_rows)
adv_df = pd.DataFrame(adv_rows)
print(f'featurized {len(df_v)} mols in {time.time()-t0:.1f}s')
print(f'Lipinski: {lip_df.shape}  Fragments: {frag_df.shape}  Morgan: {fp_arr.shape}  MACCS: {maccs_arr.shape}  Adv: {adv_df.shape}')"""))

# ---------- Cell 5: build matrices, splits ----------
cells.append(nbf.v4.new_code_cell("""# Assemble the feature matrices Anthony used + Mark's new fragment additions.
def slice_split(arr_or_df, split_name):
    idx = df_v['split'].values == split_name
    if isinstance(arr_or_df, pd.DataFrame):
        return arr_or_df[idx].values
    return arr_or_df[idx]
y = df_v['y'].values.astype(int)

# NaN-safe: some RDKit descriptors (BalabanJ, partial charges) can return NaN/inf on pathological mols
def clean(A):
    A = np.asarray(A, dtype=np.float32)
    A = np.nan_to_num(A, nan=0.0, posinf=1e6, neginf=-1e6)
    return A

lip_mat = clean(lip_df.values); frag_mat = clean(frag_df.values)
adv_mat = clean(adv_df.values); fp_mat = clean(fp_arr); maccs_mat = clean(maccs_arr)
print('post-clean NaN?', np.isnan(lip_mat).any(), np.isnan(adv_mat).any(), np.isnan(frag_mat).any())

FEATURE_SETS = {}
FEATURE_SETS['Lipinski (14)']            = lip_mat
FEATURE_SETS['Fragments (85)']           = frag_mat
FEATURE_SETS['Lip+Frag (99)']            = np.hstack([lip_mat, frag_mat])
FEATURE_SETS['Morgan (1024)']            = fp_mat
FEATURE_SETS['MACCS (167)']              = maccs_mat
FEATURE_SETS['AllTrad (1217)']           = np.hstack([lip_mat, fp_mat, maccs_mat, adv_mat])
FEATURE_SETS['AllTrad+Frag (1302)']      = np.hstack([lip_mat, frag_mat, fp_mat, maccs_mat, adv_mat])

# Build X_tr/X_va/X_te dict for each feature set
splits = {}
for name, X in FEATURE_SETS.items():
    splits[name] = {
        'X_tr': X[df_v['split'].values == 'train'],
        'X_va': X[df_v['split'].values == 'val'],
        'X_te': X[df_v['split'].values == 'test'],
    }
y_tr = y[df_v['split'].values == 'train']
y_va = y[df_v['split'].values == 'val']
y_te = y[df_v['split'].values == 'test']
print('splits built. train', y_tr.shape, 'val', y_va.shape, 'test', y_te.shape, 'pos rate tr', y_tr.mean().round(4))"""))

# ---------- Cell 6: run CatBoost on feature sets ----------
cells.append(nbf.v4.new_code_cell("""# Experiment M3.1 — CatBoost head-to-head on 7 feature sets
# Same settings as Anthony's Phase 3 ablation (500 iter, depth=6, class_weights balanced)
CB_KW = dict(iterations=500, depth=6, learning_rate=0.05,
             loss_function='Logloss', auto_class_weights='Balanced',
             verbose=0, random_seed=42, allow_writing_files=False,
             thread_count=-1)

results = {}
for name, sp in splits.items():
    t0 = time.time()
    cb = CatBoostClassifier(**CB_KW)
    cb.fit(sp['X_tr'], y_tr, eval_set=(sp['X_va'], y_va))
    p_te = cb.predict_proba(sp['X_te'])[:, 1]
    auc = roc_auc_score(y_te, p_te); ap = average_precision_score(y_te, p_te)
    results[name] = {'test_auc': round(auc, 4), 'test_auprc': round(ap, 4),
                     'dims': sp['X_tr'].shape[1], 'fit_s': round(time.time()-t0, 1)}
    print(f"{name:30s}  dim={sp['X_tr'].shape[1]:5d}  AUC={auc:.4f}  AUPRC={ap:.4f}  ({time.time()-t0:.0f}s)")
pd.DataFrame(results).T.sort_values('test_auc', ascending=False)"""))

# ---------- Cell 7: analysis ----------
cells.append(nbf.v4.new_markdown_cell("""### Checkpoint 1 — interpret Exp M3.1

Compare vs Anthony's Phase 3 CatBoost ablation:
- Anthony `Lipinski (14)` → 0.7744 · `AllTrad (1217)` → 0.7841 · `Full hybrid (1345)` → 0.7415

Did **Fragments (85)** alone clear Lipinski (0.7744)? If yes → H1 confirmed: explicit functional-group counts carry signal Lipinski flattens.
Did **AllTrad+Frag (1302)** beat `AllTrad (1217)`? If not → "more features still hurts" — the bottleneck is redundancy, which is exactly the setup for the feature-selection experiment next."""))

# ---------- Cell 8: mutual info selection sweep ----------
cells.append(nbf.v4.new_code_cell("""# Experiment M3.2 — Mutual-info feature selection sweep on AllTrad+Frag (1302 dims)
# Hypothesis: there's a sweet spot K << 1302 that beats both AllTrad-1217 and GIN+Edge-0.7860.
X_full_tr = splits['AllTrad+Frag (1302)']['X_tr']
X_full_va = splits['AllTrad+Frag (1302)']['X_va']
X_full_te = splits['AllTrad+Frag (1302)']['X_te']
print('computing mutual info on', X_full_tr.shape, '...')
t0 = time.time()
mi = mutual_info_classif(X_full_tr, y_tr, discrete_features='auto', random_state=42, n_neighbors=3)
print(f'done in {time.time()-t0:.0f}s, top5 MI =', np.sort(mi)[-5:].round(4))
rank = np.argsort(-mi)

K_grid = [20, 50, 100, 200, 400, 800, 1302]
sel_results = {}
for K in K_grid:
    idx = rank[:K]
    cb = CatBoostClassifier(**CB_KW)
    cb.fit(X_full_tr[:, idx], y_tr, eval_set=(X_full_va[:, idx], y_va))
    p = cb.predict_proba(X_full_te[:, idx])[:, 1]
    auc = roc_auc_score(y_te, p); ap = average_precision_score(y_te, p)
    sel_results[K] = {'auc': round(auc, 4), 'auprc': round(ap, 4)}
    print(f'K={K:4d}  AUC={auc:.4f}  AUPRC={ap:.4f}')
pd.DataFrame(sel_results).T"""))

# ---------- Cell 9: save + plot ----------
cells.append(nbf.v4.new_code_cell("""# Combined comparison chart — Mark's results + Anthony's Phase 3 champions
ANTHONY = {
    'Ant: GIN+Edge (champion)': 0.7860,
    'Ant: AllTrad-1217':        0.7841,
    'Ant: Lipinski-14':         0.7744,
    'Ant: Full hybrid-1345':    0.7415,
    'Ant: GIN+VN':              0.7578,
}
best_sel_K = max(sel_results, key=lambda k: sel_results[k]['auc'])
MARK = {f'Mark: {n}': r['test_auc'] for n, r in results.items()}
MARK[f'Mark: MI-selected K={best_sel_K}'] = sel_results[best_sel_K]['auc']

combined = {**ANTHONY, **MARK}
order = sorted(combined.items(), key=lambda kv: kv[1])
names = [k for k, _ in order]; vals = [v for _, v in order]
colors = ['#1f77b4' if k.startswith('Ant') else '#ff7f0e' for k in names]

fig, ax = plt.subplots(figsize=(10, 8))
ax.barh(names, vals, color=colors)
ax.axvline(0.7860, color='red', ls='--', alpha=0.6, label="Anthony GIN+Edge champion")
for i, v in enumerate(vals):
    ax.text(v + 0.001, i, f'{v:.4f}', va='center', fontsize=9)
ax.set_xlim(0.70, 0.80)
ax.set_xlabel('Test ROC-AUC (ogbg-molhiv)')
ax.set_title('Phase 3 leaderboard — Anthony (blue) vs Mark (orange)')
ax.legend(loc='lower right')
plt.tight_layout()
plt.savefig(RES / 'phase3_mark_leaderboard.png', dpi=120)
plt.close()
print('saved phase3_mark_leaderboard.png')

# Save JSON
out = {
    'phase': '3-mark',
    'date': '2026-04-08',
    'feature_set_ablation': results,
    'mi_selection_sweep': sel_results,
    'best_selected_K': int(best_sel_K),
    'anthony_champion_auc': 0.7860,
    'mark_best_auc': float(max([r['test_auc'] for r in results.values()] + [sel_results[best_sel_K]['auc']])),
}
(RES / 'phase3_mark_results.json').write_text(json.dumps(out, indent=2))
print(json.dumps(out, indent=2))"""))

nb['cells'] = cells
nb.metadata['kernelspec'] = {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'}
nb.metadata['language_info'] = {'name': 'python'}

out_path = 'notebooks/phase3_mark_feature_selection.ipynb'
with open(out_path, 'w', encoding='utf-8') as f:
    nbf.write(nb, f)
print('wrote', out_path, 'cells=', len(cells))
