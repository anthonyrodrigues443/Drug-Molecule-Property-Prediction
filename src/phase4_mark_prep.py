"""Phase 4 Step 1: Precompute features + MI + run Optuna. Save everything to disk."""
import os, json, time, warnings, random, pickle
warnings.filterwarnings('ignore')
os.environ['MPLBACKEND'] = 'Agg'
import numpy as np
import pandas as pd
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

import torch
_orig_load = torch.load
torch.load = lambda *a, **k: _orig_load(*a, **{**k, 'weights_only': False})

from ogb.graphproppred import GraphPropPredDataset
from rdkit import Chem
from rdkit.Chem import Descriptors, Fragments, rdMolDescriptors, AllChem, MACCSkeys

from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.feature_selection import mutual_info_classif
from catboost import CatBoostClassifier
import optuna; optuna.logging.set_verbosity(optuna.logging.WARNING)

BASE = Path(__file__).parent.parent
RES = BASE / 'results'; RES.mkdir(exist_ok=True)
CACHE = BASE / 'data' / 'processed'; CACHE.mkdir(exist_ok=True)
np.random.seed(42); random.seed(42)

FRAG_FUNCS = [(n, getattr(Fragments, n)) for n in sorted(dir(Fragments)) if n.startswith('fr_')]
feat_names = ([f'lip_{i}' for i in range(14)] + [f'maccs_{i}' for i in range(167)] +
              [f'morgan_{i}' for i in range(1024)] + [name for name, _ in FRAG_FUNCS] +
              [f'adv_{i}' for i in range(12)])
feat_dim = 1302

def compute_all(smi):
    mol = Chem.MolFromSmiles(smi)
    if mol is None: return None
    mw=Descriptors.MolWt(mol); logp=Descriptors.MolLogP(mol)
    hbd=rdMolDescriptors.CalcNumHBD(mol); hba=rdMolDescriptors.CalcNumHBA(mol)
    tpsa=Descriptors.TPSA(mol); rb=rdMolDescriptors.CalcNumRotatableBonds(mol)
    rings=rdMolDescriptors.CalcNumRings(mol); arom=rdMolDescriptors.CalcNumAromaticRings(mol)
    hal=sum(1 for a in mol.GetAtoms() if a.GetAtomicNum() in (9,17,35,53))
    sp3=rdMolDescriptors.CalcFractionCSP3(mol)
    nhet=sum(1 for a in mol.GetAtoms() if a.GetAtomicNum() not in (1,6))
    try:
        from rdkit.Chem import QED; qed_val=QED.qed(mol)
    except: qed_val=0.0
    lip14=[mw,logp,hbd,hba,tpsa,rb,rings,arom,hal,sp3,nhet,qed_val,
           Descriptors.NumRadicalElectrons(mol),Descriptors.NumValenceElectrons(mol)]
    maccs=list(MACCSkeys.GenMACCSKeys(mol).ToList())
    morgan=list(AllChem.GetMorganFingerprintAsBitVect(mol,radius=2,nBits=1024).ToList())
    frags=[int(fn(mol)) for _,fn in FRAG_FUNCS]
    adv=[logp/max(mw,1),hbd+hba,hbd*hba,rb/max(rings,1),arom/max(rings,1),
         sp3*mw,nhet/max(mol.GetNumHeavyAtoms(),1),(hbd+hba)/max(tpsa,1),
         rb*logp,hal/max(mol.GetNumHeavyAtoms(),1),sp3*logp,mw/max(mol.GetNumHeavyAtoms(),1)]
    return lip14+maccs+morgan+frags+adv

cache_file = CACHE / 'phase4_mark_data.npz'
feat_names_file = CACHE / 'feat_names.json'
if cache_file.exists():
    print('Loading cached features...')
    d = np.load(cache_file, allow_pickle=True)
    X_tr=d['X_tr']; y_tr=d['y_tr']; X_va=d['X_va']; y_va=d['y_va']
    X_te=d['X_te']; y_te=d['y_te']
    smiles_te=d['smiles_te'].tolist()
    mi_scores=d['mi_scores']; top_400_idx=d['top_400_idx']
    print(f'  Loaded: train={X_tr.shape} val={X_va.shape} test={X_te.shape}')
else:
    print('Computing features (first time ~5 min)...')
    t0=time.time()
    dataset=GraphPropPredDataset(name='ogbg-molhiv',root=str(BASE/'data'/'raw'))
    split_idx=dataset.get_idx_split()
    smiles_df=pd.read_csv(BASE/'data'/'raw'/'ogbg_molhiv'/'mapping'/'mol.csv.gz')
    labels=dataset.labels.flatten()
    train_idx=set(split_idx['train'].tolist()); val_idx=set(split_idx['valid'].tolist())
    test_idx=set(split_idx['test'].tolist())
    df=smiles_df.copy(); df['y']=labels
    df['split']=['train' if i in train_idx else 'val' if i in val_idx else 'test' for i in range(len(df))]
    all_feats=[compute_all(s) for s in df['smiles']]
    X_all=np.array([f if f is not None else [0]*feat_dim for f in all_feats], dtype=np.float32)
    y_all=df['y'].values.astype(int)
    tr_m=df['split']=='train'; va_m=df['split']=='val'; te_m=df['split']=='test'
    X_tr,y_tr=X_all[tr_m],y_all[tr_m]
    X_va,y_va=X_all[va_m],y_all[va_m]
    X_te,y_te=X_all[te_m],y_all[te_m]
    smiles_te=df.loc[te_m,'smiles'].tolist()
    print(f'  Features done in {time.time()-t0:.1f}s | Computing MI...')
    t0=time.time()
    mi_scores=mutual_info_classif(X_tr,y_tr,random_state=42)
    top_400_idx=np.argsort(mi_scores)[::-1][:400]
    print(f'  MI done in {time.time()-t0:.1f}s | top feature: {feat_names[top_400_idx[0]]}')
    np.savez_compressed(cache_file, X_tr=X_tr,y_tr=y_tr,X_va=X_va,y_va=y_va,X_te=X_te,y_te=y_te,
                        mi_scores=mi_scores,top_400_idx=top_400_idx,smiles_te=np.array(smiles_te))
    with open(feat_names_file,'w') as f: json.dump(feat_names,f)
    print(f'  Cached to {cache_file}')

X_tr_400=X_tr[:,top_400_idx]; X_va_400=X_va[:,top_400_idx]; X_te_400=X_te[:,top_400_idx]

# ─── Default CatBoost ────────────────────────────────────────────────────────
print('Default CatBoost (MI-400)...')
cb_def=CatBoostClassifier(iterations=500,learning_rate=0.05,depth=6,eval_metric='AUC',
                          random_seed=42,verbose=0,auto_class_weights='Balanced',task_type='CPU')
cb_def.fit(X_tr_400,y_tr,eval_set=(X_va_400,y_va),early_stopping_rounds=50)
p_va_def=cb_def.predict_proba(X_va_400)[:,1]; p_te_def=cb_def.predict_proba(X_te_400)[:,1]
auc_va_def=roc_auc_score(y_va,p_va_def); auc_te_def=roc_auc_score(y_te,p_te_def)
auprc_te_def=average_precision_score(y_te,p_te_def)
print(f'  Default | Val={auc_va_def:.4f} | Test={auc_te_def:.4f} | Phase3=0.8105 delta={auc_te_def-0.8105:+.4f}')

# ─── Optuna ──────────────────────────────────────────────────────────────────
print('Optuna (40 trials)...')
def objective(trial):
    p=dict(iterations=trial.suggest_int('iterations',300,800,step=100),
           learning_rate=trial.suggest_float('learning_rate',0.01,0.12,log=True),
           depth=trial.suggest_int('depth',4,8),
           l2_leaf_reg=trial.suggest_float('l2_leaf_reg',1.0,20.0,log=True),
           min_data_in_leaf=trial.suggest_int('min_data_in_leaf',5,50),
           random_strength=trial.suggest_float('random_strength',0.5,5.0),
           bagging_temperature=trial.suggest_float('bagging_temperature',0.0,2.0),
           border_count=trial.suggest_categorical('border_count',[64,128,254]))
    m=CatBoostClassifier(**p,eval_metric='AUC',random_seed=42,verbose=0,
                         auto_class_weights='Balanced',task_type='CPU')
    m.fit(X_tr_400,y_tr,eval_set=(X_va_400,y_va),early_stopping_rounds=30)
    return roc_auc_score(y_va,m.predict_proba(X_va_400)[:,1])

t0=time.time()
study=optuna.create_study(direction='maximize',sampler=optuna.samplers.TPESampler(seed=42))
study.optimize(objective,n_trials=40)
print(f'  Optuna done in {time.time()-t0:.1f}s | Best val={study.best_value:.4f}')
print(f'  Best params: {study.best_params}')

cb_tun=CatBoostClassifier(**study.best_params,eval_metric='AUC',random_seed=42,verbose=0,
                           auto_class_weights='Balanced',task_type='CPU')
cb_tun.fit(X_tr_400,y_tr,eval_set=(X_va_400,y_va),early_stopping_rounds=30)
p_va_tun=cb_tun.predict_proba(X_va_400)[:,1]; p_te_tun=cb_tun.predict_proba(X_te_400)[:,1]
auc_va_tun=roc_auc_score(y_va,p_va_tun); auc_te_tun=roc_auc_score(y_te,p_te_tun)
auprc_te_tun=average_precision_score(y_te,p_te_tun)
print(f'  Tuned  | Val={auc_va_tun:.4f} | Test={auc_te_tun:.4f} | delta={auc_te_tun-auc_te_def:+.4f}')

# Save Optuna plot
trial_vals=[t.value for t in study.trials if t.value is not None]
fig,(ax1,ax2)=plt.subplots(1,2,figsize=(12,4))
ax1.plot(trial_vals,alpha=0.5,color='steelblue')
ax1.axhline(study.best_value,color='red',ls='--',label=f'Best={study.best_value:.4f}')
ax1.axhline(auc_va_def,color='gray',ls=':',label=f'Default={auc_va_def:.4f}')
ax1.set_xlabel('Trial'); ax1.set_ylabel('Val AUC'); ax1.legend(); ax1.set_title('Optuna Trials (40)')
ax2.plot(np.maximum.accumulate(trial_vals),color='darkorange',lw=2,label='Running best')
ax2.axhline(auc_va_def,color='gray',ls=':',label=f'Default={auc_va_def:.4f}')
ax2.set_xlabel('Trial'); ax2.legend(); ax2.set_title('Search Convergence')
plt.tight_layout()
plt.savefig(RES/'phase4_mark_optuna_history.png',dpi=150,bbox_inches='tight'); plt.close()
print('  Saved: phase4_mark_optuna_history.png')

# Save predictions and model state
optuna_data={'auc_va_def':float(auc_va_def),'auc_te_def':float(auc_te_def),'auprc_te_def':float(auprc_te_def),
             'auc_va_tun':float(auc_va_tun),'auc_te_tun':float(auc_te_tun),'auprc_te_tun':float(auprc_te_tun),
             'best_val':float(study.best_value),'best_params':study.best_params}
np.save(CACHE/'p4_preds.npy',np.stack([p_te_tun,p_va_tun]))
with open(CACHE/'p4_optuna.json','w') as f: json.dump(optuna_data,f)
print('Saved predictions and Optuna results. Run phase4_mark_analysis.py next.')
