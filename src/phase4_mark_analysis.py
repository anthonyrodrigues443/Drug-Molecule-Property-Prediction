"""Phase 4 Step 2: Stability + Error Analysis + Feature Importance (loads from cache)."""
import os, json, time, warnings
warnings.filterwarnings('ignore')
os.environ['MPLBACKEND'] = 'Agg'
import numpy as np
import pandas as pd
import matplotlib; matplotlib.use('Agg')
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

from sklearn.metrics import roc_auc_score, confusion_matrix, roc_curve
from catboost import CatBoostClassifier

BASE = Path(__file__).parent.parent
RES = BASE / 'results'; RES.mkdir(exist_ok=True)
CACHE = BASE / 'data' / 'processed'

# ─── Load cached data ────────────────────────────────────────────────────────
print('[1] Loading cached data...')
d = np.load(CACHE / 'phase4_mark_data.npz', allow_pickle=True)
X_tr=d['X_tr']; y_tr=d['y_tr']; X_va=d['X_va']; y_va=d['y_va']
X_te=d['X_te']; y_te=d['y_te']
smiles_te=d['smiles_te'].tolist()
mi_scores=d['mi_scores']; top_400_idx=d['top_400_idx']

with open(CACHE/'p4_optuna.json') as f: od=json.load(f)
preds = np.load(CACHE/'p4_preds.npy')
p_te_tun=preds[0]; p_va_tun=preds[1]

auc_va_def=od['auc_va_def']; auc_te_def=od['auc_te_def']; auprc_te_def=od['auprc_te_def']
auc_va_tun=od['auc_va_tun']; auc_te_tun=od['auc_te_tun']; auprc_te_tun=od['auprc_te_tun']

FRAG_FUNCS = [(n, getattr(Fragments, n)) for n in sorted(dir(Fragments)) if n.startswith('fr_')]
feat_names = ([f'lip_{i}' for i in range(14)] + [f'maccs_{i}' for i in range(167)] +
              [f'morgan_{i}' for i in range(1024)] + [name for name,_ in FRAG_FUNCS] +
              [f'adv_{i}' for i in range(12)])

X_tr_400=X_tr[:,top_400_idx]; X_va_400=X_va[:,top_400_idx]; X_te_400=X_te[:,top_400_idx]
print(f'  Loaded | train={X_tr.shape} val={X_va.shape} test={X_te.shape}')
print(f'  Default: Val={auc_va_def:.4f} Test={auc_te_def:.4f}')
print(f'  Tuned:   Val={auc_va_tun:.4f} Test={auc_te_tun:.4f}')

# ─── K-stability ─────────────────────────────────────────────────────────────
print('[2] K-stability (5 bootstrap × K-sweep, global MI ranking)...')
K_values = [100, 200, 400, 600]  # reduced for speed
rng = np.random.RandomState(42)
n_tr = len(X_tr)
all_K_idx = {k: np.argsort(mi_scores)[::-1][:k] for k in K_values}
boot_results = []

for b in range(3):  # 3 bootstraps
    idx_b = rng.choice(n_tr, size=int(0.8*n_tr), replace=False)
    boot_row = {'boot': b}
    for k in K_values:
        idx_k = all_K_idx[k]
        cb_b = CatBoostClassifier(iterations=80, learning_rate=0.10, depth=5,
                                  eval_metric='AUC', random_seed=42, verbose=0,
                                  auto_class_weights='Balanced', task_type='CPU')
        cb_b.fit(X_tr[idx_b][:, idx_k], y_tr[idx_b],
                 eval_set=(X_va[:, idx_k], y_va), early_stopping_rounds=10)
        boot_row[k] = roc_auc_score(y_va, cb_b.predict_proba(X_va[:, idx_k])[:,1])
    boot_row['best_K'] = max(K_values, key=lambda k: boot_row[k])
    boot_results.append(boot_row)
    print(f'  Boot {b}: ' + ' | '.join(f'K={k}:{boot_row[k]:.4f}' for k in K_values) +
          f' -> best={boot_row["best_K"]}')

stab_df = pd.DataFrame(boot_results)
best_k_freq = stab_df['best_K'].mode()[0]
print(f'  Most frequent best K: {best_k_freq} | per-boot: {stab_df["best_K"].tolist()}')
if 400 in K_values:
    print(f'  K=400 mean={stab_df[400].mean():.4f}+/-{stab_df[400].std():.4f}')

# Stability plot
mean_a=stab_df[K_values].mean(); std_a=stab_df[K_values].std()
fig,(ax1,ax2)=plt.subplots(1,2,figsize=(12,5))
for _,row in stab_df.iterrows():
    ax1.plot(K_values,[row[k] for k in K_values],'o-',alpha=0.5,lw=1.5)
ax1.plot(K_values,mean_a.values,'k-',lw=2.5,label='Mean')
ax1.fill_between(K_values,(mean_a-std_a).values,(mean_a+std_a).values,alpha=0.15,color='black',label='+/- 1 std')
ax1.axvline(400,color='red',ls='--',alpha=0.7,label='K=400 (Phase 3)')
ax1.set_xlabel('K'); ax1.set_ylabel('Val ROC-AUC'); ax1.set_title('K-stability (5×80% bootstrap)')
ax1.legend(fontsize=8)
hm=stab_df[K_values].values
im=ax2.imshow(hm,aspect='auto',cmap='YlOrRd')
ax2.set_xticks(range(len(K_values))); ax2.set_xticklabels(K_values)
n_boots = len(stab_df); ax2.set_yticks(range(n_boots)); ax2.set_yticklabels([f'Boot {i}' for i in range(n_boots)])
ax2.set_title('Val AUC heatmap')
plt.colorbar(im,ax=ax2)
n_boots = len(stab_df)
for r in range(n_boots):
    for c in range(len(K_values)):
        ax2.text(c,r,f'{hm[r,c]:.3f}',ha='center',va='center',fontsize=8,
                 color='white' if hm[r,c]>hm.mean() else 'black')
plt.tight_layout()
plt.savefig(RES/'phase4_mark_stability.png',dpi=150,bbox_inches='tight'); plt.close()
print('  Saved: phase4_mark_stability.png')

# ─── Error analysis ──────────────────────────────────────────────────────────
print('[3] Error analysis...')
fpr_va,tpr_va,thresh_va=roc_curve(y_va,p_va_tun)
youden_thresh=thresh_va[np.argmax(tpr_va-fpr_va)]
y_pred=(p_te_tun>=youden_thresh).astype(int)
tn,fp,fn,tp=confusion_matrix(y_te,y_pred).ravel()
prec=tp/max(tp+fp,1); rec=tp/max(tp+fn,1); f1=2*prec*rec/max(prec+rec,1e-9)
print(f'  Youden={youden_thresh:.4f} | TP={tp} TN={tn} FP={fp} FN={fn}')
print(f'  Prec={prec:.4f} Rec={rec:.4f} F1={f1:.4f}')

# Build test df with molecular properties
te_df = pd.DataFrame({'smiles': smiles_te, 'y': y_te, 'pred_prob': p_te_tun, 'pred_label': y_pred})
def get_props(smi):
    mol=Chem.MolFromSmiles(smi)
    if mol is None: return {}
    return {'mw':Descriptors.MolWt(mol),'logp':Descriptors.MolLogP(mol),
            'hbd':rdMolDescriptors.CalcNumHBD(mol),'hba':rdMolDescriptors.CalcNumHBA(mol),
            'tpsa':Descriptors.TPSA(mol),'rb':rdMolDescriptors.CalcNumRotatableBonds(mol),
            'rings':rdMolDescriptors.CalcNumRings(mol),'arom':rdMolDescriptors.CalcNumAromaticRings(mol),
            'heavyatoms':mol.GetNumHeavyAtoms(),'frac_csp3':rdMolDescriptors.CalcFractionCSP3(mol),
            'halogens':sum(1 for a in mol.GetAtoms() if a.GetAtomicNum() in (9,17,35,53))}

props_df=pd.DataFrame([get_props(s) for s in te_df['smiles']])
te_df=pd.concat([te_df,props_df],axis=1)
te_df['error_class']='TN'
te_df.loc[(te_df['y']==1)&(te_df['pred_label']==1),'error_class']='TP'
te_df.loc[(te_df['y']==0)&(te_df['pred_label']==1),'error_class']='FP'
te_df.loc[(te_df['y']==1)&(te_df['pred_label']==0),'error_class']='FN'
print(f'  Classes: {te_df["error_class"].value_counts().to_dict()}')

prop_cols=['mw','logp','hbd','hba','tpsa','rb','rings','arom','heavyatoms','frac_csp3','halogens']
print('  Mean properties by class:')
print(te_df.groupby('error_class')[prop_cols].mean().round(2).to_string())

fn_tp_diffs={}
fn_m=te_df['error_class']=='FN'; tp_m=te_df['error_class']=='TP'
print('\n  FN vs TP significant differences:')
for col in prop_cols:
    fn_v=te_df.loc[fn_m,col].dropna(); tp_v=te_df.loc[tp_m,col].dropna()
    if len(fn_v)>3 and len(tp_v)>3:
        _,p=scipy_stats.mannwhitneyu(fn_v,tp_v,alternative='two-sided')
        fn_tp_diffs[col]={'fn_mean':float(fn_v.mean()),'tp_mean':float(tp_v.mean()),
                          'p_value':float(p),'significant':p<0.05}
        if p<0.05:
            print(f'    {col:12s}: FN={fn_v.mean():.2f} vs TP={tp_v.mean():.2f}  p={p:.3f}{"**" if p<0.01 else "*"}')

# Lipinski violations
te_df['lip_viol']=((te_df['mw']>500).astype(int)+(te_df['logp']>5).astype(int)+
                   (te_df['hbd']>5).astype(int)+(te_df['hba']>10).astype(int))
act=te_df[te_df['y']==1]
easy=act[act['lip_viol']==0]; hard=act[act['lip_viol']>=2]
easy_rec=(easy['pred_label']==1).mean() if len(easy) else 0
hard_rec=(hard['pred_label']==1).mean() if len(hard) else 0
print(f'  Lipinski 0-viol: recall={easy_rec:.3f} (n={len(easy)}) | >=2-viol: recall={hard_rec:.3f} (n={len(hard)})')

uncertain=te_df[(te_df['pred_prob']>0.35)&(te_df['pred_prob']<0.65)]
near_err=(uncertain['y']!=uncertain['pred_label']).mean() if len(uncertain) else 0
print(f'  Near-threshold zone: n={len(uncertain)}, error={near_err:.3f}')

# Scaffold analysis
def get_scaffold(smi):
    try:
        mol=Chem.MolFromSmiles(smi)
        if mol: return MurckoScaffold.MurckoScaffoldSmiles(mol=mol)
    except: pass
    return 'no_scaffold'
te_df['scaffold']=[get_scaffold(s) for s in te_df['smiles']]
sc=te_df.groupby('scaffold').agg(n=('y','count'),actives=('y','sum'),
    FN=('error_class',lambda x:(x=='FN').sum()),TP=('error_class',lambda x:(x=='TP').sum())).reset_index()
sc['fn_rate']=sc['FN']/(sc['FN']+sc['TP']).clip(lower=1)
high_fn=sc[sc['actives']>=3].nlargest(5,'fn_rate')
print('  Top 5 high-FN scaffolds:')
print(high_fn[['scaffold','n','actives','TP','FN','fn_rate']].to_string())

# Error plots
colors={'TP':'#2196F3','TN':'#4CAF50','FP':'#FF9800','FN':'#F44336'}
fig,axes=plt.subplots(2,4,figsize=(16,8))
for ax,prop in zip(axes.flat,['mw','logp','tpsa','rings','hba','hbd','frac_csp3','halogens']):
    for cls in ['TP','TN','FP','FN']:
        v=te_df.loc[te_df['error_class']==cls,prop].dropna()
        if len(v): ax.hist(v,bins=25,alpha=0.5,label=cls,color=colors[cls],density=True)
    ax.set_xlabel(prop); ax.set_title(prop); ax.legend(fontsize=7)
plt.suptitle('Molecular Properties by Error Class',fontsize=12,fontweight='bold')
plt.tight_layout()
plt.savefig(RES/'phase4_mark_error_properties.png',dpi=150,bbox_inches='tight'); plt.close()

fig,axes=plt.subplots(1,3,figsize=(15,5))
for cls in ['TP','FP']:
    v=te_df.loc[te_df['error_class']==cls,'pred_prob'].values
    axes[0].hist(v,bins=30,alpha=0.6,label=f'{cls}(n={len(v)})',color=colors[cls])
for cls in ['TN','FN']:
    v=te_df.loc[te_df['error_class']==cls,'pred_prob'].values
    axes[1].hist(v,bins=30,alpha=0.6,label=f'{cls}(n={len(v)})',color=colors[cls])
axes[0].set_title('TP vs FP confidence'); axes[0].legend(); axes[0].set_xlabel('P(active)')
axes[1].set_title('TN vs FN confidence'); axes[1].legend(); axes[1].set_xlabel('P(active)')
spc=te_df.groupby('error_class')['scaffold'].nunique()
axes[2].bar(spc.index,spc.values,color=[colors.get(c,'gray') for c in spc.index])
axes[2].set_title('Scaffold diversity per class')
for i,(c,v) in enumerate(spc.items()): axes[2].text(i,v+1,str(v),ha='center')
plt.tight_layout()
plt.savefig(RES/'phase4_mark_error_confidence.png',dpi=150,bbox_inches='tight'); plt.close()
print('  Saved error analysis plots')

# ─── Feature importance ───────────────────────────────────────────────────────
print('[4] Feature importance (refit tuned model)...')
cb_tun=CatBoostClassifier(**od['best_params'],eval_metric='AUC',random_seed=42,verbose=0,
                           auto_class_weights='Balanced',task_type='CPU')
cb_tun.fit(X_tr_400,y_tr,eval_set=(X_va_400,y_va),early_stopping_rounds=30)
importances=cb_tun.get_feature_importance()
feat_400_names=[feat_names[i] for i in top_400_idx]
imp_df=pd.DataFrame({'feature':feat_400_names,'importance':importances})
imp_df=imp_df.sort_values('importance',ascending=False).reset_index(drop=True)
imp_df['cumulative']=imp_df['importance'].cumsum()/imp_df['importance'].sum()

def feat_cat(n):
    if n.startswith('lip_'): return 'Lipinski'
    if n.startswith('maccs_'): return 'MACCS'
    if n.startswith('morgan_'): return 'Morgan FP'
    if n.startswith('fr_'): return 'Fragment'
    if n.startswith('adv_'): return 'Advanced'
    return 'Other'

imp_df['category']=imp_df['feature'].apply(feat_cat)
cat_imp=imp_df.groupby('category')['importance'].sum().sort_values(ascending=False)
total=cat_imp.sum()
n50=(imp_df['cumulative']<=0.50).sum()+1
n80=(imp_df['cumulative']<=0.80).sum()+1
n95=(imp_df['cumulative']<=0.95).sum()+1
print('  Category importance:')
for c,v in cat_imp.items(): print(f'    {c:12s}: {v/total*100:.1f}%')
print(f'  50%: top {n50} | 80%: top {n80} | 95%: top {n95}')
print(f'  Top 10: {imp_df["feature"].head(10).tolist()}')

bar_colors={'Lipinski':'#2196F3','MACCS':'#4CAF50','Morgan FP':'#FF9800','Fragment':'#9C27B0','Advanced':'#F44336'}
fig,axes=plt.subplots(1,3,figsize=(17,5))
top25=imp_df.head(25)
axes[0].barh(top25['feature'][::-1],top25['importance'][::-1],
             color=[bar_colors.get(feat_cat(f),'gray') for f in top25['feature'][::-1]])
axes[0].set_xlabel('Importance'); axes[0].set_title('Top 25 Features (Tuned CatBoost + MI-400)')
from matplotlib.patches import Patch
axes[0].legend(handles=[Patch(facecolor=v,label=k) for k,v in bar_colors.items()],fontsize=7)
axes[1].plot(range(1,len(imp_df)+1),imp_df['cumulative'],color='purple')
for pct,col in [(0.5,'blue'),(0.8,'orange'),(0.95,'red')]:
    n=(imp_df['cumulative']<=pct).sum()+1
    axes[1].axhline(pct,color=col,ls='--',alpha=0.7,label=f'{int(pct*100)}%: top {n}')
axes[1].set_xlabel('# features'); axes[1].set_title('How many features matter?'); axes[1].legend(fontsize=8)
axes[2].pie(cat_imp.values,labels=cat_imp.index,autopct='%1.1f%%',
            colors=[bar_colors.get(c,'gray') for c in cat_imp.index])
axes[2].set_title('Importance by Category')
plt.tight_layout()
plt.savefig(RES/'phase4_mark_feature_importance.png',dpi=150,bbox_inches='tight'); plt.close()
print('  Saved: phase4_mark_feature_importance.png')

# ─── Save results ─────────────────────────────────────────────────────────────
phase4={
    'phase':'4-mark','date':'2026-04-09',
    'baseline':{'val_auc':auc_va_def,'test_auc':auc_te_def,'test_auprc':auprc_te_def},
    'optuna':{'n_trials':40,'best_val_auc':od['best_val'],'best_params':od['best_params'],
              'tuned_val_auc':auc_va_tun,'tuned_test_auc':auc_te_tun,'tuned_test_auprc':auprc_te_tun,
              'delta_val':float(od['best_val']-auc_va_def),'delta_test':float(auc_te_tun-auc_te_def),
              'val_test_gap_default':float(auc_va_def-auc_te_def),
              'val_test_gap_tuned':float(auc_va_tun-auc_te_tun)},
    'stability':{'method':'bootstrap_global_mi','k_values':K_values,
                 'mean_per_k':{str(k):float(stab_df[k].mean()) for k in K_values},
                 'std_per_k':{str(k):float(stab_df[k].std()) for k in K_values},
                 'best_k_per_boot':stab_df['best_K'].tolist(),'most_frequent_k':int(best_k_freq)},
    'error_analysis':{'youden_threshold':float(youden_thresh),'TP':int(tp),'TN':int(tn),'FP':int(fp),'FN':int(fn),
                      'precision':float(prec),'recall':float(rec),'f1':float(f1),
                      'near_threshold_error_rate':float(near_err),
                      'lipinski_0viol_recall':float(easy_rec),'lipinski_2plus_recall':float(hard_rec),
                      'fn_tp_significant_diffs':{k:v for k,v in fn_tp_diffs.items() if v['significant']}},
    'feature_importance':{'top10':imp_df['feature'].head(10).tolist(),
                          'n_for_50pct':n50,'n_for_80pct':n80,'n_for_95pct':n95,
                          'category_pct':{k:float(v/total) for k,v in cat_imp.items()}}
}
with open(RES/'phase4_mark_results.json','w') as f: json.dump(phase4,f,indent=2)
print('Saved: phase4_mark_results.json')
metrics_path=RES/'metrics.json'
try:
    with open(metrics_path) as f: all_m=json.load(f)
    if not isinstance(all_m,list): all_m=[all_m]
except: all_m=[]
all_m.append(phase4)
with open(metrics_path,'w') as f: json.dump(all_m,f,indent=2)

# ─── Final summary ────────────────────────────────────────────────────────────
print('\n'+'='*62)
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
print(f'  Tuning val lift:    {od["best_val"]-auc_va_def:+.4f}')
print(f'  Tuning test lift:   {auc_te_tun-auc_te_def:+.4f}')
print(f'  Val-Test gap grew:  {auc_va_def-auc_te_def:+.4f} -> {auc_va_tun-auc_te_tun:+.4f}')
print(f'  K stability: most frequent best K = {best_k_freq}')
print(f'  50% importance in {n50} features (of 400)')
print(f'  MACCS: {cat_imp.get("MACCS",0)/total*100:.1f}% importance (12.8% of pool)')
print(f'  Lipinski-compliant recall: {easy_rec:.3f} | violators: {hard_rec:.3f}')
print(f'\nDone!')
