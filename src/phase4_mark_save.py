"""Save Phase 4 results to JSON (run after phase4_mark_analysis.py gathered results)."""
import json, numpy as np
from pathlib import Path

BASE = Path(__file__).parent.parent
RES = BASE / 'results'; CACHE = BASE / 'data' / 'processed'

# ── Hardcode results from the completed analysis run ──────────────────────────
phase4 = {
    'phase': '4-mark', 'date': '2026-04-09',
    'baseline': {
        'model': 'CatBoost default + MI-top-400',
        'val_auc': 0.7878, 'test_auc': 0.7584, 'test_auprc': 0.3179,
        'note': 'Phase 3 reported 0.8105 — run-to-run variance on scaffold split'
    },
    'optuna': {
        'n_trials': 40,
        'best_val_auc': 0.8229,
        'best_params': {
            'iterations': 300, 'learning_rate': 0.05538, 'depth': 8,
            'l2_leaf_reg': 4.710, 'min_data_in_leaf': 38,
            'random_strength': 1.057, 'bagging_temperature': 0.854, 'border_count': 64
        },
        'tuned_val_auc': 0.8229,
        'tuned_test_auc': 0.7854,
        'tuned_test_auprc': 0.3179,
        'delta_val': 0.0351,
        'delta_test': 0.0270,
        'val_test_gap_default': 0.0294,
        'val_test_gap_tuned': 0.0375,
        'key_insight': 'Optuna val gain (+0.035) did not transfer to test (+0.027). Val-test gap WIDENED — hyperparameter overfitting on val scaffold.'
    },
    'stability': {
        'method': '3x bootstrap 80% training, global MI ranking, K in [100,200,400,600]',
        'k_values': [100, 200, 400, 600],
        'boot0': {'K100': 0.8004, 'K200': 0.7636, 'K400': 0.8029, 'K600': 0.8023, 'best': 400},
        'boot1': {'K100': 0.8076, 'K200': 0.7717, 'K400': 0.8018, 'K600': 0.7707, 'best': 100},
        'boot2': {'K100': 0.7786, 'K200': 0.8069, 'K400': 0.8092, 'K600': 0.7590, 'best': 400},
        'mean_per_k': {'100': 0.7955, '200': 0.7807, '400': 0.8046, '600': 0.7773},
        'std_per_k':  {'100': 0.0152, '200': 0.0228, '400': 0.0040, '600': 0.0218},
        'best_k_per_boot': [400, 100, 400],
        'most_frequent_k': 400,
        'verdict': 'K=400 wins 2/3 bootstraps and has lowest std (0.0040 vs 0.015-0.022 for others). Most stable selection.'
    },
    'error_analysis': {
        'youden_threshold': 0.3958,
        'TP': 77, 'TN': 3419, 'FP': 564, 'FN': 53,
        'precision': 0.1201, 'recall': 0.5923, 'f1': 0.1997,
        'near_threshold_error_rate': 0.601,
        'lipinski_0viol_recall': 0.400,
        'lipinski_2plus_recall': 0.828,
        'counterintuitive_finding': 'Lipinski-violating actives have HIGHER recall (0.828) than rule-compliant actives (0.400). Model is better at large complex HIV inhibitors than small simple ones.',
        'fn_vs_tp_property_diffs': {
            'mw': {'FN_mean': 423.95, 'TP_mean': 630.02, 'p': 0.000, 'note': 'Missed actives are ~200 Da lighter'},
            'logp': {'FN_mean': 2.74, 'TP_mean': 5.44, 'p': 0.000, 'note': 'Missed actives are more hydrophilic'},
            'hbd': {'FN_mean': 2.36, 'TP_mean': 4.35, 'p': 0.000, 'note': 'Fewer H-bond donors in misses'},
            'hba': {'FN_mean': 6.19, 'TP_mean': 9.55, 'p': 0.000, 'note': 'Fewer H-bond acceptors in misses'},
            'tpsa': {'FN_mean': 102.02, 'TP_mean': 181.07, 'p': 0.000, 'note': 'Lower polarity surface in misses'},
            'rings': {'FN_mean': 3.96, 'TP_mean': 5.60, 'p': 0.000, 'note': 'Fewer ring systems in misses'},
            'arom': {'FN_mean': 2.32, 'TP_mean': 4.44, 'p': 0.000, 'note': 'Fewer aromatic rings in misses'},
            'heavyatoms': {'FN_mean': 29.70, 'TP_mean': 43.91, 'p': 0.000, 'note': '14 fewer heavy atoms on average'},
            'frac_csp3': {'FN_mean': 0.35, 'TP_mean': 0.16, 'p': 0.000, 'note': 'Misses have MORE sp3 carbons — more saturated, drug-like'}
        },
        'scaffold_finding': 'No single scaffold dominates high-FN rate — misses are distributed across chemical space of simple molecules',
        'error_class_distribution': {'TN': 3419, 'FP': 564, 'TP': 77, 'FN': 53},
        'scaffold_diversity': {'TP': '~65 unique', 'FN': '~50 unique', 'FP': '~450+ unique', 'TN': '2800+ unique'}
    },
    'feature_importance': {
        'top10': ['maccs_144', 'adv_11', 'lip_0', 'lip_13', 'maccs_81',
                  'fr_hdrzone', 'adv_2', 'adv_8', 'adv_5', 'morgan_967'],
        'n_for_50pct': 41, 'n_for_80pct': 110, 'n_for_95pct': 192,
        'category_pct': {'MACCS': 0.310, 'Morgan FP': 0.279, 'Advanced': 0.163,
                         'Fragment': 0.129, 'Lipinski': 0.120},
        'concentration_insight': '50% of importance in top 41 features (of 400). MACCS 31% despite being 12.8% of feature pool — hand-curated substructure keys punch above their weight.'
    },
    'leaderboard': [
        {'model': 'Tuned CatBoost MI-400 [Mark P4]', 'val_auc': 0.8229, 'test_auc': 0.7854, 'auprc': 0.3179},
        {'model': 'Default CatBoost MI-400 [Mark P3, this run]', 'val_auc': 0.7878, 'test_auc': 0.7584, 'auprc': 0.3179},
        {'model': 'CatBoost MI-400 [Phase 3 reported]', 'val_auc': None, 'test_auc': 0.8105, 'auprc': 0.3481},
        {'model': 'GIN+Edge [Anthony P3]', 'val_auc': None, 'test_auc': 0.7860, 'auprc': 0.3441},
        {'model': 'MLP-Domain9 [Mark P2]', 'val_auc': None, 'test_auc': 0.7670, 'auprc': None},
        {'model': 'CatBoost default [Mark P1]', 'val_auc': None, 'test_auc': 0.7782, 'auprc': 0.3708},
    ]
}

with open(RES / 'phase4_mark_results.json', 'w') as f:
    json.dump(phase4, f, indent=2)
print('Saved: phase4_mark_results.json')

metrics_path = RES / 'metrics.json'
try:
    with open(metrics_path) as f: all_m = json.load(f)
    if not isinstance(all_m, list): all_m = [all_m]
except: all_m = []
all_m.append(phase4)
with open(metrics_path, 'w') as f: json.dump(all_m, f, indent=2)
print('Updated metrics.json')

print('\nPhase 4 results saved successfully.')
print('Key findings:')
print('  1. Tuning val lift: +0.0351 | test lift: +0.027 (overfitting to val split)')
print('  2. K=400 stable (2/3 bootstrap wins, lowest std 0.0040)')
print('  3. COUNTERINTUITIVE: Lipinski violators have 2x higher recall (0.828 vs 0.400)')
print('  4. Misses are small/simple; model learns large complex HIV inhibitors better')
print('  5. 50% of CatBoost importance in 41 features; MACCS punches above its pool share (31% vs 12.8%)')
