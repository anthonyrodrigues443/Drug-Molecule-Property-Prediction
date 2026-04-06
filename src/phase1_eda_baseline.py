"""
Phase 1: Domain Research + Dataset + EDA + Baseline Models
Drug Molecule Property Prediction — ESOL (aqueous solubility) dataset
Target: log(mol/L) solubility, primary metric: RMSE

Key questions answered today:
1. What does the ESOL molecular space look like? (Lipinski compliance, diversity)
2. Can domain-aware Lipinski features outperform random Morgan fingerprints?
3. What is the baseline RMSE using the simplest model? Target from literature: ~0.58 RMSE.
4. Does Lipinski Rule of 5 hold predictively for solubility? (spoiler: partial)
"""

import os
import sys
import json
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from datetime import datetime

# ML
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score, KFold

import xgboost as xgb

warnings.filterwarnings("ignore")

# Paths
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"
RESULTS_DIR = BASE_DIR / "results"
REPORTS_DIR = BASE_DIR / "reports"

RESULTS_DIR.mkdir(exist_ok=True)
REPORTS_DIR.mkdir(exist_ok=True)
(DATA_DIR / "processed").mkdir(exist_ok=True)

sys.path.insert(0, str(BASE_DIR / "src"))
from data_pipeline import (
    download_esol,
    build_feature_matrix,
    compute_lipinski_features,
    compute_morgan_fingerprints,
)

sns.set_theme(style="whitegrid", palette="husl", font_scale=1.1)

# ─────────────────────────────────────────────────────────────
# 1. DOWNLOAD & DESCRIBE DATASET
# ─────────────────────────────────────────────────────────────
print("=" * 60)
print("PHASE 1: ESOL Dataset — Domain Research + EDA + Baseline")
print("=" * 60)

print("\n[1] Downloading ESOL from MoleculeNet (scaffold split)...")
esol_df = download_esol()
print(f"Total molecules: {len(esol_df)}")
print(f"Split distribution:\n{esol_df['split'].value_counts().to_string()}")
print(f"\nTarget stats (log solubility):")
print(esol_df["log_solubility"].describe().round(3).to_string())

# ─────────────────────────────────────────────────────────────
# 2. COMPUTE MOLECULAR FEATURES
# ─────────────────────────────────────────────────────────────
print("\n[2] Computing Lipinski features + Morgan fingerprints (2048 bits, r=2)...")
df_features = build_feature_matrix(esol_df, fp_radius=2, fp_bits=2048)
df_features.to_csv(DATA_DIR / "processed" / "esol_features.csv", index=False)
print(f"Feature matrix shape: {df_features.shape}")
print(f"Valid molecules: {len(df_features)} / {len(esol_df)}")

# ─────────────────────────────────────────────────────────────
# 3. EDA — DOMAIN STATISTICS
# ─────────────────────────────────────────────────────────────
print("\n[3] EDA — Domain statistics...")

lip_cols = ["mol_weight", "logp", "hbd", "hba", "tpsa", "rotatable_bonds",
            "aromatic_rings", "heavy_atom_count", "qed", "fraction_csp3"]

lip_stats = df_features[lip_cols + ["log_solubility"]].describe().round(3)
print("\nKey molecular descriptors (Lipinski + ADMET):")
print(lip_stats.to_string())

# Lipinski compliance
n_total = len(df_features)
n_pass = df_features["passes_lipinski"].sum()
print(f"\nLipinski Ro5 compliance: {n_pass}/{n_total} ({100*n_pass/n_total:.1f}%)")

violation_counts = df_features["lipinski_violations"].value_counts().sort_index()
print(f"Violation distribution:\n{violation_counts.to_string()}")

# ─────────────────────────────────────────────────────────────
# 4. EDA PLOTS
# ─────────────────────────────────────────────────────────────
print("\n[4] Generating EDA plots...")

fig, axes = plt.subplots(2, 3, figsize=(16, 10))
fig.suptitle("ESOL Dataset — Molecular Space EDA\n(1,128 Drug-like Molecules)", fontsize=14, fontweight="bold")

# 4a. Target distribution
ax = axes[0, 0]
ax.hist(df_features["log_solubility"], bins=40, color="#2ecc71", edgecolor="white", alpha=0.85)
ax.axvline(df_features["log_solubility"].mean(), color="red", linestyle="--", label=f'Mean: {df_features["log_solubility"].mean():.2f}')
ax.set_xlabel("log(mol/L) Solubility")
ax.set_ylabel("Count")
ax.set_title("A. Target Distribution (ESOL)")
ax.legend()

# 4b. Molecular weight distribution
ax = axes[0, 1]
ax.hist(df_features["mol_weight"], bins=40, color="#3498db", edgecolor="white", alpha=0.85)
ax.axvline(500, color="red", linestyle="--", label="Lipinski limit (500 Da)")
ax.set_xlabel("Molecular Weight (Da)")
ax.set_ylabel("Count")
ax.set_title("B. Molecular Weight")
ax.legend()

# 4c. logP distribution
ax = axes[0, 2]
ax.hist(df_features["logp"], bins=40, color="#e74c3c", edgecolor="white", alpha=0.85)
ax.axvline(5, color="black", linestyle="--", label="Lipinski limit (5)")
ax.set_xlabel("logP (lipophilicity)")
ax.set_ylabel("Count")
ax.set_title("C. logP Distribution")
ax.legend()

# 4d. logP vs log_solubility (key relationship)
ax = axes[1, 0]
sc = ax.scatter(df_features["logp"], df_features["log_solubility"],
                c=df_features["mol_weight"], cmap="viridis", alpha=0.5, s=15)
plt.colorbar(sc, ax=ax, label="MW (Da)")
ax.set_xlabel("logP")
ax.set_ylabel("log Solubility (mol/L)")
ax.set_title("D. logP vs Solubility\n(colored by MW)")

# 4e. TPSA vs solubility
ax = axes[1, 1]
sc = ax.scatter(df_features["tpsa"], df_features["log_solubility"],
                c=df_features["hba"], cmap="plasma", alpha=0.5, s=15)
plt.colorbar(sc, ax=ax, label="H-bond acceptors")
ax.set_xlabel("TPSA (Å²)")
ax.set_ylabel("log Solubility (mol/L)")
ax.set_title("E. TPSA vs Solubility\n(colored by HBA)")

# 4f. Lipinski violations vs solubility
ax = axes[1, 2]
viol_groups = df_features.groupby("lipinski_violations")["log_solubility"]
positions = sorted(df_features["lipinski_violations"].unique())
data_to_plot = [viol_groups.get_group(v).values for v in positions]
bp = ax.boxplot(data_to_plot, labels=positions, patch_artist=True,
                medianprops=dict(color="red", linewidth=2))
colors = ["#2ecc71", "#f39c12", "#e74c3c", "#8e44ad"]
for patch, color in zip(bp["boxes"], colors[:len(positions)]):
    patch.set_facecolor(color)
    patch.set_alpha(0.7)
ax.set_xlabel("Lipinski Violations (0=best)")
ax.set_ylabel("log Solubility")
ax.set_title("F. Lipinski Violations vs Solubility\n(key domain insight)")

plt.tight_layout()
plt.savefig(RESULTS_DIR / "phase1_eda_overview.png", dpi=150, bbox_inches="tight")
plt.close()
print("  Saved: results/phase1_eda_overview.png")

# ─────────────────────────────────────────────────────────────
# 5. CORRELATION ANALYSIS — which Lipinski features matter most?
# ─────────────────────────────────────────────────────────────
print("\n[5] Feature correlation with log_solubility...")
corr_cols = ["log_solubility", "logp", "mol_weight", "hbd", "hba", "tpsa",
             "rotatable_bonds", "aromatic_rings", "heavy_atom_count",
             "fraction_csp3", "qed", "molar_refractivity", "lipinski_violations"]
corr_matrix = df_features[corr_cols].corr()

print("\nCorrelation with log_solubility (sorted):")
sol_corr = corr_matrix["log_solubility"].drop("log_solubility").sort_values(key=abs, ascending=False)
for feat, val in sol_corr.items():
    print(f"  {feat:30s}: {val:+.3f}")

fig, ax = plt.subplots(figsize=(11, 9))
mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
sns.heatmap(corr_matrix, mask=mask, annot=True, fmt=".2f", cmap="RdYlGn",
            center=0, vmin=-1, vmax=1, ax=ax,
            linewidths=0.5, cbar_kws={"shrink": 0.8})
ax.set_title("Lipinski Feature Correlation Matrix\n(ESOL Dataset)", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig(RESULTS_DIR / "phase1_correlation_heatmap.png", dpi=150, bbox_inches="tight")
plt.close()
print("  Saved: results/phase1_correlation_heatmap.png")

# ─────────────────────────────────────────────────────────────
# 6. SCAFFOLD DIVERSITY ANALYSIS
# ─────────────────────────────────────────────────────────────
print("\n[6] Murcko scaffold diversity analysis...")
n_scaffolds = df_features["murcko_scaffold"].nunique()
n_singleton_scaffolds = (df_features["murcko_scaffold"].value_counts() == 1).sum()
print(f"Unique Murcko scaffolds: {n_scaffolds}")
print(f"Singleton scaffolds: {n_singleton_scaffolds} ({100*n_singleton_scaffolds/n_scaffolds:.1f}%)")
print(f"Scaffold diversity: {n_scaffolds/len(df_features):.3f} (higher = more diverse)")

top_scaffolds = df_features["murcko_scaffold"].value_counts().head(10)
print("\nTop 10 most common scaffolds (SMILES):")
for smiles, count in top_scaffolds.items():
    print(f"  {count:4d} molecules  {smiles[:60]}")

# ─────────────────────────────────────────────────────────────
# 7. PREPARE TRAIN/TEST SPLITS
# ─────────────────────────────────────────────────────────────
print("\n[7] Preparing train/test sets...")
fp_cols = [c for c in df_features.columns if c.startswith("fp_")]
lip_only_cols = ["logp", "mol_weight", "hbd", "hba", "tpsa", "rotatable_bonds",
                 "aromatic_rings", "heavy_atom_count", "fraction_csp3",
                 "molar_refractivity", "qed", "lipinski_violations"]
all_domain_cols = lip_only_cols  # domain features only (no fingerprints)
combined_cols = lip_only_cols + fp_cols  # domain + fingerprints

train_df = df_features[df_features["split"] == "train"].copy()
val_df = df_features[df_features["split"] == "val"].copy()
test_df = df_features[df_features["split"] == "test"].copy()

X_train_fp = train_df[fp_cols].values
X_val_fp = val_df[fp_cols].values
X_test_fp = test_df[fp_cols].values

X_train_lip = train_df[lip_only_cols].values
X_val_lip = val_df[lip_only_cols].values
X_test_lip = test_df[lip_only_cols].values

X_train_comb = train_df[combined_cols].values
X_val_comb = val_df[combined_cols].values
X_test_comb = test_df[combined_cols].values

y_train = train_df["log_solubility"].values
y_val = val_df["log_solubility"].values
y_test = test_df["log_solubility"].values

print(f"Train: {len(y_train)} | Val: {len(y_val)} | Test: {len(y_test)}")
print(f"Feature sets: FP-only ({len(fp_cols)} bits), Lipinski-only ({len(lip_only_cols)} feats), Combined ({len(combined_cols)} feats)")

# ─────────────────────────────────────────────────────────────
# 8. BASELINE MODELS
# ─────────────────────────────────────────────────────────────
print("\n[8] Running baseline models...")
print("-" * 80)

def evaluate(model, X_tr, y_tr, X_te, y_te, name, feature_set):
    model.fit(X_tr, y_tr)
    y_pred = model.predict(X_te)
    rmse = np.sqrt(mean_squared_error(y_te, y_pred))
    mae = mean_absolute_error(y_te, y_pred)
    r2 = r2_score(y_te, y_pred)
    print(f"  {name:45s} | Features: {feature_set:12s} | RMSE: {rmse:.3f} | MAE: {mae:.3f} | R²: {r2:.3f}")
    return {"model": name, "features": feature_set, "rmse": round(rmse, 4),
            "mae": round(mae, 4), "r2": round(r2, 4)}


results = []

# Baseline 1: Predict mean
mean_pred = np.full_like(y_test, y_train.mean())
mean_rmse = np.sqrt(mean_squared_error(y_test, mean_pred))
print(f"  {'Mean baseline (predict mean)':45s} | Features: {'none':12s} | RMSE: {mean_rmse:.3f} | MAE: --- | R²: 0.000")
results.append({"model": "Mean baseline", "features": "none", "rmse": round(mean_rmse, 4), "mae": None, "r2": 0.0})

# Baseline 2: Linear regression — Lipinski only
scaler = StandardScaler()
X_tr_lip_s = scaler.fit_transform(X_train_lip)
X_te_lip_s = scaler.transform(X_test_lip)
results.append(evaluate(LinearRegression(), X_tr_lip_s, y_train, X_te_lip_s, y_test,
                        "Linear Regression", "Lipinski-only"))

# Baseline 3: Ridge — Lipinski only
results.append(evaluate(Ridge(alpha=1.0), X_tr_lip_s, y_train, X_te_lip_s, y_test,
                        "Ridge Regression", "Lipinski-only"))

# Baseline 4: Linear regression — Morgan FP only
scaler_fp = StandardScaler()
X_tr_fp_s = scaler_fp.fit_transform(X_train_fp)
X_te_fp_s = scaler_fp.transform(X_test_fp)
results.append(evaluate(Ridge(alpha=1.0), X_tr_fp_s, y_train, X_te_fp_s, y_test,
                        "Ridge Regression", "Morgan-FP-only"))

# Baseline 5: Random Forest — Lipinski only
results.append(evaluate(RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1),
                        X_train_lip, y_train, X_test_lip, y_test, "Random Forest", "Lipinski-only"))

# Baseline 6: Random Forest — Morgan FP only
results.append(evaluate(RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1),
                        X_train_fp, y_train, X_test_fp, y_test, "Random Forest", "Morgan-FP-only"))

# Baseline 7: Random Forest — Combined
results.append(evaluate(RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1),
                        X_train_comb, y_train, X_test_comb, y_test, "Random Forest", "Combined"))

# Baseline 8: XGBoost — Lipinski only
results.append(evaluate(xgb.XGBRegressor(n_estimators=100, max_depth=4, learning_rate=0.1,
                                         random_state=42, verbosity=0),
                        X_train_lip, y_train, X_test_lip, y_test, "XGBoost", "Lipinski-only"))

# Baseline 9: XGBoost — Morgan FP
results.append(evaluate(xgb.XGBRegressor(n_estimators=100, max_depth=4, learning_rate=0.1,
                                         random_state=42, verbosity=0),
                        X_train_fp, y_train, X_test_fp, y_test, "XGBoost", "Morgan-FP-only"))

# Baseline 10: XGBoost — Combined
results.append(evaluate(xgb.XGBRegressor(n_estimators=100, max_depth=4, learning_rate=0.1,
                                         random_state=42, verbosity=0),
                        X_train_comb, y_train, X_test_comb, y_test, "XGBoost", "Combined"))

# ─────────────────────────────────────────────────────────────
# 9. COMPARISON TABLE
# ─────────────────────────────────────────────────────────────
print("\n" + "=" * 80)
print("BASELINE MODEL COMPARISON (primary metric: RMSE, lower is better)")
print("Published SOTA for ESOL: Attentive FP ≈ 0.584 RMSE | Random Forest ≈ 0.73 RMSE")
print("=" * 80)

results_df = pd.DataFrame(results).sort_values("rmse")
print(results_df.to_string(index=False))

champion = results_df.iloc[0]
print(f"\nChampion baseline: {champion['model']} ({champion['features']}) — RMSE: {champion['rmse']:.3f}")

# ─────────────────────────────────────────────────────────────
# 10. KEY FINDING: Domain features vs fingerprints breakdown
# ─────────────────────────────────────────────────────────────
print("\n[KEY FINDING] Lipinski features vs Morgan fingerprints vs combined:")
rf_lip = results_df[(results_df["model"] == "Random Forest") & (results_df["features"] == "Lipinski-only")].iloc[0]
rf_fp = results_df[(results_df["model"] == "Random Forest") & (results_df["features"] == "Morgan-FP-only")].iloc[0]
rf_comb = results_df[(results_df["model"] == "Random Forest") & (results_df["features"] == "Combined")].iloc[0]
xgb_lip = results_df[(results_df["model"] == "XGBoost") & (results_df["features"] == "Lipinski-only")].iloc[0]
xgb_fp = results_df[(results_df["model"] == "XGBoost") & (results_df["features"] == "Morgan-FP-only")].iloc[0]
xgb_comb = results_df[(results_df["model"] == "XGBoost") & (results_df["features"] == "Combined")].iloc[0]

print(f"  RF  Lipinski-only RMSE: {rf_lip['rmse']:.3f}")
print(f"  RF  Morgan FP-only RMSE: {rf_fp['rmse']:.3f}")
print(f"  RF  Combined RMSE: {rf_comb['rmse']:.3f}")
print(f"  XGB Lipinski-only RMSE: {xgb_lip['rmse']:.3f}")
print(f"  XGB Morgan FP-only RMSE: {xgb_fp['rmse']:.3f}")
print(f"  XGB Combined RMSE: {xgb_comb['rmse']:.3f}")

fp_advantage_rf = rf_lip["rmse"] - rf_fp["rmse"]
fp_advantage_xgb = xgb_lip["rmse"] - xgb_fp["rmse"]
combined_vs_fp_rf = rf_fp["rmse"] - rf_comb["rmse"]
combined_vs_fp_xgb = xgb_fp["rmse"] - xgb_comb["rmse"]
print(f"\n  FP advantage over Lipinski (RF): {fp_advantage_rf:+.3f} RMSE")
print(f"  FP advantage over Lipinski (XGB): {fp_advantage_xgb:+.3f} RMSE")
print(f"  Combined vs FP-only (RF):  {combined_vs_fp_rf:+.3f} RMSE (negative = combined better)")
print(f"  Combined vs FP-only (XGB): {combined_vs_fp_xgb:+.3f} RMSE (negative = combined better)")

# ─────────────────────────────────────────────────────────────
# 11. KEY FINDING: logP is the #1 predictor
# ─────────────────────────────────────────────────────────────
print("\n[KEY FINDING] Top Lipinski correlations with solubility:")
for feat, val in sol_corr.items():
    print(f"  {feat:30s}: {val:+.3f}")

# ─────────────────────────────────────────────────────────────
# 12. PREDICTION SCATTER PLOTS
# ─────────────────────────────────────────────────────────────
print("\n[9] Generating prediction plots for best models...")

# Refit champions for plotting
rf_champion = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
rf_champion.fit(X_train_comb, y_train)
y_pred_rf = rf_champion.predict(X_test_comb)

xgb_champion = xgb.XGBRegressor(n_estimators=100, max_depth=4, learning_rate=0.1,
                                  random_state=42, verbosity=0)
xgb_champion.fit(X_train_comb, y_train)
y_pred_xgb = xgb_champion.predict(X_test_comb)

# Simple logP-only linear model (interpretable insight)
from sklearn.linear_model import LinearRegression as LR
logp_vals_train = train_df[["logp"]].values
logp_vals_test = test_df[["logp"]].values
lr_logp = LR().fit(logp_vals_train, y_train)
y_pred_logp = lr_logp.predict(logp_vals_test)
rmse_logp = np.sqrt(mean_squared_error(y_test, y_pred_logp))
print(f"  logP-only linear model RMSE: {rmse_logp:.3f}")

fig, axes = plt.subplots(1, 3, figsize=(17, 5))
fig.suptitle("ESOL Phase 1: Baseline Prediction Quality", fontsize=13, fontweight="bold")

for ax, y_pred, name, rmse_val, color in zip(
    axes,
    [y_pred_logp, y_pred_rf, y_pred_xgb],
    ["Linear(logP only)", "RF (Combined)", "XGBoost (Combined)"],
    [rmse_logp, rf_comb["rmse"], xgb_comb["rmse"]],
    ["#e74c3c", "#2ecc71", "#3498db"]
):
    ax.scatter(y_test, y_pred, alpha=0.5, s=20, color=color)
    lo = min(y_test.min(), y_pred.min()) - 0.5
    hi = max(y_test.max(), y_pred.max()) + 0.5
    ax.plot([lo, hi], [lo, hi], "k--", alpha=0.8, label="Perfect")
    ax.set_xlabel("Actual log Solubility")
    ax.set_ylabel("Predicted log Solubility")
    ax.set_title(f"{name}\nRMSE={rmse_val:.3f}")
    ax.legend()

plt.tight_layout()
plt.savefig(RESULTS_DIR / "phase1_prediction_scatter.png", dpi=150, bbox_inches="tight")
plt.close()
print("  Saved: results/phase1_prediction_scatter.png")

# ─────────────────────────────────────────────────────────────
# 13. SAVE RESULTS
# ─────────────────────────────────────────────────────────────
print("\n[10] Saving results...")

dataset_stats = {
    "dataset": "ESOL (Delaney)",
    "source": "MoleculeNet via DeepChem",
    "total_molecules": int(len(df_features)),
    "train": int(len(train_df)),
    "val": int(len(val_df)),
    "test": int(len(test_df)),
    "target": "log_solubility (log(mol/L))",
    "primary_metric": "RMSE (lower is better)",
    "target_mean": float(df_features["log_solubility"].mean()),
    "target_std": float(df_features["log_solubility"].std()),
    "target_min": float(df_features["log_solubility"].min()),
    "target_max": float(df_features["log_solubility"].max()),
    "lipinski_compliance_pct": float(100 * n_pass / n_total),
    "unique_murcko_scaffolds": int(n_scaffolds),
    "scaffold_diversity": float(n_scaffolds / len(df_features)),
    "top_correlation_with_solubility": {
        feat: float(val) for feat, val in sol_corr.items()
    },
    "published_benchmarks": {
        "Random Forest (Duvenaud et al., 2015)": 0.73,
        "MPNN (Gilmer et al., 2017)": 0.72,
        "AttentiveFP (Xiong et al., 2020)": 0.584,
        "D-MPNN (Chemprop, Yang et al., 2019)": 0.555,
    },
    "date": "2026-04-06",
    "phase": 1,
}

metrics_out = {
    "phase": 1,
    "date": "2026-04-06",
    "dataset_stats": dataset_stats,
    "experiments": results_df.to_dict("records"),
    "key_findings": {
        "champion_baseline": f"{champion['model']} ({champion['features']})",
        "champion_rmse": float(champion["rmse"]),
        "mean_baseline_rmse": float(mean_rmse),
        "logp_only_rmse": float(rmse_logp),
        "logp_correlation_with_solubility": float(sol_corr["logp"]),
        "fp_advantage_over_lipinski_rf": float(fp_advantage_rf),
        "combined_advantage_over_fp_rf": float(combined_vs_fp_rf),
        "lipinski_compliance_pct": float(100 * n_pass / n_total),
        "top_correlated_feature": str(sol_corr.index[0]),
        "top_correlation_value": float(sol_corr.iloc[0]),
    },
}

metrics_path = RESULTS_DIR / "metrics.json"
if metrics_path.exists():
    with open(metrics_path) as f:
        existing = json.load(f)
    if isinstance(existing, list):
        existing.append(metrics_out)
    else:
        existing = [existing, metrics_out]
    with open(metrics_path, "w") as f:
        json.dump(existing, f, indent=2)
else:
    with open(metrics_path, "w") as f:
        json.dump([metrics_out], f, indent=2)

print("  Saved: results/metrics.json")

# EXPERIMENT LOG
exp_log_path = RESULTS_DIR / "EXPERIMENT_LOG.md"
with open(exp_log_path, "w") as f:
    f.write("# Drug Molecule Property Prediction — Experiment Log\n\n")
    f.write(f"**Project:** DL-1 Drug Molecule Property Prediction\n")
    f.write(f"**Dataset:** ESOL (MoleculeNet) — {len(df_features)} molecules\n")
    f.write(f"**Primary Metric:** RMSE (lower is better)\n")
    f.write(f"**Published SOTA:** AttentiveFP RMSE=0.584\n\n")
    f.write("---\n\n")
    f.write("## Phase 1: EDA + Baseline Models (2026-04-06)\n\n")
    f.write("| Rank | Model | Feature Set | RMSE | MAE | R² |\n")
    f.write("|------|-------|-------------|------|-----|----|\n")
    for i, row in results_df.reset_index(drop=True).iterrows():
        mae_str = f"{row['mae']:.4f}" if row['mae'] is not None else "N/A"
        f.write(f"| {i+1} | {row['model']} | {row['features']} | {row['rmse']:.4f} | {mae_str} | {row['r2']:.4f} |\n")
    f.write(f"\n**Published benchmarks for reference:**\n")
    f.write("| Paper | Model | RMSE |\n")
    f.write("|-------|-------|------|\n")
    f.write("| Duvenaud et al., 2015 | Graph CNN | 0.73 |\n")
    f.write("| Gilmer et al., 2017 | MPNN | 0.72 |\n")
    f.write("| Yang et al., 2019 | D-MPNN (Chemprop) | 0.555 |\n")
    f.write("| Xiong et al., 2020 | AttentiveFP | 0.584 |\n\n")

print("  Saved: results/EXPERIMENT_LOG.md")

print("\n" + "=" * 80)
print("PHASE 1 COMPLETE — SUMMARY")
print("=" * 80)
print(f"Champion: {champion['model']} ({champion['features']}) — RMSE: {champion['rmse']:.3f}")
print(f"Mean baseline RMSE: {mean_rmse:.3f} | logP-only RMSE: {rmse_logp:.3f}")
print(f"logP is the strongest single predictor: r={float(sol_corr['logp']):.3f} with solubility")
print(f"Lipinski compliance: {100*n_pass/n_total:.1f}% of molecules")
print(f"Dataset diversity (scaffold): {n_scaffolds} unique Murcko scaffolds / {len(df_features)} molecules")
print(f"\nFiles created:")
print("  data/processed/esol_features.csv")
print("  results/phase1_eda_overview.png")
print("  results/phase1_correlation_heatmap.png")
print("  results/phase1_prediction_scatter.png")
print("  results/metrics.json")
print("  results/EXPERIMENT_LOG.md")
