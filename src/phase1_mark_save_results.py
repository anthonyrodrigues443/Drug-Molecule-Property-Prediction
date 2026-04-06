"""
Phase 1 Mark: Save results and generate plots from computed experiment data.
(Re-runs the core computation and saves outputs correctly)
"""

import os
import sys
import json
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

os.environ["BABEL_DATADIR"] = (
    r"C:\Users\antho\AppData\Local\Programs\Python\Python311\Lib"
    r"\site-packages\openbabel\bin\data"
)
os.environ["PYTHONUTF8"] = "1"

from openbabel import pybel
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
import xgboost as xgb

warnings.filterwarnings("ignore")

ROOT = Path(__file__).parent.parent
DATA_DIR = ROOT / "data"
RESULTS_DIR = ROOT / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Load data
df = pd.read_csv(DATA_DIR / "raw" / "esol_delaney.csv")
df.columns = [
    "compound_id", "delaney_predicted_logS", "min_degree",
    "mol_weight", "hbd", "ring_count", "rotatable_bonds",
    "polar_surface_area", "measured_logS", "smiles",
]
df = df.dropna(subset=["smiles", "measured_logS"]).reset_index(drop=True)

DOMAIN_FEATS = ["mol_weight", "hbd", "ring_count", "rotatable_bonds", "polar_surface_area", "min_degree"]
X_domain = df[DOMAIN_FEATS].values
y = df["measured_logS"].values

# Train/test split
decile_labels = pd.qcut(y, q=10, labels=False, duplicates="drop")
X_idx = np.arange(len(df))
train_idx, test_idx = train_test_split(X_idx, test_size=0.20, random_state=42, stratify=decile_labels)

def smiles_to_fingerprint(smiles, fp_type):
    try:
        mol = pybel.readstring("smi", smiles)
        fp_obj = mol.calcfp(fp_type)
        fp_words = fp_obj.fp
        if not fp_words:
            return None
        bits = []
        for word in fp_words:
            for bit_pos in range(32):
                bits.append((word >> bit_pos) & 1)
        return np.array(bits, dtype=np.float32)
    except Exception:
        return None

# Use only ECFP fingerprints (FP3/FP4 have 32-bit degenerate outputs)
FP_TYPES = ["ecfp0", "ecfp2", "ecfp4", "ecfp6", "ecfp8", "fp2"]

print("Computing fingerprints...", file=sys.stderr)
fps_dict = {}
for fp_type in FP_TYPES:
    fps = [smiles_to_fingerprint(smi, fp_type) for smi in df["smiles"]]
    fps_dict[fp_type] = fps

all_valid = [i for i in range(len(df)) if all(fps_dict[fp_type][i] is not None for fp_type in FP_TYPES)]
train_idx_valid = [i for i in train_idx if i in set(all_valid)]
test_idx_valid  = [i for i in test_idx  if i in set(all_valid)]

y_train = y[train_idx_valid]
y_test  = y[test_idx_valid]

def eval_model(model, X_tr, y_tr, X_te, y_te, name, features, n_feats):
    model.fit(X_tr, y_tr)
    preds = model.predict(X_te)
    rmse = round(float(np.sqrt(mean_squared_error(y_te, preds))), 4)
    mae  = round(float(mean_absolute_error(y_te, preds)), 4)
    r2   = round(float(r2_score(y_te, preds)), 4)
    return {"model": name, "features": features, "n_features": n_feats,
            "rmse": rmse, "mae": mae, "r2": r2}

results = []

# Domain features
X_dom_train = X_domain[train_idx_valid]
X_dom_test  = X_domain[test_idx_valid]
scaler_dom = StandardScaler()
X_dom_tr_sc = scaler_dom.fit_transform(X_dom_train)
X_dom_te_sc = scaler_dom.transform(X_dom_test)

results.append(eval_model(Ridge(alpha=1.0), X_dom_tr_sc, y_train, X_dom_te_sc, y_test,
               "Ridge (Delaney 6-feat)", "delaney_6", 6))
results.append(eval_model(RandomForestRegressor(n_estimators=200, random_state=42),
               X_dom_train, y_train, X_dom_test, y_test,
               "RF (Delaney 6-feat)", "delaney_6", 6))
results.append(eval_model(xgb.XGBRegressor(n_estimators=300, learning_rate=0.05, max_depth=5,
               subsample=0.8, random_state=42, verbosity=0),
               X_dom_train, y_train, X_dom_test, y_test,
               "XGBoost (Delaney 6-feat)", "delaney_6", 6))

# Delaney equation baseline
y_delaney = df["delaney_predicted_logS"].values[test_idx_valid]
rmse_del = round(float(np.sqrt(mean_squared_error(y_test, y_delaney))), 4)
r2_del   = round(float(r2_score(y_test, y_delaney)), 4)
results.append({"model": "Delaney Equation (4-feat)", "features": "delaney_eq",
                "n_features": 4, "rmse": rmse_del, "mae": 0.0, "r2": r2_del})

# Fingerprint experiments
print("Running fingerprint models...", file=sys.stderr)
for fp_type in FP_TYPES:
    X_fp = np.array([fps_dict[fp_type][i] for i in train_idx_valid])
    X_fp_test = np.array([fps_dict[fp_type][i] for i in test_idx_valid])
    n_feats = X_fp.shape[1]

    scaler_fp = StandardScaler()
    X_fp_sc = scaler_fp.fit_transform(X_fp)
    X_fp_te_sc = scaler_fp.transform(X_fp_test)

    results.append(eval_model(Ridge(alpha=10.0), X_fp_sc, y_train, X_fp_te_sc, y_test,
                   f"Ridge ({fp_type})", fp_type, n_feats))
    results.append(eval_model(RandomForestRegressor(n_estimators=200, random_state=42),
                   X_fp, y_train, X_fp_test, y_test,
                   f"RF ({fp_type})", fp_type, n_feats))
    results.append(eval_model(xgb.XGBRegressor(n_estimators=300, learning_rate=0.05, max_depth=5,
                   subsample=0.8, random_state=42, verbosity=0),
                   X_fp, y_train, X_fp_test, y_test,
                   f"XGBoost ({fp_type})", fp_type, n_feats))

df_results = pd.DataFrame(results)

# -------------------------------------------------------------------------
# Extract key numbers
# -------------------------------------------------------------------------
xgb_res = df_results[df_results["model"].str.startswith("XGBoost")]
xgb_domain_rmse = xgb_res[xgb_res["features"] == "delaney_6"]["rmse"].values[0]
ecfp4_rmse = xgb_res[xgb_res["features"] == "ecfp4"]["rmse"].values[0]
fp2_rmse   = xgb_res[xgb_res["features"] == "fp2"]["rmse"].values[0]
delaney_eq_rmse = rmse_del

fp_rows = xgb_res[xgb_res["features"].str.startswith("ecfp")]
ecfp_rmse_list = fp_rows.sort_values("features")["rmse"].tolist()
ecfp_types     = ["ECFP0", "ECFP2", "ECFP4", "ECFP6", "ECFP8"]
ecfp_radii     = [0, 1, 2, 3, 4]

best_fp_row = xgb_res[xgb_res["features"].isin(FP_TYPES)].sort_values("rmse").iloc[0]

# -------------------------------------------------------------------------
# Plots
# -------------------------------------------------------------------------
print("Generating plots...", file=sys.stderr)
fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
fig.patch.set_facecolor("#0D1117")

# -- Plot 1: All fingerprint types vs domain --
ax = axes[0]
ax.set_facecolor("#161B22")

bar_data = xgb_res[xgb_res["features"].isin(FP_TYPES + ["delaney_6"])].copy()
bar_data["label"] = bar_data["features"].str.upper().replace({"DELANEY_6": "Domain\n6-feat"})
bar_data = bar_data.sort_values("rmse")

bar_colors = []
for f in bar_data["features"]:
    if f == "delaney_6":
        bar_colors.append("#4CAF50")
    elif f.startswith("ecfp"):
        bar_colors.append("#2196F3")
    else:
        bar_colors.append("#FF9800")

bars = ax.bar(range(len(bar_data)), bar_data["rmse"], color=bar_colors, alpha=0.85, edgecolor="#30363D")
ax.axhline(y=0.385, color="#9C27B0", linestyle=":", linewidth=2.0,
           label="Anthony's Lipinski-12 RMSE=0.385")

ax.set_xticks(range(len(bar_data)))
ax.set_xticklabels(bar_data["label"], rotation=30, ha="right", color="white", fontsize=9)
ax.set_ylabel("RMSE (lower = better)", color="white")
ax.set_title("XGBoost: Circular vs Path vs Domain Features\n(ESOL Solubility, RMSE — lower is better)",
             color="white", fontsize=10)
ax.tick_params(colors="white")
ax.spines["bottom"].set_color("#444")
ax.spines["left"].set_color("#444")
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

from matplotlib.patches import Patch
legend_elements = [
    Patch(facecolor="#4CAF50", label="Domain features (Delaney 6-feat)"),
    Patch(facecolor="#2196F3", label="Circular (ECFP) fingerprints"),
    Patch(facecolor="#FF9800", label="Path-based (FP2)"),
    plt.Line2D([0], [0], color="#9C27B0", linestyle=":", linewidth=2.0, label="Anthony: Lipinski-12 XGB (RMSE=0.385)")
]
ax.legend(handles=legend_elements, fontsize=8, facecolor="#161B22", labelcolor="white")

for i, bar in enumerate(bars):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
            f"{bar.get_height():.3f}", ha="center", va="bottom",
            color="white", fontsize=8)

# -- Plot 2: ECFP radius sensitivity --
ax2 = axes[1]
ax2.set_facecolor("#161B22")

ax2.plot(ecfp_radii, ecfp_rmse_list, "o-", color="#2196F3", linewidth=2.5, markersize=12,
         markerfacecolor="#2196F3", markeredgecolor="white", markeredgewidth=1.5,
         label="ECFP at different radii (XGBoost)")
ax2.axhline(y=xgb_domain_rmse, color="#4CAF50", linestyle="--", linewidth=2.0,
            label=f"Domain 6-feat (RMSE={xgb_domain_rmse:.3f})")
ax2.axhline(y=delaney_eq_rmse, color="#00BCD4", linestyle="-.", linewidth=2.0,
            label=f"Delaney Eq (RMSE={delaney_eq_rmse:.3f})")
ax2.axhline(y=0.385, color="#9C27B0", linestyle=":", linewidth=2.0,
            label=f"Anthony Lipinski-12 (RMSE=0.385)")

for r, rmse in zip(ecfp_radii, ecfp_rmse_list):
    ax2.annotate(f"r={r}\n{rmse:.3f}", xy=(r, rmse),
                 xytext=(10, 8), textcoords="offset points",
                 color="white", fontsize=8, ha="left")

ax2.set_xlabel("ECFP Radius (neighborhood size)", color="white")
ax2.set_ylabel("RMSE (lower = better)", color="white")
ax2.set_title("Does ECFP Radius Matter?\nSolubility prediction vs fingerprint 'zoom level'",
              color="white", fontsize=10)
ax2.legend(fontsize=8, facecolor="#161B22", labelcolor="white")
ax2.tick_params(colors="white")
ax2.spines["bottom"].set_color("#444")
ax2.spines["left"].set_color("#444")
ax2.spines["top"].set_visible(False)
ax2.spines["right"].set_visible(False)
ax2.set_xticks(ecfp_radii)
ax2.set_xticklabels([f"r={r}" for r in ecfp_radii], color="white")

plt.tight_layout(pad=2.0)
plt.savefig(RESULTS_DIR / "phase1_mark_fingerprint_comparison.png", dpi=150, bbox_inches="tight",
            facecolor="#0D1117")
plt.close()

# Plot 2: Full heatmap
model_names = ["Ridge", "RF", "XGBoost"]
feat_names = ["delaney_6"] + FP_TYPES
feat_labels = ["Domain 6-feat"] + [f.upper() for f in FP_TYPES]

heatmap_data = np.zeros((len(model_names), len(feat_names)))
for i, mname in enumerate(model_names):
    for j, fname in enumerate(feat_names):
        matches = df_results[(df_results["features"] == fname) &
                             (df_results["model"].str.startswith(mname))]
        heatmap_data[i, j] = matches["rmse"].values[0] if len(matches) else np.nan

fig, ax = plt.subplots(figsize=(12, 4))
fig.patch.set_facecolor("#0D1117")
ax.set_facecolor("#161B22")
sns.heatmap(heatmap_data, annot=True, fmt=".3f", cmap="RdYlGn_r",
            xticklabels=feat_labels, yticklabels=model_names,
            ax=ax, vmin=0.8, vmax=1.5, linewidths=0.5,
            annot_kws={"size": 11})
ax.set_title("RMSE Heatmap: Models x Feature Types (ESOL — lower is better)",
             color="white", fontsize=12, pad=12)
ax.tick_params(colors="white", labelsize=10)
ax.set_xlabel("Feature Type", color="white")
ax.set_ylabel("Model", color="white")
plt.tight_layout()
plt.savefig(RESULTS_DIR / "phase1_mark_model_heatmap.png", dpi=150, bbox_inches="tight",
            facecolor="#0D1117")
plt.close()

# -------------------------------------------------------------------------
# Print results table (ASCII only)
# -------------------------------------------------------------------------
all_xgb = df_results[df_results["model"].str.startswith("XGBoost") |
                      (df_results["model"] == "Delaney Equation (4-feat)")].copy()
all_xgb = all_xgb.sort_values("rmse")

print("\n" + "=" * 75)
print("RESULTS TABLE (XGBoost + Delaney Equation)")
print("=" * 75)
print(f"{'Rank':<5} {'Model':<35} {'Features':<15} {'RMSE':<8} {'R2':<8}")
print("-" * 75)
for rank, (_, row) in enumerate(all_xgb.iterrows(), 1):
    print(f"{rank:<5} {row['model']:<35} {row['features']:<15} {row['rmse']:<8.4f} {row['r2']:<8.4f}")
print("-" * 75)

print(f"\nDomain 6-feat XGB:          RMSE={xgb_domain_rmse:.4f}")
print(f"Delaney Eq (4-feat):        RMSE={delaney_eq_rmse:.4f}")
print(f"ECFP4 (Anthony's radius):   RMSE={ecfp4_rmse:.4f}")
print(f"FP2 (path-based):           RMSE={fp2_rmse:.4f}")
print(f"Anthony Lipinski-12:        RMSE=0.3846 (reported)")
print(f"\nECFP radius performance:")
for r, rmse in zip(ecfp_radii, ecfp_rmse_list):
    print(f"  r={r} (ECFP{r*2}): RMSE={rmse:.4f}")

# -------------------------------------------------------------------------
# Save results JSON
# -------------------------------------------------------------------------
metrics_path = RESULTS_DIR / "metrics.json"
try:
    with open(metrics_path) as f:
        existing = json.load(f)
except (FileNotFoundError, json.JSONDecodeError):
    existing = []

mark_entry = {
    "phase": 1,
    "author": "Mark",
    "date": "2026-04-06",
    "dataset": "ESOL (Delaney 2004) - raw logS scale",
    "split": "80/20 stratified random (seed=42)",
    "primary_metric": "RMSE",
    "note": "Raw logS scale [-11.6, 1.58] — Anthony uses DeepChem normalized scale. Relative comparisons within this experiment are valid.",
    "experiments": results,
    "key_findings": {
        "xgb_domain_6feat_rmse": float(xgb_domain_rmse),
        "delaney_equation_rmse": float(delaney_eq_rmse),
        "ecfp4_rmse": float(ecfp4_rmse),
        "fp2_path_rmse": float(fp2_rmse),
        "best_fingerprint_type": str(best_fp_row["features"]),
        "best_fingerprint_rmse": float(best_fp_row["rmse"]),
        "ecfp_radius_findings": {
            f"r{r}": float(rmse) for r, rmse in zip(ecfp_radii, ecfp_rmse_list)
        },
        "radius_sweet_spot": int(ecfp_radii[ecfp_rmse_list.index(min(ecfp_rmse_list))]),
        "domain_vs_fingerprint_gap": float(best_fp_row["rmse"] - xgb_domain_rmse),
    }
}
existing.append(mark_entry)
with open(metrics_path, "w") as f:
    json.dump(existing, f, indent=2)
print("\nSaved results/metrics.json")

# Save experiment log entry
exp_log_path = RESULTS_DIR / "EXPERIMENT_LOG.md"
with open(exp_log_path, "a", encoding="utf-8") as f:
    f.write("\n\n## 2026-04-06 | Phase 1 (Mark) — Fingerprint Radius & Type Sensitivity\n\n")
    f.write("| Rank | Model | Features | n_feats | RMSE | R2 |\n")
    f.write("|------|-------|----------|---------|------|----|\n")
    for rank, (_, row) in enumerate(all_xgb.iterrows(), 1):
        f.write(f"| {rank} | {row['model']} | {row['features']} | {row.get('n_features', '-')} | {row['rmse']} | {row['r2']} |\n")
print("Updated results/EXPERIMENT_LOG.md")

print("\nPhase 1 Mark: DONE. All results saved.")
