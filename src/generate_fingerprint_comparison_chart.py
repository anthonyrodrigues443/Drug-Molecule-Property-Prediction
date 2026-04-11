"""Generate phase1_mark_fingerprint_comparison.png from hardcoded experiment results."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path

RESULTS_DIR = Path(__file__).parent.parent / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# ── Data from Phase 1 ECFP radius sensitivity experiment (ESOL/Delaney dataset) ──
models = [
    "XGBoost\n(ECFP0, r=0, 4096-bit)",
    "XGBoost\n(ECFP4, r=2, 4096-bit)",
    "XGBoost\n(FP2 path, 1024-bit)",
    "XGBoost\n(Delaney 6-feat)",
    "Delaney Equation\n(4 features, 2004)",
]
rmse = [1.3582, 1.1268, 1.1025, 0.9318, 0.9134]
r2   = [0.578,  0.710,  0.722,  0.802,  0.809]

# Colour by representation type
colors = [
    "#FF5722",  # circular FP — bad
    "#FF9800",  # circular FP — best radius
    "#FFC107",  # path FP
    "#64B5F6",  # domain + XGB
    "#1565C0",  # domain equation (champion)
]

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle(
    "Fingerprint Radius Sensitivity vs Domain Baseline\n"
    "Solubility Prediction — ESOL/Delaney Dataset (1,128 compounds)",
    fontsize=13, fontweight="bold", y=1.01
)

# ── Left: RMSE (lower is better) ──────────────────────────────────────
ax = axes[0]
bars = ax.barh(range(len(models)), rmse, color=colors, edgecolor="white", linewidth=0.6, height=0.6)
ax.axvline(0.9134, color="#1565C0", ls="--", lw=1.8, label="Delaney baseline (0.913)")
ax.set_yticks(range(len(models)))
ax.set_yticklabels(models, fontsize=9)
ax.set_xlabel("RMSE  (lower is better)", fontsize=10)
ax.set_title("RMSE Comparison", fontsize=11)
ax.set_xlim(0.7, 1.55)
ax.invert_yaxis()

# Value labels
for i, v in enumerate(rmse):
    ax.text(v + 0.01, i, f"{v:.3f}", va="center", fontsize=9, fontweight="bold")

ax.legend(fontsize=8, loc="lower right")

# ── Right: R² (higher is better) ──────────────────────────────────────
ax2 = axes[1]
ax2.barh(range(len(models)), r2, color=colors, edgecolor="white", linewidth=0.6, height=0.6)
ax2.axvline(0.809, color="#1565C0", ls="--", lw=1.8, label="Delaney baseline (0.809)")
ax2.set_yticks(range(len(models)))
ax2.set_yticklabels(models, fontsize=9)
ax2.set_xlabel("R²  (higher is better)", fontsize=10)
ax2.set_title("R² Comparison", fontsize=11)
ax2.set_xlim(0.4, 0.92)
ax2.invert_yaxis()

for i, v in enumerate(r2):
    ax2.text(v + 0.005, i, f"{v:.3f}", va="center", fontsize=9, fontweight="bold")

ax2.legend(fontsize=8, loc="lower right")

# ── Legend ─────────────────────────────────────────────────────────────
legend_patches = [
    mpatches.Patch(color="#1565C0", label="Domain equation (Delaney 4-feat)"),
    mpatches.Patch(color="#64B5F6", label="Domain features + XGBoost"),
    mpatches.Patch(color="#FFC107", label="Path fingerprints (FP2)"),
    mpatches.Patch(color="#FF9800", label="Circular FP — optimal radius (ECFP4, r=2)"),
    mpatches.Patch(color="#FF5722", label="Circular FP — no context (ECFP0, r=0)"),
]
fig.legend(
    handles=legend_patches,
    loc="lower center",
    ncol=3,
    fontsize=8,
    bbox_to_anchor=(0.5, -0.08),
    frameon=True,
)

plt.tight_layout()
out = RESULTS_DIR / "phase1_mark_fingerprint_comparison.png"
plt.savefig(out, dpi=150, bbox_inches="tight")
print(f"Saved: {out}")
