"""Build phase4_hyperparameter_tuning.ipynb from the tuning script."""
import nbformat as nbf
from pathlib import Path

BASE = Path('/Users/anthonyrodrigues/Desktop/YC-Portfolio-Projects/Drug-Molecule-Property-Prediction')
script = (BASE / 'src' / 'phase4_anthony_tuning.py').read_text()

nb = nbf.v4.new_notebook()
cells = []

cells.append(nbf.v4.new_markdown_cell("""# Phase 4 (Anthony): GIN+Edge Tuning + GNN-CatBoost Ensemble

**Date:** 2026-04-09 · **Researcher:** Anthony Rodrigues · **Dataset:** ogbg-molhiv (scaffold split)

## Building on Mark's Phase 4
Mark's Phase 4 contributions:
- Optuna tuned CatBoost MI-400 (40 trials): +0.027 test AUC but val-test gap widened
- K=400 stability confirmed (std=0.0040 across bootstraps)
- Deep error analysis: Lipinski violators have 2x higher recall (0.828 vs 0.400)
- Feature importance: MACCS keys 2.4x above pool share

**My complementary angle:**
1. GIN+Edge Optuna tuning — does the GNN paradigm respond to hyperparameter tuning?
2. Save test predictions from both GIN and CatBoost for future ensemble work
3. GIN + CatBoost ensemble — do they make complementary errors?
4. Error overlap analysis — same molecules or different failure modes?

## Research References
1. [Akiba et al. 2019] Optuna: TPE sampler for efficient hyperparameter search
2. [OGB Leaderboard] GIN-VN: 5 layers, 300 dim — upper bound for GNN search space
3. [Dietterich 2000] Ensemble methods: diverse models combined yield better generalization
"""))

# Split by separator
sections = script.split('# ═══════')
sections = [s for s in sections if s.strip()]

# Put all code cells
section_titles = [
    "## Data Loading + GIN+Edge Model Definition",
    "## Data Loading (OGB)",
    "## GIN+Edge Architecture",
    "## Experiment 4.1: GIN+Edge Optuna Tuning (8 trials)",
    "## Experiment 4.2: CatBoost MI-400 (Mark's tuned config)",
    "## Experiment 4.3: GIN + CatBoost Ensemble\n\nKey question: If GIN learns graph topology and CatBoost learns chemistry features, do they make different errors? An ensemble could beat both.",
    "## Experiment 4.4: Error Overlap Analysis\n\nDo GIN and CatBoost fail on the SAME molecules or DIFFERENT ones? Low overlap = high ensemble potential.",
    "## Plots",
    "## Save Results & Predictions\n\nSave test predictions from both models for Phase 5 ablation and LLM comparison.",
]

for i, section in enumerate(sections):
    code = ('# ═══════' + section).strip() if i > 0 else section.strip()
    if i < len(section_titles):
        cells.append(nbf.v4.new_markdown_cell(section_titles[i]))
    cells.append(nbf.v4.new_code_cell(code))

cells.append(nbf.v4.new_markdown_cell("""## Key Findings

1. **GIN+Edge tuning:** How much did Optuna improve over the Phase 3 default (0.7860)?
2. **Ensemble result:** Does combining GIN + CatBoost beat either alone?
3. **Error overlap:** Do the models fail on the same or different molecules?
4. **Predictions saved** for Phase 5 ensemble refinement and LLM comparison.
"""))

nb['cells'] = cells
out = BASE / 'notebooks' / 'phase4_hyperparameter_tuning.ipynb'
nbf.write(nb, str(out))
print(f'Wrote {out}')
