"""Build phase4_hyperparameter_tuning.ipynb from the experiment script."""
import nbformat as nbf
from pathlib import Path

BASE = Path('/Users/anthonyrodrigues/Desktop/YC-Portfolio-Projects/Drug-Molecule-Property-Prediction')
script = (BASE / 'src' / 'phase4_hyperparameter_tuning.py').read_text()

nb = nbf.v4.new_notebook()
cells = []

# Markdown intro
cells.append(nbf.v4.new_markdown_cell("""# Phase 4: Hyperparameter Tuning + Error Analysis — Drug Molecule Property Prediction

**Date:** 2026-04-09 · **Researcher:** Anthony Rodrigues · **Dataset:** ogbg-molhiv (scaffold split)

## Building on Phase 3
- **Phase 3 Anthony champion:** GIN+Edge = 0.7860 AUC (first GNN to beat CatBoost)
- **Phase 3 Mark champion:** CatBoost MI-top-400 = 0.8105 AUC (overall leader)
- **SOTA target:** 0.8476 (GIN-VN from OGB leaderboard)

## Today's Questions
1. **Can Optuna tuning close the gap?** GIN+Edge default config (128d, 3 layers, 0.5 drop) vs tuned. How much AUC does hyperparameter choice leave on the table?
2. **Can CatBoost be pushed further?** Mark's MI-top-400 used default CatBoost. Tuning depth, lr, regularization could help.
3. **Where does the model fail?** Scaffold-based error analysis — do rare scaffolds (singletons) cause most errors?
4. **Is more data the answer?** Learning curves — is CatBoost saturated at 32K samples?

## Research References
1. [Akiba et al. 2019] Optuna: TPE sampler for efficient hyperparameter search
2. [OGB Leaderboard] GIN-VN config: 5 layers, 300 dim, 0.5 dropout, 100 epochs
3. [Prokhorenkova et al. 2018] CatBoost: depth 6-10, lr 0.01-0.3 for tabular
4. [Bemis & Murcko 1996] Scaffold-based analysis for molecular datasets
"""))

# Split script into logical sections
sections = script.split('# ══════')

# First section: imports + data loading
cells.append(nbf.v4.new_code_cell(sections[0].strip()))

# Data loading
cells.append(nbf.v4.new_markdown_cell("## Data Loading"))
cells.append(nbf.v4.new_code_cell(('# ══════' + sections[1]).strip()))

# GIN model definition
cells.append(nbf.v4.new_markdown_cell("## GIN+Edge Model (Tunable Architecture)"))
cells.append(nbf.v4.new_code_cell(('# ══════' + sections[2]).strip()))

# GIN Optuna tuning
cells.append(nbf.v4.new_markdown_cell("""## Experiment 4.1: Optuna Tuning of GIN+Edge

Searching over: hidden_dim ∈ {64, 128, 256}, num_layers ∈ [2,5], dropout ∈ [0.2,0.7], lr ∈ [1e-4, 1e-2], batch_size ∈ {128, 256, 512}, pool_type ∈ {add, mean}.

20 trials with TPE sampler. Each trial trains for up to 40 epochs with patience=10.
"""))
cells.append(nbf.v4.new_code_cell(('# ══════' + sections[3]).strip()))

# Feature computation
cells.append(nbf.v4.new_markdown_cell("## Feature Computation for CatBoost"))
cells.append(nbf.v4.new_code_cell(('# ══════' + sections[4]).strip()))

# CatBoost tuning
cells.append(nbf.v4.new_markdown_cell("""## Experiment 4.2: Optuna Tuning of CatBoost MI-top-400

Searching over: depth ∈ [4,10], lr ∈ [0.01, 0.3], l2_leaf_reg ∈ [1, 30], iterations ∈ [300, 1500], border_count ∈ {32, 64, 128, 254}, random_strength ∈ [0, 5], bagging_temperature ∈ [0, 5].

30 trials. Mark's Phase 3 used default CatBoost params — tuning could unlock significant gains.
"""))
cells.append(nbf.v4.new_code_cell(('# ══════' + sections[5]).strip()))

# Error analysis
cells.append(nbf.v4.new_markdown_cell("""## Experiment 4.3: Error Analysis

### Scaffold Analysis
The OGB scaffold split puts molecules with the same Murcko scaffold in the same split. This means test scaffolds are NEVER seen in training — the model must generalize to new chemical structures.

Key questions:
- Do rare scaffolds (singletons) have worse accuracy?
- Are errors concentrated in specific molecular property ranges?
- What do the hardest examples look like?
"""))
cells.append(nbf.v4.new_code_cell(('# ══════' + sections[6]).strip()))

# Learning curves
cells.append(nbf.v4.new_markdown_cell("""## Experiment 4.4: Learning Curves

Is the model data-limited or model-limited? If AUC is still climbing at 100% data, more data would help. If it's plateaued, we need better features or architectures.
"""))
cells.append(nbf.v4.new_code_cell(('# ══════' + sections[7]).strip()))

# Plots
cells.append(nbf.v4.new_markdown_cell("## Plots"))
cells.append(nbf.v4.new_code_cell(('# ══════' + sections[8]).strip()))

# Master comparison
cells.append(nbf.v4.new_markdown_cell("""## Master Comparison Table

All models across all phases, ranked by test ROC-AUC.
"""))
cells.append(nbf.v4.new_code_cell(('# ══════' + sections[9]).strip()))

# Save results
cells.append(nbf.v4.new_markdown_cell("## Save Results"))
cells.append(nbf.v4.new_code_cell(('# ══════' + sections[10]).strip()))

# Key findings
cells.append(nbf.v4.new_markdown_cell("""## Key Findings

*(To be filled after execution — see printed output above)*

1. **GIN+Edge tuning result:** Did tuning beat the Phase 3 default (0.7860)?
2. **CatBoost tuning result:** Did tuning beat Mark's Phase 3 default (0.8105)?
3. **Error analysis insight:** Where do models fail and why?
4. **Learning curve verdict:** Is more data or better features the path forward?
"""))

nb['cells'] = cells
out_path = BASE / 'notebooks' / 'phase4_hyperparameter_tuning.ipynb'
nbf.write(nb, str(out_path))
print(f'Wrote {out_path}')
