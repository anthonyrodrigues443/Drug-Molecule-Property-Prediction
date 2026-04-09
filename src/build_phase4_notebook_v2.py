"""Build phase4_hyperparameter_tuning.ipynb from the results script."""
import nbformat as nbf
from pathlib import Path

BASE = Path('/Users/anthonyrodrigues/Desktop/YC-Portfolio-Projects/Drug-Molecule-Property-Prediction')
script = (BASE / 'src' / 'phase4_build_results.py').read_text()

nb = nbf.v4.new_notebook()
cells = []

cells.append(nbf.v4.new_markdown_cell("""# Phase 4: Hyperparameter Tuning + Error Analysis — Drug Molecule Property Prediction

**Date:** 2026-04-09 · **Researcher:** Anthony Rodrigues · **Dataset:** ogbg-molhiv (scaffold split)

## Building on Phase 3
- **Phase 3 Anthony champion:** GIN+Edge = 0.7860 AUC (first GNN to beat CatBoost)
- **Phase 3 Mark champion:** CatBoost MI-top-400 = 0.8105 AUC (overall leader)
- **SOTA target:** 0.8476 (GIN-VN from OGB leaderboard)

## Today's Questions
1. Can Optuna tuning close the gap? GIN+Edge default config (128d, 3 layers) vs tuned
2. Can CatBoost MI-400 be pushed further with tuned hyperparameters?
3. Where does the model fail? Scaffold-based and property-based error analysis
4. Is more data the answer? Learning curves

## Research References
1. [Akiba et al. 2019] Optuna: TPE sampler for efficient hyperparameter search
2. [OGB Leaderboard] GIN-VN: 5 layers, 300 dim, 0.5 dropout, 100 epochs
3. [Prokhorenkova et al. 2018] CatBoost: depth 6-10, lr 0.01-0.3 for tabular
4. [Bemis & Murcko 1996] Scaffold-based analysis for molecular datasets
"""))

# Split into sections
sections = script.split('# ═══════')
# Filter out empty sections
sections = [s for s in sections if s.strip()]

# Imports + header
cells.append(nbf.v4.new_code_cell(sections[0].strip()))

# GIN Results
cells.append(nbf.v4.new_markdown_cell("""## Experiment 4.1: GIN+Edge Optuna Tuning (8 trials)

Searched: hidden_dim ∈ {64, 128, 256}, num_layers ∈ [2,5], dropout ∈ [0.2,0.7], lr ∈ [1e-4, 1e-2], batch_size ∈ {256, 512}, pool_type ∈ {add, mean}. Each trial trained for up to 25 epochs with patience=8.
"""))
cells.append(nbf.v4.new_code_cell(('# ═══════' + sections[1]).strip()))

# CatBoost Results
cells.append(nbf.v4.new_markdown_cell("""## Experiment 4.2: CatBoost MI-400 Optuna Tuning (20 trials)

Searched: depth ∈ [4,10], lr ∈ [0.01, 0.3], l2_leaf_reg ∈ [1, 30], iterations ∈ [300, 1500]. Mark's Phase 3 used CatBoost defaults — does tuning help?
"""))
cells.append(nbf.v4.new_code_cell(('# ═══════' + sections[2]).strip()))

# Error Analysis with retrained CB
cells.append(nbf.v4.new_markdown_cell("""## Error Analysis: Feature Computation & CatBoost Retraining
"""))
cells.append(nbf.v4.new_code_cell(('# ═══════' + sections[3]).strip()))

# Scaffold Analysis
cells.append(nbf.v4.new_markdown_cell("""## Scaffold-Based Error Analysis

OGB scaffold split ensures test scaffolds are NEVER seen in training. How does performance vary by scaffold frequency?
"""))
cells.append(nbf.v4.new_code_cell(('# ═══════' + sections[4]).strip()))

# Error by molecular properties
cells.append(nbf.v4.new_markdown_cell("""## Error Analysis by Molecular Properties

Which molecular property ranges cause the most errors?
"""))
cells.append(nbf.v4.new_code_cell(('# ═══════' + sections[5]).strip()))

# Threshold optimization
cells.append(nbf.v4.new_markdown_cell("""## Threshold Optimization

With 3.5% positive rate, the default 0.5 threshold is suboptimal. What threshold maximizes F1?
"""))
cells.append(nbf.v4.new_code_cell(('# ═══════' + sections[6]).strip()))

# Hardest examples
cells.append(nbf.v4.new_markdown_cell("""## Hardest Examples: Most Confident Errors
"""))
cells.append(nbf.v4.new_code_cell(('# ═══════' + sections[7]).strip()))

# Learning curves
cells.append(nbf.v4.new_markdown_cell("""## Learning Curves: Is More Data the Answer?
"""))
cells.append(nbf.v4.new_code_cell(('# ═══════' + sections[8]).strip()))

# Plots
cells.append(nbf.v4.new_markdown_cell("## Plots"))
cells.append(nbf.v4.new_code_cell(('# ═══════' + sections[9]).strip()))

# Master comparison
cells.append(nbf.v4.new_markdown_cell("## Master Comparison Table"))
cells.append(nbf.v4.new_code_cell(('# ═══════' + sections[10]).strip()))

# Save
cells.append(nbf.v4.new_markdown_cell("## Save Results"))
cells.append(nbf.v4.new_code_cell(('# ═══════' + sections[11]).strip()))

# Key findings
cells.append(nbf.v4.new_markdown_cell("""## Key Findings

1. **Hyperparameter tuning yields MARGINAL gains:** GIN+Edge improved by only +0.004 AUC (0.7860 → 0.7904). CatBoost tuning FAILED to beat Mark's Phase 3 default (0.8105 vs best tuned 0.7909).

2. **COUNTERINTUITIVE: Smaller GNN generalizes better.** 64-dim GIN (0.7904 test) beats 256-dim GIN (0.7631 test) on scaffold split. Bigger model = bigger val-test gap. This confirms the scaffold split punishes memorization.

3. **The bottleneck is NOT hyperparameters — it's feature selection quality.** Mark's MI-top-400 got 0.8105 with DEFAULT CatBoost. My Optuna-tuned CatBoost on a fresh MI-top-400 gets only 0.7909. The MI selection has stochastic variance that dominates model tuning.

4. **Error patterns reveal structural bias:** Models struggle most with large molecules (MW>500: 81% acc vs 98% for MW<200), high lipophilicity (logP>5: 89% vs 96%), and multi-ring systems (>5 rings: 88% vs 96%). These are exactly the molecules that drug discovery cares about most.

5. **The learning curve is non-monotonic** — performance doesn't improve linearly with more data. This suggests the model is learning scaffold-specific patterns that don't transfer across the split.

## Implications for Phase 5
- Don't chase hyperparameters — the gain ceiling is ~0.005 AUC
- Focus on feature selection stability (ensemble of MI runs, or use permutation importance)
- Try loss function modifications (focal loss for hard examples, cost-sensitive for high-MW molecules)
- Explore ensemble of GIN+Edge + CatBoost — they make different errors
"""))

nb['cells'] = cells
out_path = BASE / 'notebooks' / 'phase4_hyperparameter_tuning.ipynb'
nbf.write(nb, str(out_path))
print(f'Wrote {out_path}')
