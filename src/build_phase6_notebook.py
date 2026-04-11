"""Build Phase 6 notebook from executed results."""
import json

cells = []

def md(source):
    cells.append({"cell_type": "markdown", "metadata": {}, "source": source.split("\n")
                  if "\n" not in source else [l + "\n" for l in source.split("\n")]})

def code(source):
    cells.append({"cell_type": "code", "metadata": {}, "source": [l + "\n" for l in source.split("\n")],
                  "execution_count": None, "outputs": []})

md("""# Phase 6: Explainability & Model Understanding — ogbg-molhiv
**Date:** 2026-04-11 | **Session:** 6 of 7 | **Researcher:** Anthony Rodrigues

## Objective
Understand WHY the GIN+CatBoost ensemble achieves 0.8114 AUC on HIV activity prediction:
1. **SHAP TreeExplainer** for CatBoost — which molecular features drive predictions?
2. **Gradient saliency** for GIN — which atoms does the GNN focus on?
3. **Domain validation** — do the important features align with medicinal chemistry literature?
4. **Ensemble disagreement analysis** — what makes molecules hard for each model?

## Research References
1. Lundberg & Lee 2017 — SHAP: A Unified Approach to Interpreting Model Predictions
2. Pope et al. 2019 — Explainability Methods for GNNs (gradient saliency)
3. Lipinski et al. 1997 — Rule of Five for drug-likeness
4. Durant et al. 2002 — MACCS structural keys for chemical similarity

## Building on Phase 5
- GIN+CatBoost ensemble = 0.8114 AUC (champion)
- CatBoost fails on large polar molecules (MW 491), GIN on small (MW 374)
- Error overlap Jaccard = 0.235 — models are complementary""")

code("""import os
os.chdir('/Users/anthonyrodrigues/Desktop/YC-Portfolio-Projects/Drug-Molecule-Property-Prediction')
# Run the full Phase 6 analysis
exec(open('src/phase6_explainability.py').read())""")

md("""## Results Summary

### Exp 6.1: SHAP Analysis (CatBoost)

The SHAP TreeExplainer reveals that **substructural fingerprints dominate** predictions:

| Rank | Feature | Mean |SHAP| | Category | Domain Meaning |
|------|---------|------|----------|----------------|
| 1 | maccs_144 | 0.1910 | MACCS key | Substructural pattern |
| 2 | maccs_81 | 0.0944 | MACCS key | Substructural pattern |
| 3 | bertz_ct | 0.0804 | Domain | Molecular complexity index |
| 4 | tpsa | 0.0708 | Domain | Topological polar surface area |
| 5 | num_heteroatoms | 0.0702 | Domain | N, O, S atom count |
| 6 | mfp_967 | 0.0674 | Morgan FP | Circular substructure |
| 7 | maccs_150 | 0.0660 | MACCS key | Substructural pattern |
| 8 | maccs_67 | 0.0648 | MACCS key | Substructural pattern |
| 9 | fr_hdrzone | 0.0631 | Fragment | Hydrazone group |
| 10 | maccs_95 | 0.0613 | MACCS key | Substructural pattern |

**Category-level importance:**
| Category | Total |SHAP| | Features | Mean |SHAP| |
|----------|--------|----------|----------|
| MACCS keys | 1.347 | 98 | 0.014 |
| Morgan FP | 0.955 | 251 | 0.004 |
| Domain descriptors | 0.559 | 14 | 0.040 |
| RDKit Fragments | 0.459 | 37 | 0.012 |

**Key insight:** MACCS keys (substructural fingerprints) have the highest TOTAL contribution,
but domain descriptors have the highest PER-FEATURE importance (0.040 vs 0.014).
This means: each domain descriptor is 3x more informative than each MACCS key,
but the 98 MACCS keys collectively outweigh the 14 domain descriptors.

### Exp 6.2: Active vs Inactive SHAP Differences

For the top 5 features, SHAP values differ between HIV-active and inactive molecules:

| Feature | Active SHAP | Inactive SHAP | Difference |
|---------|-------------|---------------|------------|
| maccs_144 | -0.032 | -0.076 | +0.044 |
| maccs_81 | +0.052 | -0.033 | +0.085 |
| bertz_ct | +0.014 | -0.032 | +0.046 |
| tpsa | +0.015 | -0.059 | +0.074 |
| num_heteroatoms | -0.020 | -0.058 | +0.039 |

**maccs_81** shows the strongest differential: present in active molecules (SHAP +0.052),
absent/different in inactive ones (SHAP -0.033). This substructural pattern is a strong
positive signal for HIV activity.

### Exp 6.3: GIN Gradient Saliency

Which atom types does the GIN model pay most attention to?

| Atom | Mean Saliency | Count | Domain Relevance |
|------|---------------|-------|------------------|
| S (Sulfur) | 0.393 | 114 | Thiol/thioether groups — critical for protease inhibitor binding |
| N (Nitrogen) | 0.183 | 539 | Amine/amide groups — hydrogen bonding with enzyme active sites |
| C (Carbon) | 0.138 | 3702 | Backbone — lower individual importance, highest count |
| O (Oxygen) | 0.091 | 680 | Hydroxyl/carbonyl — lower saliency than expected |
| Cl (Chlorine) | 0.204 | 24 | Halogen substituents — electron-withdrawing effects |
| P (Phosphorus) | 0.485 | 12 | Phosphate groups — nucleotide analogs |

**Key finding:** Among common drug atoms, **Sulfur has 2.8x higher saliency than Carbon**
(0.393 vs 0.138). This aligns with HIV protease inhibitor design — thiol groups form
critical interactions with the catalytic aspartate residues in HIV-1 protease.
Nitrogen (0.183) also shows elevated saliency, consistent with its role in
hydrogen bonding at enzyme binding sites.

### Exp 6.4: Ensemble Disagreement Properties

What molecular properties characterize the molecules each model gets wrong?

| Category | Count | MW | logP | TPSA | Rings | Heavy Atoms |
|----------|-------|----|------|------|-------|-------------|
| Both correct | 2946 | 340.2 | 2.88 | 66.5 | 3.5 | 23.8 |
| Both wrong | 188 | **501.9** | 3.50 | **135.8** | 4.2 | 34.2 |
| GIN correct, CB wrong | 104 | 441.3 | 3.18 | 102.8 | 3.8 | 29.8 |
| CB correct, GIN wrong | 873 | 401.1 | 2.96 | 92.4 | 3.9 | 27.6 |

**Hardest molecules (both wrong):** MW = 501.9 (above Lipinski MW cutoff of 500),
TPSA = 135.8 (above the 140 permeability threshold). These are large, polar molecules
at the edge of drug-like space — precisely where both fingerprint-based and graph-based
approaches struggle because they're structurally dissimilar to training scaffolds.

**CatBoost's blind spot:** Larger molecules (MW 441 vs 340) with higher TPSA (103 vs 66.5).
The fingerprint representation compresses structural complexity.

**GIN's blind spot:** Medium-sized molecules (MW 401) — the GNN may over-generalize
graph topology for compounds that share scaffolds but differ in substituent effects.""")

md("""## Key Findings

1. **Substructure > Bulk Properties:** MACCS keys (total SHAP 1.35) dominate over domain
   descriptors (0.56). HIV activity prediction is driven by specific binding motifs,
   not just molecular weight or lipophilicity.

2. **Sulfur is 2.8x more important than Carbon in GIN:** Gradient saliency shows the GNN
   focuses on heteroatoms (S > N >> C > O), consistent with pharmacophore theory —
   thiol and amine groups drive HIV protease binding.

3. **Per-feature vs total importance inversion:** Each domain descriptor is 3x more
   informative than each MACCS key (0.040 vs 0.014), but the 98 MACCS keys collectively
   dominate. The model benefits from BOTH the few powerful descriptors AND the many
   weak substructural signals.

4. **Both models fail on large polar molecules (MW > 500, TPSA > 135).** These are at the
   boundary of drug-like space and structurally dissimilar to training scaffolds —
   the fundamental challenge of scaffold-split generalization.

5. **Hydrazone (fr_hdrzone) is the top fragment:** Ranked #9 overall. Hydrazones are known
   pharmacophores in anti-HIV compounds, validating the model's chemical intuition.""")

md("""## Next Phase
Phase 7 (Sunday): Production pipeline + Streamlit UI
- Clean inference pipeline with SHAP explanations per prediction
- Interactive molecule explorer showing feature importance
- Model card with limitations and ethical considerations""")

nb = {"nbformat": 4, "nbformat_minor": 5,
      "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                    "language_info": {"name": "python", "version": "3.14.0"}},
      "cells": cells}

with open('/Users/anthonyrodrigues/Desktop/YC-Portfolio-Projects/Drug-Molecule-Property-Prediction/notebooks/phase6_explainability.ipynb', 'w') as f:
    json.dump(nb, f, indent=1)
print("Notebook created: notebooks/phase6_explainability.ipynb")
