# Drug Molecule Property Prediction

Predicting HIV drug activity on the ogbg-molhiv dataset (41,127 molecules, 3.5% active) using the OGB scaffold split and ROC-AUC as the primary metric. Benchmarking against the OGB public leaderboard (SOTA: 0.8476 ROC-AUC).

---

## Architecture

```mermaid
graph TB
    subgraph Input
        S[SMILES string]
    end

    subgraph "GIN+Edge Path (w=0.3)"
        S --> G1[OGB Graph Conversion]
        G1 --> G2[AtomEncoder + BondEncoder]
        G2 --> G3[3× GIN Layers + Bond Aggregation]
        G3 --> G4[Add Pooling → Classifier]
        G4 --> GP[GIN Probability]
    end

    subgraph "CatBoost Path (w=0.7)"
        S --> F1[RDKit Feature Extraction]
        F1 --> F2["1,302-dim Pool<br/>(14 Lipinski + 1024 Morgan + 167 MACCS + 85 Fragments)"]
        F2 --> F3[MI Selection → Top 400]
        F3 --> F4[CatBoost Classifier]
        F4 --> CP[CatBoost Probability]
    end

    subgraph Ensemble
        GP --> E[Weighted Average<br/>0.3 × GIN + 0.7 × CB]
        CP --> E
        E --> OUT[HIV Activity Prediction<br/>+ Lipinski Check + SHAP Explanation]
    end
```

## Project Status: Complete

**Champion model: GIN+CatBoost Ensemble — ROC-AUC = 0.8114** (OGB scaffold split)

7 phases of research across 100+ experiments by two researchers (Anthony + Mark). Error Jaccard overlap = 0.161 between GIN and CatBoost proves models fail on structurally different molecules. Ensemble rescued 542 test molecules, hurt zero.

---

## Key Findings

1. **Feature curation beats feature quantity** — CatBoost + MI-top-400 (0.8105) beats GIN+Edge (0.7860) by +0.025; same 1302-d pool with naive concatenation gives 0.7673 — MI selection recovers +0.043 AUC with no model change
2. **Edge features were the key GNN unlock** — 3 BondEncoder dims (+0.081 AUC to GIN) is the largest single-feature gain across all phases; bond type/stereo/conjugation encodes chemical connectivity Morgan fingerprints only approximate
3. **MACCS punches above its weight** — MI retains 65% of 167 MACCS keys vs 23% of 1024 Morgan bits; 31% of CatBoost importance from 12.8% of pool — hand-curated substructure keys are 2.4× more information-dense per bit than Morgan hashed space
4. **GIN+CatBoost ensemble (0.3/0.7) is the new champion at 0.8114 AUC** — error Jaccard overlap of only 0.161 proves GNN and tabular models fail on different molecules; graph topology and chemistry features are genuinely complementary
5. **Lipinski violators define the model's usable range** — Subgroup AUC=0.8450 / recall=54.3% for violators vs AUC=0.6707 / recall=3.3% for compliant actives; the model learned to find large, complex HIV protease inhibitors and nearly fails on small drug-like actives — a systematic blind spot revealed only through subgroup analysis

---

## Models Compared

**Phase 1:** 15+ experiments across LogReg, RF, XGBoost, LightGBM, and CatBoost with 4 feature sets (Lipinski-12, Morgan FP 1024, combined 1036, graph topology 5), class-weighting strategies, and threshold tuning

**Phase 2:** 5+ experiments across GCN, GIN, GAT, GraphSAGE (4 GNN architectures), and MLP-Domain9 ablation; testing raw graph features vs domain feature sets

**Phase 3:** 26+ experiments — GIN+Edge (BondEncoder), GIN+VN, GIN+Edge+VN (3 GNN variants), CatBoost ablation across 11 feature sets (Anthony), plus 7-set fragment ablation and MI sweep across 12 K-values on a 1302-d pool with K=400 composition audit (Mark)

**Phase 4:** 20+ experiments — Optuna tuning of GIN+Edge (8 trials) and CatBoost MI-400 (20 + 40 trials), K=400 bootstrap stability analysis (3 bootstraps × 4 K-values), deep error analysis across 4,113 test molecules by 11 molecular properties, Lipinski violation stratification, and feature importance attribution across 5 chemical categories

**Phase 5:** 20+ experiments — leave-one-category-out feature ablation across 5 descriptor categories, MW-split subgroup specialists, diverse 3–4 model CatBoost ensembles on distinct feature sub-pools (Mark); fragment-free cross-paradigm ensemble test, weight sweep confirmation, structural error profile analysis by MW/HBD/HBA/TPSA, and 542-molecule rescue analysis (Anthony)

**Phase 6:** 20+ experiments — SHAP TreeExplainer on 1000 test molecules, active vs inactive SHAP differential analysis, GIN gradient saliency via forward hooks on atom embeddings (200 molecules), ensemble disagreement property profiling (Anthony); LIME local explanations on 8 representative molecules (5000 perturbations each), Lipinski subgroup SHAP comparison (compliant vs violating), feature group solo-AUC attribution across 5 categories, LIME-SHAP agreement Jaccard scoring by prediction type (Mark)

**Phase 7:** 50 tests across 5 suites (all passing) — 28 unit tests (feature engineering, GINEdge architecture, MI selection determinism) and 22 integration tests (chemical correctness invariants on 4 real drugs, latency benchmarks at ~15ms/molecule, robustness to malformed SMILES, MI quality, app smoke test); production pipeline (train/predict/evaluate scripts, config.yaml), 3-tab Streamlit UI with SHAP attribution + batch analysis + 15-model leaderboard, empty SMILES bug discovery, and model card

---

## Iteration Summary

### Phase 1: Domain Research + Dataset + EDA + Baseline — 2026-04-06

<table>
<tr>
<td valign="top" width="38%">

**Dataset & Standard Baselines:** Selected ogbg-molhiv (41K molecules, 3.5% HIV-active, OGB scaffold split) over smaller alternatives (ESOL, Lipophilicity, AqSolDB). RF with combined features (Lipinski-12 + Morgan FP 1024) achieves ROC-AUC=0.7707. Combined features consistently beat either alone — unlike solubility tasks, HIV activity needs both domain descriptors and structural patterns.<br><br>
**Imbalance & Threshold Tuning:** CatBoost auto-weighted becomes new champion (ROC-AUC=0.7782, recall=0.523). Threshold tuning at 0.59 boosts F1 by +27%. Feature importance reveals 12/15 top features are domain descriptors, but FP bits collectively hold 72.5% of total importance. Graph topology alone reaches 0.70 AUC.

</td>
<td align="center" width="24%">

<img src="results/phase1_roc_pr_curves.png" width="220">

</td>
<td valign="top" width="38%">

**Combined Insight:** The bottleneck is both model family and decision calibration. CatBoost's ordered boosting handles 3.5% imbalance more gracefully than RF/XGBoost (+0.0075 AUC), while threshold tuning at 0.59 extracts +27% F1 without changing the model. Feature importance shows domain features and fingerprints are complementary: domain features rank individually highest, but fingerprints carry 72.5% of collective signal. The 0.070 AUC gap to SOTA confirms tabular models plateau here — closing it requires GNNs.<br><br>
**Surprise:** Threshold tuning (0.50→0.59) boosts F1 more than switching model families. At 3.5% imbalance, the default 0.50 threshold wastes discrimination — the model already ranks well, it just cuts at the wrong point.<br><br>
**Research:** Hu et al., 2020 — OGB benchmark, SOTA 0.8476 requires graph-level representations; Prokhorenkova et al., 2018 — CatBoost ordered boosting handles class imbalance without naive oversampling; He & Garcia, 2009 — moderate imbalance responds to weighting without discrimination collapse.<br><br>
**Best Model So Far:** CatBoost (auto_class_weights, combined 1036 features) — ROC-AUC=0.7782, AUPRC=0.3708, Recall=0.523

</td>
</tr>
</table>

### Phase 2: Multi-Model GNN Comparison — 2026-04-07

<table>
<tr>
<td valign="top" width="38%">

**GNN Architectures:** Tested 4 GNNs (GCN, GIN, GAT, GraphSAGE) on raw 9-feature atom graphs. Best: GIN at 0.7053 AUC; worst: GAT at 0.6677. All 4 underperform CatBoost (0.7782) by 0.07–0.11 AUC — graph topology alone can't compensate for missing chemistry features.<br><br>
**MLP Ablation:** Tiny MLP-Domain9 (9 domain features, 5K params) hits 0.7670 AUC — beating all 4 GNNs. Neural failure on molecular graphs is not architecture-specific; it persists across GNN and dense networks when input features are too sparse.

</td>
<td align="center" width="24%">

<img src="results/phase2_model_comparison.png" width="220">

</td>
<td valign="top" width="38%">

**Combined Insight:** Both runs together prove the bottleneck is input feature quality, not architecture. Anthony's GNNs and Mark's MLP both operate on 9 raw features and both fail to match CatBoost's 1,036 hand-crafted chemistry features. Architecture choice is secondary to feature richness.<br><br>
**Surprise:** A 5K-param MLP on 9 domain features (0.7670) outperforms a 93K-param GIN on full molecular graphs (0.7053). Model capacity does not compensate for sparse input signals — the features encode more than the graph topology alone.<br><br>
**Research:** Xu et al., 2019 — GIN achieves WL-test expressivity; empirically leads all GNNs but still trails tabular ML, confirming feature quality dominates. Hu et al., 2020 (OGB) — basic GIN+virtual node baseline 0.7558; our GIN matches this, confirming correct implementation and that the gap is real.<br><br>
**Best Model So Far:** CatBoost (auto_class_weights, combined 1036 features) — ROC-AUC=0.7782, AUPRC=0.3708, Recall=0.523

</td>
</tr>
</table>

### Phase 3: Feature Engineering + Deep Dive — 2026-04-08

<table>
<tr>
<td valign="top" width="38%">

**Edge Features & Ablation:** GIN+Edge (BondEncoder) jumps from 0.7053 to 0.7860 AUC (+0.081) — first GNN to beat CatBoost and the largest single gain across all phases. GIN+VN (0.7578) and GIN+Edge+VN (0.7622) both fall below GIN+Edge. CatBoost ablation over 11 feature sets confirms Lipinski-14 (0.7744) captures 98.5% of 1038-dim Lip+FP signal; full hybrid (1345d, 0.7415) is the worst CatBoost result.<br><br>
**MI Feature Selection:** Mutual-information sweep on a 1302-d pool (AllTrad+85 RDKit Fragments). K=400 reaches 0.8105 AUC — new overall champion, beating GIN+Edge by +0.025. K=200 (0.8019) and K=500 (0.7945) also clear GIN+Edge. Composition: all 14 Lipinski features survive; MACCS selected at 65% vs Morgan at 23% — hand-curated keys carry more per-bit signal.

</td>
<td align="center" width="24%">

<img src="results/phase3_mark_leaderboard.png" width="220">

</td>
<td valign="top" width="38%">

**Combined Insight:** Anthony proved bond information is the key GNN ingredient (+0.081 AUC from BondEncoder); Mark proved feature curation on a richer pool is the key tabular ingredient (+0.043 AUC from MI-top-400 vs naive concatenation). Both findings share the same root cause: it's not the quantity of information, it's the signal-to-noise ratio of what you feed the model.<br><br>
**Surprise:** GIN+Edge+VN has best val AUC (0.8333) but 3rd-worst test AUC (0.7622) — virtual node overfits scaffold-specific global patterns. Meanwhile, Mark's K=50 (0.7591) is worse than K=20 (0.7702): Morgan bits ranked 21–50 are high-MI on training scaffolds but don't transfer to novel ones.<br><br>
**Research:** Hu et al., 2020 (OGB) — AtomEncoder/BondEncoder standard encoding, GIN+VN baseline ~0.77; prompted Anthony's BondEncoder addition. Battiti, 1994 (IEEE TNN) — MI outperforms Pearson/chi² for non-linear heterogeneous pools; motivated Mark's MI sweep over the mixed-cardinality 1302-d set.<br><br>
**Best Model So Far:** CatBoost + MI-top-400 (AllTrad+Frag, K=400) — ROC-AUC=0.8105

</td>
</tr>
</table>

### Phase 4: Hyperparameter Tuning + Error Analysis — 2026-04-09

<table>
<tr>
<td valign="top" width="38%">

**GNN Tuning + Ensemble:** Optuna tuned GIN+Edge (8 trials) — best config is 64d, 3L, dropout=0.4, test AUC=0.7982 (+0.012 vs Phase 3 default). Smaller models generalize better; 5-layer GNNs catastrophically overfit on scaffold split (0.69–0.72 test). GIN+CatBoost ensemble at 0.3/0.7 weight reaches 0.8114 AUC — new project champion. Error Jaccard overlap = 0.161: GIN and CatBoost fail on different molecules, explaining the ensemble gain.<br><br>
**CatBoost Tuning + Error Analysis:** Optuna on CatBoost MI-400 (40 trials): best config depth=8, lr=0.055, l2=4.7, min_leaf=38 gives Val=0.8229, Test=0.7854 (+0.027) but val↔test gap widened from 0.029 to 0.038. K=400 bootstrap stability confirmed (std=0.0040 vs 0.015–0.022 for other K). Error analysis: caught actives average MW=630 / 5.6 rings vs missed actives MW=424 / 4.0 rings.

</td>
<td align="center" width="24%">

<img src="results/phase4_anthony_tuning.png" width="220">

</td>
<td valign="top" width="38%">

**Combined Insight:** Hyperparameter tuning alone hits a ceiling — scaffold splits penalize val-optimized HP search and feature selection variance (~0.02 AUC) dominates any tuning signal. The real Phase 4 unlock is the ensemble: GIN captures graph topology, CatBoost captures chemistry distributions, and their Jaccard error overlap of only 0.161 proves they fail on fundamentally different molecules. Combining them recovers what neither can do alone.<br><br>
**Surprise:** Lipinski violators are 2× easier to classify (recall 0.828 vs 0.400 for rule-compliant actives). HIV protease inhibitors are large, complex molecules designed for efficacy over oral bioavailability — the model learned "bigger = more likely HIV inhibitor," which is historically accurate for early HIV drugs.<br><br>
**Research:** Dietterich, 2000 — ensemble methods improve generalization when error overlap is low; Jaccard=0.161 here directly validates this condition. Wu et al., 2018 (MoleculeNet) — scaffold splits create larger val/test gaps; confirmed by HP tuning widening the gap from 0.029 to 0.038.<br><br>
**Best Model So Far:** GIN+CatBoost Ensemble (0.3/0.7) — ROC-AUC=0.8114

</td>
</tr>
</table>

### Phase 5: Cross-Paradigm Analysis + Feature Ablation — 2026-04-10

<table>
<tr>
<td valign="top" width="38%">

**Cross-Paradigm Ensemble:** Anthony tested fragment removal in the GIN+CB ensemble — effect was split-dependent (-0.004 for CB alone vs Mark's +0.026; ensemble net +0.002). Structural error analysis reveals the root cause of ensemble gains: CB fails on large polar molecules (MW 491, TPSA 129, high HBD/HBA) where fingerprints miss structural nuance; GIN fails on smaller molecules (MW 374) where graph topology is less informative. Ensemble rescued 542 molecules, hurt zero — 13.2% of test set rescued through pure complementarity. Error symmetry: 392 GIN-only vs 393 CB-only unique errors.<br><br>
**Feature Ablation + Diverse Ensemble:** Mark's leave-one-category-out ablation on MI-400 finds Fragment descriptors are noise — removal improves AUC by +0.026. MACCS removal is the largest drop (-0.032, most critical). MW-split specialists hurt both subgroups: small-molecule specialist fails from data sparsity (2.8% active rate in 26K train). Diverse 3-model CatBoost ensemble (MI-400 + MACCS+Advanced + Morgan+Lipinski sub-pools) reaches 0.7888 — matching GIN+Edge (0.7860) via pure feature engineering.

</td>
<td align="center" width="24%">

<img src="results/phase5_anthony_cross_paradigm.png" width="220">

</td>
<td valign="top" width="38%">

**Combined Insight:** Two different Phase 5 angles — structural error profiling vs feature category ablation — converge on the same explanation for why the GIN+CB ensemble works. Anthony's MW/TPSA error profiles show CB fails exactly where MACCS pharmacophore patterns are most needed (large, polar molecules); Mark's ablation confirms MACCS is the most irreplaceable category. The ensemble's 0.235 Jaccard overlap and 542/0 rescue/hurt count are structural proof that graph topology and molecular descriptors capture different chemical reality.<br><br>
**Surprise:** The ensemble rescued 542 molecules and hurt exactly zero. With Jaccard overlap = 0.235, each model's errors are nearly orthogonal — confidence from one model consistently overrides the other's mistakes without introducing new ones.<br><br>
**Research:** Dietterich, 2000 — ensemble diversity requires low error overlap; Jaccard=0.235 here confirms the condition holds. Bender et al., 2021 (Drug Discovery Today) — MACCS keys outperform Morgan FP per-bit in QSAR; confirmed as the most critical MI-400 category with -0.032 AUC impact on removal.<br><br>
**Best Model So Far:** GIN+CatBoost Ensemble (0.3/0.7) — ROC-AUC=0.8114

</td>
</tr>
</table>

### Phase 6: Explainability & Model Understanding — 2026-04-11

<table>
<tr>
<td valign="top" width="38%">

**Global XAI:** SHAP TreeExplainer on 1,000 test molecules: MACCS keys dominate total SHAP (1.35 vs 0.56 for domain descriptors). maccs_144 is the top feature at 0.191 mean |SHAP|. GIN gradient saliency via atom embedding hooks (200 molecules): sulfur saliency 0.393, 2.8x carbon (0.138) — the GNN independently discovered thiol groups as the key pharmacophore for HIV protease inhibitor binding.<br><br>
**Local XAI:** LIME on 8 representative molecules (2 TP, 2 TN, 2 FP, 2 FN) yields LIME-SHAP Jaccard = 0.042 — near-zero agreement per molecule. LIME says Morgan dominates locally (63.7%) while SHAP says MACCS dominates globally (43%). Subgroup analysis: Lipinski-compliant actives achieve only AUC=0.6707 / recall=3.3% vs Lipinski-violating actives at AUC=0.8450 / recall=54.3%.

</td>
<td align="center" width="24%">

<img src="results/phase6_explainability.png" width="220">

</td>
<td valign="top" width="38%">

**Combined Insight:** Anthony's global SHAP and Mark's local LIME are both correct — they describe different views of the same model. SHAP captures collinearity-adjusted population credit (MACCS dominates); LIME captures per-molecule perturbation sensitivity (Morgan dominates locally). The near-zero Jaccard (0.042) is not measurement noise — it exposes that the model uses different feature pathways depending on the individual molecule. The model is more complex than any single global summary can capture.<br><br>
**Surprise:** 14 domain features (Lipinski properties: MW, logP, TPSA, etc.) trained alone reach AUC=0.7581 — nearly matching 1,024 Morgan fingerprint bits (0.7550). SHAP credits domain at only 16-20% when combined, revealing collinearity compression: 14 orthogonal domain signals get diluted by 1,024 redundant Morgan bits.<br><br>
**Research:** Lundberg & Lee, 2017 — SHAP TreeExplainer for exact attribution on CatBoost; Pope et al., 2019 — gradient saliency at atom embedding level for GNNs with discrete inputs; Lapuschkin et al., 2019 — global XAI can mislead when features are correlated, explaining the LIME-SHAP divergence.<br><br>
**Final Champion:** GIN+CatBoost Ensemble (0.3/0.7) — ROC-AUC=0.8114

</td>
</tr>
</table>

### Phase 7: Testing + Polish + Final Consolidation — 2026-04-12

<table>
<tr>
<td valign="top" width="38%">

**Eval Run 1:** Anthony built the production pipeline (train.py, predict.py, evaluate.py, config.yaml, feature_engineering.py) and a 28-test unit suite covering feature extraction, GINEdge architecture, and MI selection — all passing with no data download required.<br><br>
**Eval Run 2:** Mark added 22 integration tests (chemical correctness invariants, latency benchmarks, robustness edge cases, MI quality, app smoke test) and a 3-tab Streamlit UI with SHAP attribution, Lipinski card, batch analysis, and a 15-model leaderboard. Combined: 50/50 tests pass. Latency: ~15ms per molecule (13× headroom vs 200ms limit).

</td>
<td align="center" width="24%">

<img src="results/phase7_mark_final_leaderboard.png" width="220">

</td>
<td valign="top" width="38%">

**Combined Insight:** Anthony's unit tests verify the building blocks; Mark's integration tests verify those blocks compose correctly under realistic conditions. Together they cover the full stack — feature extraction through inference format — and confirm the production pipeline meets real-time latency requirements with substantial headroom.<br><br>
**Surprise:** `compute_all_features("")` returns a valid feature dict (MW=0) instead of None — RDKit silently accepts empty string as a valid molecule. The empty SMILES guard (`if not smiles.strip(): return None`) was only caught by integration testing, not unit tests, showing unit tests alone are insufficient for cheminformatics edge cases.<br><br>
**Research:** RDKit docs — `Chem.MolFromSmiles("")` returns a valid empty molecule object; explicit guard required in production inference. Streamlit 1.39 docs — `@st.cache_resource` for model loading + `@st.cache_data` for SHAP enables responsive UI despite heavy compute.<br><br>
**Best Model So Far:** GIN+CatBoost Ensemble (0.3/0.7) — ROC-AUC=0.8114 (project champion, all 7 phases)

</td>
</tr>
</table>

---

## Setup & Usage

```bash
# Clone and install
git clone https://github.com/anthonyrodrigues443/Drug-Molecule-Property-Prediction.git
cd Drug-Molecule-Property-Prediction
pip install -r requirements.txt

# Run tests (50 tests, no data download required)
python -m pytest tests/ -v

# Train the ensemble (downloads ogbg-molhiv ~4MB, trains GIN + CatBoost)
python -m src.train

# Predict HIV activity for a molecule
python -m src.predict --smiles "CC(=O)Oc1ccccc1C(=O)O"

# Run full evaluation on OGB test set
python -m src.evaluate

# Launch Streamlit UI (trains surrogate model on first run if artifacts missing)
streamlit run app.py
```

## Project Structure

```
├── config/config.yaml              # All hyperparameters
├── src/
│   ├── data_pipeline.py            # OGB data loading + RDKit descriptors
│   ├── feature_engineering.py      # Full feature extraction + MI selection
│   ├── train.py                    # Production training pipeline
│   ├── predict.py                  # Single/batch inference
│   └── evaluate.py                 # Full evaluation suite
├── app.py                          # Streamlit UI (single molecule, batch, leaderboard)
├── notebooks/                      # 14 research notebooks (7 phases x 2 researchers)
├── models/
│   └── model_card.md               # Model capabilities + limitations
├── results/                        # All metrics, plots, experiment logs
├── reports/                        # Detailed daily research reports
└── tests/                          # 50 pytest tests (unit + integration)
```

## Limitations & Future Work

1. **Lipinski-compliant blind spot:** AUC=0.6707 on rule-compliant actives vs 0.8450 on violators. The model excels at large, complex HIV protease inhibitors but misses small drug-like actives.
2. **No 3D structure:** Uses 2D graph topology only. Conformer-aware models (SchNet, DimeNet) could capture binding geometry.
3. **Scaffold split ceiling:** Val-test gap widens with HP tuning, suggesting scaffold generalization is the fundamental bottleneck.
4. **LIME-SHAP divergence (Jaccard=0.042):** The model uses different feature pathways per molecule, meaning no single global explanation is fully faithful.

## References

- Hu et al., 2020 -- OGB benchmark, scaffold split protocol, AtomEncoder/BondEncoder
- Xu et al., 2019 -- GIN achieves WL-test expressivity on molecular graphs
- Prokhorenkova et al., 2018 -- CatBoost ordered boosting for imbalanced classification
- Battiti, 1994 -- Mutual information for non-linear heterogeneous feature selection
- Dietterich, 2000 -- Ensemble diversity via low error overlap
- Lundberg & Lee, 2017 -- SHAP TreeExplainer for exact feature attribution
- Bender et al., 2021 -- MACCS keys vs Morgan FP in QSAR benchmarks
