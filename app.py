"""
Streamlit UI for Drug Molecule HIV Activity Prediction.

Features:
- Input SMILES → HIV activity prediction (CatBoost only, no GIN required)
- Lipinski Rule-of-5 compliance check with violation highlighting
- SHAP-based feature importance (top contributors per molecule)
- Feature category breakdown (Domain / MACCS / Morgan / Fragments)
- Subgroup context: compliant vs violating recall disclaimer
- Example molecules for quick testing
- Batch analysis via text area

Run: streamlit run app.py
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import streamlit as st
import shap

from rdkit import Chem
from rdkit.Chem import Draw
from sklearn.feature_selection import mutual_info_classif
from catboost import CatBoostClassifier

from src.feature_engineering import compute_all_features, compute_features_batch

# ──────────────────────────────────────────────────────────────
# Page config
# ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="HIV Activity Predictor",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ──────────────────────────────────────────────────────────────
# Constants & example molecules
# ──────────────────────────────────────────────────────────────
EXAMPLE_MOLECULES = {
    "Aspirin (inactive)": "CC(=O)Oc1ccccc1C(=O)O",
    "Amprenavir (HIV protease inhibitor, ACTIVE)": "CC(C)CN(CC(O)C(Cc1ccccc1)NC(=O)OC1CCOC1)S(=O)(=O)c1ccc(N)cc1",
    "Caffeine (inactive)": "Cn1c(=O)c2c(ncn2C)n(C)c1=O",
    "Lopinavir (HIV protease inhibitor, ACTIVE)": "CC1=CC(=CC(=C1)C)C(=O)NC(CC2=CC=CC=C2)CC(CC3=CC=NC=C3)NC(=O)C(CC(C)C)NC4=NC5=CC=CC=C5S4",
    "Indinavir (HIV protease inhibitor, ACTIVE)": "OC(=O)[C@@H]1CCCN1C[C@@H](O)C[C@@H](Cc1ccccc1)NC(=O)[C@@H]1CN2CC[C@H]2CC1",
}

# Feature group boundaries (must match compute_all_features output order)
DOMAIN_FEATS = [
    "mol_weight", "logp", "hbd", "hba", "tpsa", "rotatable_bonds",
    "aromatic_rings", "ring_count", "heavy_atom_count", "fraction_csp3",
    "num_heteroatoms", "lipinski_violations", "bertz_ct", "labute_asa",
]
LIPINSKI_RULES = {
    "mol_weight":        ("Mol. Weight", "≤ 500 Da", lambda v: v <= 500),
    "logp":              ("LogP",        "≤ 5",       lambda v: v <= 5),
    "hbd":               ("H-Bond Donors", "≤ 5",    lambda v: v <= 5),
    "hba":               ("H-Bond Acceptors", "≤ 10", lambda v: v <= 10),
}


# ──────────────────────────────────────────────────────────────
# Cached model loader (trains a lightweight surrogate if no
# saved model is found, so the UI works standalone)
# ──────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading CatBoost model…")
def load_model():
    """
    Try to load the saved CatBoost + MI selector.
    If models/ artifacts don't exist, train a quick surrogate on 5,000 molecules
    from ogbg-molhiv so the UI is always functional.
    """
    import joblib
    from pathlib import Path

    BASE = Path(__file__).parent
    cbm_path   = BASE / "models" / "catboost_mi400.cbm"
    sel_path   = BASE / "models" / "feature_selector.joblib"

    if cbm_path.exists() and sel_path.exists():
        cb = CatBoostClassifier()
        cb.load_model(str(cbm_path))
        selector = joblib.load(sel_path)
        mi_idx  = selector["mi_indices"]
        fcols   = selector["feature_columns"]
        source  = "production"
    else:
        # Build surrogate using processed parquet if available
        parquet = BASE / "data" / "processed" / "ogbg_molhiv_features.parquet"
        if parquet.exists():
            df = pd.read_parquet(parquet)
        else:
            # Last resort: download from OGB
            from ogb.graphproppred import GraphPropPredDataset
            dataset = GraphPropPredDataset(name="ogbg-molhiv", root=str(BASE / "data" / "raw"))
            split_idx = dataset.get_idx_split()
            smiles_df = pd.read_csv(BASE / "data" / "raw" / "ogbg_molhiv" / "mapping" / "mol.csv.gz")
            labels = dataset.labels.flatten()
            train_idx = split_idx["train"].tolist()[:3000]
            rows = []
            for i in train_idx:
                f = compute_all_features(smiles_df["smiles"].iloc[i])
                if f is not None:
                    f["y"] = int(labels[i])
                    rows.append(f)
            df = pd.DataFrame(rows)

        fcols = [c for c in df.columns if c not in {"y", "split", "idx", "smiles", "hiv_active"}]
        X = df[fcols].values.astype(np.float32)
        np.nan_to_num(X, copy=False)
        y = df["y"].values if "y" in df.columns else df["hiv_active"].values

        mi = mutual_info_classif(X, y, random_state=42, n_neighbors=5)
        mi_idx = np.argsort(mi)[-400:]

        cb = CatBoostClassifier(
            iterations=300, depth=7, learning_rate=0.05,
            auto_class_weights="Balanced", eval_metric="AUC",
            random_seed=42, verbose=0,
        )
        cb.fit(X[:, mi_idx], y)
        source = "surrogate"

    return cb, mi_idx, fcols, source


# ──────────────────────────────────────────────────────────────
# Prediction helpers
# ──────────────────────────────────────────────────────────────
def predict_molecule(smiles: str, cb, mi_idx, fcols) -> dict | None:
    feats = compute_all_features(smiles)
    if feats is None:
        return None
    feat_vec = np.array([feats.get(c, 0) for c in fcols], dtype=np.float32)
    np.nan_to_num(feat_vec, copy=False)
    prob = float(cb.predict_proba(feat_vec[mi_idx].reshape(1, -1))[0, 1])
    return {"probability": prob, "feats": feats, "feat_vec": feat_vec}


@st.cache_data(show_spinner=False)
def compute_shap_values(_cb, background_smiles, mi_idx, fcols, query_smiles_list):
    """Compute SHAP for query molecules against a background sample."""
    # Background
    bg_rows = []
    for s in background_smiles:
        f = compute_all_features(s)
        if f is not None:
            bg_rows.append(np.array([f.get(c, 0) for c in fcols], dtype=np.float32)[mi_idx])
    bg = np.array(bg_rows, dtype=np.float32)
    np.nan_to_num(bg, copy=False)

    # Query
    qrows = []
    for s in query_smiles_list:
        f = compute_all_features(s)
        if f is not None:
            qrows.append(np.array([f.get(c, 0) for c in fcols], dtype=np.float32)[mi_idx])
    qX = np.array(qrows, dtype=np.float32)
    np.nan_to_num(qX, copy=False)

    explainer = shap.TreeExplainer(_cb, data=bg, model_output="probability")
    sv = explainer(qX)
    return sv


# ──────────────────────────────────────────────────────────────
# Plotting helpers
# ──────────────────────────────────────────────────────────────
def lipinski_card(feats: dict):
    cols = st.columns(4)
    for col, (key, (label, rule, check)) in zip(cols, LIPINSKI_RULES.items()):
        val = feats.get(key, float("nan"))
        ok  = check(val)
        color = "#27ae60" if ok else "#e74c3c"
        icon  = "✅" if ok else "❌"
        col.markdown(
            f"""
            <div style='padding:10px;border-radius:8px;background:{color}22;border:1px solid {color};text-align:center'>
                <div style='font-size:1.4em'>{icon}</div>
                <div style='font-weight:bold;color:{color}'>{label}</div>
                <div style='font-size:1.1em'>{val:.2f}</div>
                <div style='color:#888;font-size:.85em'>{rule}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    viols = feats.get("lipinski_violations", 0)
    if viols == 0:
        st.success("Passes all Lipinski rules — drug-like oral bioavailability predicted.")
    elif viols == 1:
        st.warning(f"1 Lipinski violation — borderline drug-likeness.")
    else:
        st.error(
            f"{int(viols)} Lipinski violations — molecule is **NOT drug-like** "
            "(but our model finds violators 2× easier to classify as HIV-active — "
            "Lipinski-breaking protease inhibitors are over-represented in HIV data)."
        )


def feature_category_pie(feats: dict, mi_idx, fcols):
    """Proportion of MI-selected features per category."""
    sel_names = [fcols[i] for i in mi_idx]
    cats = {"Domain": 0, "MACCS": 0, "Morgan": 0, "Fragments": 0, "Other": 0}
    for n in sel_names:
        if n in DOMAIN_FEATS:
            cats["Domain"] += 1
        elif n.startswith("maccs_"):
            cats["MACCS"] += 1
        elif n.startswith("mfp_"):
            cats["Morgan"] += 1
        elif n.startswith("fr_"):
            cats["Fragments"] += 1
        else:
            cats["Other"] += 1
    labels = [k for k, v in cats.items() if v > 0]
    sizes  = [v for v in cats.values() if v > 0]
    colors = ["#3498db", "#e67e22", "#2ecc71", "#e74c3c", "#9b59b6"][:len(labels)]
    fig, ax = plt.subplots(figsize=(4, 3))
    ax.pie(sizes, labels=labels, colors=colors, autopct="%1.0f%%", startangle=90,
           textprops={"fontsize": 9})
    ax.set_title("MI-400 Feature Composition", fontsize=10)
    plt.tight_layout()
    return fig


def shap_bar_chart(shap_vals, feature_names, top_k=15):
    """Simple horizontal bar chart of top SHAP contributions."""
    vals = shap_vals.values[0] if hasattr(shap_vals, "values") else shap_vals[0]
    pairs = sorted(zip(feature_names, vals), key=lambda x: abs(x[1]), reverse=True)[:top_k]
    names  = [p[0] for p in reversed(pairs)]
    values = [p[1] for p in reversed(pairs)]
    colors = ["#e74c3c" if v > 0 else "#3498db" for v in values]

    fig, ax = plt.subplots(figsize=(6, max(3, top_k * 0.4)))
    bars = ax.barh(names, values, color=colors, edgecolor="white", height=0.7)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("SHAP value (contribution to HIV-active probability)")
    ax.set_title(f"Top {top_k} Feature Contributions (SHAP)", fontsize=11)
    ax.tick_params(labelsize=8)
    red_p  = mpatches.Patch(color="#e74c3c", label="→ HIV-active")
    blue_p = mpatches.Patch(color="#3498db", label="→ HIV-inactive")
    ax.legend(handles=[red_p, blue_p], fontsize=8, loc="lower right")
    plt.tight_layout()
    return fig


def render_molecule_svg(smiles: str) -> str | None:
    try:
        from rdkit.Chem import rdMolDescriptors
        from rdkit.Chem.Draw import rdMolDraw2D
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        drawer = rdMolDraw2D.MolDraw2DSVG(300, 200)
        drawer.DrawMolecule(mol)
        drawer.FinishDrawing()
        return drawer.GetDrawingText()
    except Exception:
        return None


# ──────────────────────────────────────────────────────────────
# Sidebar
# ──────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🧬 HIV Activity Predictor")
    st.markdown("**Drug Molecule Property Prediction**  \n_YC Portfolio — 7-day Research Sprint_")
    st.markdown("---")

    st.markdown("### Model Performance")
    metrics = {
        "GIN+CatBoost Ensemble": "0.8114",
        "CatBoost MI-400 (best run)": "0.8105",
        "Anthony GIN+Edge": "0.7860",
        "Mark 3-model Ensemble": "0.7888",
        "OGB SOTA": "0.8476",
    }
    for model, auc in metrics.items():
        bold = "**" if model in ("GIN+CatBoost Ensemble", "OGB SOTA") else ""
        st.markdown(f"{bold}{model}: ROC-AUC = {auc}{bold}")

    st.markdown("---")
    st.markdown("### Key Findings")
    st.markdown("""
- Feature curation beats quantity (+0.043 AUC from MI-400 selection)
- MACCS keys: 2.4× more info-dense than Morgan FP
- Lipinski-violating actives: **2× easier to classify** (AUC 0.85 vs 0.67)
- LIME ↔ SHAP agreement: only **4%** per molecule
- Fragment descriptors in MI-400 are **noise** (removal +0.026 AUC)
    """)

    st.markdown("---")
    st.markdown("### Dataset")
    st.markdown("""
**ogbg-molhiv** (OGB scaffold split)
- 41,127 molecules
- 3.5% HIV-active
- SOTA: 0.8476 ROC-AUC
    """)

    st.markdown("---")
    st.markdown("[GitHub Repo](https://github.com/anthonyrodrigues443/Drug-Molecule-Property-Prediction) | "
                "Phase 7 — Sunday wrap-up")


# ──────────────────────────────────────────────────────────────
# Main area
# ──────────────────────────────────────────────────────────────
st.title("🧬 HIV Activity Predictor")
st.markdown(
    "Enter a SMILES string to predict **HIV activity probability**, check "
    "Lipinski compliance, and inspect which molecular features drive the prediction."
)

# Load model
cb, mi_idx, fcols, source = load_model()
if source == "surrogate":
    st.warning(
        "⚠️  Trained model artifacts not found — running **surrogate model** "
        "(lightweight CatBoost trained on ogbg-molhiv subset). "
        "Run `python -m src.train` to build the production ensemble."
    )
sel_feature_names = [fcols[i] for i in mi_idx]

# ── Tabs ──────────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs(["🔬 Single Molecule", "📊 Batch Analysis", "📋 All Experiments"])

# ─────────────────────────────────────────────────────────────
# TAB 1: Single molecule analysis
# ─────────────────────────────────────────────────────────────
with tab1:
    col_inp, col_ex = st.columns([2, 1])

    with col_ex:
        st.markdown("**Example molecules:**")
        chosen_ex = st.selectbox("Load example", ["— choose —"] + list(EXAMPLE_MOLECULES.keys()))
        if chosen_ex != "— choose —":
            default_smiles = EXAMPLE_MOLECULES[chosen_ex]
        else:
            default_smiles = ""

    with col_inp:
        smiles_input = st.text_input(
            "SMILES string",
            value=default_smiles,
            placeholder="e.g. CC(=O)Oc1ccccc1C(=O)O",
        )

    if smiles_input.strip():
        result = predict_molecule(smiles_input.strip(), cb, mi_idx, fcols)

        if result is None:
            st.error("Invalid SMILES — could not parse molecule.")
        else:
            prob  = result["probability"]
            feats = result["feats"]

            # ── Prediction banner ──────────────────────────────
            label = "HIV-ACTIVE" if prob >= 0.5 else "HIV-INACTIVE"
            color = "#e74c3c" if prob >= 0.5 else "#27ae60"
            st.markdown(
                f"""
                <div style='padding:16px;border-radius:10px;background:{color}22;border:2px solid {color};
                            text-align:center;margin-bottom:16px'>
                    <span style='font-size:2em;font-weight:bold;color:{color}'>{label}</span><br>
                    <span style='font-size:1.3em'>Probability: <b>{prob:.3f}</b></span>
                </div>
                """,
                unsafe_allow_html=True,
            )

            # Subgroup disclaimer
            viols = feats.get("lipinski_violations", 0)
            if viols == 0 and prob < 0.5:
                st.info(
                    "ℹ️  Lipinski-compliant inactive predictions should be interpreted with caution: "
                    "our model has only **3.3% recall** on Lipinski-compliant HIV actives "
                    "(true positives from this subgroup are frequently missed)."
                )

            # ── Layout: structure | lipinski | category pie ───
            c1, c2, c3 = st.columns([1, 1.8, 1.2])

            with c1:
                st.markdown("**Molecule Structure**")
                svg = render_molecule_svg(smiles_input)
                if svg:
                    st.image(svg.encode(), use_container_width=True)
                else:
                    st.markdown("_(structure rendering unavailable)_")

            with c2:
                st.markdown("**Lipinski Rule of 5**")
                lipinski_card(feats)

            with c3:
                st.markdown("**MI-400 Feature Composition**")
                pie_fig = feature_category_pie(feats, mi_idx, fcols)
                st.pyplot(pie_fig, use_container_width=True)
                plt.close(pie_fig)

            # ── Advanced descriptors ───────────────────────────
            with st.expander("📐 Full Molecular Descriptors"):
                desc_df = pd.DataFrame([{
                    "Descriptor": k.replace("_", " ").title(),
                    "Value": f"{feats[k]:.4f}" if isinstance(feats[k], float) else str(feats[k]),
                } for k in DOMAIN_FEATS])
                st.dataframe(desc_df, use_container_width=True, hide_index=True)

            # ── SHAP explanation ───────────────────────────────
            with st.expander("🔍 SHAP Feature Attribution (top 15 contributors)"):
                bg_smiles = [
                    "CC(=O)Oc1ccccc1C(=O)O",           # aspirin
                    "Cn1c(=O)c2c(ncn2C)n(C)c1=O",       # caffeine
                    "CC(C)Cc1ccc(cc1)C(C)C(=O)O",       # ibuprofen
                    "CC12CCC3C(C1CCC2O)CCC4=CC(=O)CCC34C",  # testosterone
                    "OC(=O)c1ccccc1O",                  # salicylic acid
                    "CN1C=NC2=C1C(=O)N(C(=O)N2C)C",     # theophylline
                    "OC[C@H]1OC(O)[C@H](O)[C@@H](O)[C@@H]1O",  # glucose
                    "CC(C)CN(CC(O)C(Cc1ccccc1)NC(=O)OC1CCOC1)S(=O)(=O)c1ccc(N)cc1",  # amprenavir
                ]
                with st.spinner("Computing SHAP values…"):
                    try:
                        sv = compute_shap_values(cb, bg_smiles, mi_idx, fcols, [smiles_input.strip()])
                        shap_fig = shap_bar_chart(sv, sel_feature_names, top_k=15)
                        st.pyplot(shap_fig, use_container_width=True)
                        plt.close(shap_fig)

                        # Show top 5 as table
                        vals   = sv.values[0]
                        top_df = pd.DataFrame({
                            "Feature":      sel_feature_names,
                            "SHAP Value":   vals,
                            "Direction":    ["→ HIV-active" if v > 0 else "→ HIV-inactive" for v in vals],
                        }).reindex(np.argsort(np.abs(vals))[::-1]).head(10).reset_index(drop=True)
                        top_df["SHAP Value"] = top_df["SHAP Value"].map("{:.4f}".format)
                        st.dataframe(top_df, use_container_width=True, hide_index=True)
                    except Exception as e:
                        st.warning(f"SHAP computation failed: {e}")


# ─────────────────────────────────────────────────────────────
# TAB 2: Batch analysis
# ─────────────────────────────────────────────────────────────
with tab2:
    st.markdown("### Batch Prediction")
    st.markdown("Enter one SMILES per line:")
    batch_text = st.text_area(
        "SMILES (one per line)",
        height=150,
        placeholder="\n".join(list(EXAMPLE_MOLECULES.values())[:3]),
    )
    if st.button("Run Batch Prediction", type="primary"):
        lines = [l.strip() for l in batch_text.strip().split("\n") if l.strip()]
        if not lines:
            st.warning("Please enter at least one SMILES.")
        else:
            rows = []
            for smi in lines:
                r = predict_molecule(smi, cb, mi_idx, fcols)
                if r is None:
                    rows.append({
                        "SMILES": smi[:40] + ("…" if len(smi) > 40 else ""),
                        "Probability": None,
                        "Prediction": "INVALID",
                        "Mol Weight": None,
                        "LogP": None,
                        "Lipinski Violations": None,
                    })
                else:
                    f = r["feats"]
                    rows.append({
                        "SMILES": smi[:40] + ("…" if len(smi) > 40 else ""),
                        "Probability": round(r["probability"], 4),
                        "Prediction": "HIV-ACTIVE" if r["probability"] >= 0.5 else "HIV-INACTIVE",
                        "Mol Weight": round(f["mol_weight"], 1),
                        "LogP": round(f["logp"], 2),
                        "Lipinski Violations": int(f["lipinski_violations"]),
                    })
            df_out = pd.DataFrame(rows)
            st.dataframe(
                df_out.style.applymap(
                    lambda v: "color: #e74c3c; font-weight: bold" if v == "HIV-ACTIVE"
                    else ("color: #27ae60" if v == "HIV-INACTIVE" else ""),
                    subset=["Prediction"],
                ),
                use_container_width=True, hide_index=True,
            )

            actives  = (df_out["Prediction"] == "HIV-ACTIVE").sum()
            invalids = (df_out["Prediction"] == "INVALID").sum()
            st.markdown(
                f"**Summary:** {len(lines)} molecules | "
                f"{actives} predicted active | "
                f"{len(lines)-actives-invalids} inactive | "
                f"{invalids} invalid SMILES"
            )


# ─────────────────────────────────────────────────────────────
# TAB 3: All experiments summary
# ─────────────────────────────────────────────────────────────
with tab3:
    st.markdown("### Full Experiment Leaderboard (7-day research sprint)")
    st.markdown("_100+ experiments across 2 researchers. All on OGB scaffold split._")

    leaderboard = [
        {"Phase": "P7", "Researcher": "Anthony", "Model": "GIN+CatBoost Ensemble",            "ROC-AUC": 0.8114, "Notes": "Production champion"},
        {"Phase": "P3", "Researcher": "Mark",    "Model": "CatBoost MI-400 (best run)",         "ROC-AUC": 0.8105, "Notes": "Feature curation"},
        {"Phase": "P3", "Researcher": "Mark",    "Model": "CatBoost MI-200",                    "ROC-AUC": 0.8019, "Notes": "Stable above 0.80"},
        {"Phase": "P3", "Researcher": "Anthony", "Model": "GIN+Edge",                           "ROC-AUC": 0.7860, "Notes": "Graph convolution champion"},
        {"Phase": "P3", "Researcher": "Anthony", "Model": "CatBoost AllTrad-1217",              "ROC-AUC": 0.7841, "Notes": "Full tabular features"},
        {"Phase": "P5", "Researcher": "Mark",    "Model": "3-model Ensemble (MI pools)",        "ROC-AUC": 0.7888, "Notes": "Tabular diversity = GNN"},
        {"Phase": "P4", "Researcher": "Mark",    "Model": "Tuned CatBoost MI-400",              "ROC-AUC": 0.7854, "Notes": "Optuna 40 trials"},
        {"Phase": "P5", "Researcher": "Anthony", "Model": "GIN+CB Cross-Paradigm Ensemble",     "ROC-AUC": 0.7862, "Notes": "+542 rescued, 0 hurt"},
        {"Phase": "P2", "Researcher": "Mark",    "Model": "MLP-Domain9 (9 features, 5K params)","ROC-AUC": 0.7670, "Notes": "Beats all GNNs"},
        {"Phase": "P1", "Researcher": "Mark",    "Model": "CatBoost auto_weight",               "ROC-AUC": 0.7782, "Notes": "Class imbalance winner"},
        {"Phase": "P1", "Researcher": "Anthony", "Model": "RF Combined",                        "ROC-AUC": 0.7707, "Notes": "Phase 1 champion"},
        {"Phase": "P2", "Researcher": "Anthony", "Model": "GIN (best GNN)",                     "ROC-AUC": 0.7053, "Notes": "Graph learning limit"},
        {"Phase": "P2", "Researcher": "Mark",    "Model": "MLP-Morgan1024",                     "ROC-AUC": 0.6736, "Notes": "Neural fails on sparse FP"},
        {"Phase": "P3", "Researcher": "Mark",    "Model": "Fragments-only CatBoost",            "ROC-AUC": 0.6999, "Notes": "Worst feature set"},
        {"Phase": "—",  "Researcher": "OGB",     "Model": "SOTA (literature)",                  "ROC-AUC": 0.8476, "Notes": "Target to beat"},
    ]
    lb_df = pd.DataFrame(leaderboard).sort_values("ROC-AUC", ascending=False).reset_index(drop=True)
    lb_df.index += 1

    def highlight_rows(row):
        if row["Model"] == "GIN+CatBoost Ensemble":
            return ["background-color: #27ae6022"] * len(row)
        if row["Researcher"] == "OGB":
            return ["background-color: #f39c1222"] * len(row)
        return [""] * len(row)

    st.dataframe(lb_df.style.apply(highlight_rows, axis=1), use_container_width=True)

    st.markdown("---")
    st.markdown("### Phase Summary")
    phases = [
        ("Phase 1", "Domain Research + EDA + Baseline",
         "CatBoost auto_weight (0.7782) beats RF (0.7707). ECFP radius sensitivity confirmed: r=2 is optimal."),
        ("Phase 2", "Multi-Model Experiment",
         "GNNs (0.70-0.71) all lose to CatBoost. MLP on 9 domain features (0.7670) beats all GNNs."),
        ("Phase 3", "Feature Engineering",
         "MI-400 selection: +0.043 AUC vs full pool. MACCS 2.4× more info-dense than Morgan."),
        ("Phase 4", "Hyperparameter Tuning + Error Analysis",
         "Lipinski-violating actives: 2× higher recall (0.828 vs 0.400). Tuning overfits to val scaffold."),
        ("Phase 5", "Advanced Techniques + Ablation",
         "Fragment features in MI-400 are noise (removal +0.026). Tabular ensemble matches GNN."),
        ("Phase 6", "Explainability (SHAP + LIME)",
         "LIME ↔ SHAP agreement: 4% per molecule. Lipinski-compliant actives: 3.3% recall."),
        ("Phase 7", "Production Pipeline + Tests",
         "28 tests all pass. GIN+CatBoost ensemble at 0.8114. Streamlit UI deployed."),
    ]
    for phase, title, summary in phases:
        with st.expander(f"**{phase}: {title}**"):
            st.markdown(summary)
