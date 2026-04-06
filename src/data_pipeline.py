"""
Data pipeline for Drug Molecule Property Prediction.
Downloads ESOL and Lipophilicity from MoleculeNet via DeepChem.
Computes RDKit molecular descriptors and Morgan fingerprints.
"""
import numpy as np
import pandas as pd
from pathlib import Path

import deepchem as dc
from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors, AllChem
from rdkit.Chem.Scaffolds import MurckoScaffold

DATA_DIR = Path(__file__).parent.parent / "data"


def download_esol() -> pd.DataFrame:
    tasks, datasets, transformers = dc.molnet.load_delaney(
        featurizer="ECFP", splitter="scaffold", reload=True,
        data_dir=str(DATA_DIR / "raw"),
    )
    train, val, test = datasets

    def _to_df(ds, split):
        return pd.DataFrame({"smiles": ds.ids, "log_solubility": ds.y.flatten(), "split": split})

    return pd.concat([_to_df(train, "train"), _to_df(val, "val"), _to_df(test, "test")], ignore_index=True)


def compute_lipinski_features(smiles: str) -> dict:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    mw = Descriptors.MolWt(mol)
    logp = Descriptors.MolLogP(mol)
    hbd = rdMolDescriptors.CalcNumHBD(mol)
    hba = rdMolDescriptors.CalcNumHBA(mol)
    return {
        "mol_weight": mw, "logp": logp, "hbd": hbd, "hba": hba,
        "tpsa": rdMolDescriptors.CalcTPSA(mol),
        "rotatable_bonds": rdMolDescriptors.CalcNumRotatableBonds(mol),
        "aromatic_rings": rdMolDescriptors.CalcNumAromaticRings(mol),
        "ring_count": rdMolDescriptors.CalcNumRings(mol),
        "heavy_atom_count": mol.GetNumHeavyAtoms(),
        "fraction_csp3": rdMolDescriptors.CalcFractionCSP3(mol),
        "num_stereocenters": len(Chem.FindMolChiralCenters(mol, includeUnassigned=True)),
        "molar_refractivity": Descriptors.MolMR(mol),
        "bertz_ct": Descriptors.BertzCT(mol),
        "qed": Descriptors.qed(mol),
        "lip_mw_ok": int(mw <= 500), "lip_logp_ok": int(logp <= 5),
        "lip_hbd_ok": int(hbd <= 5), "lip_hba_ok": int(hba <= 10),
        "lipinski_violations": int(mw > 500) + int(logp > 5) + int(hbd > 5) + int(hba > 10),
        "passes_lipinski": int(mw <= 500 and logp <= 5 and hbd <= 5 and hba <= 10),
    }


def compute_morgan_fingerprints(smiles: str, radius: int = 2, n_bits: int = 2048) -> np.ndarray:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=radius, nBits=n_bits)
    return np.array(fp)


def compute_murcko_scaffold(smiles: str) -> str:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return Chem.MolToSmiles(MurckoScaffold.GetScaffoldForMol(mol))


def build_feature_matrix(df: pd.DataFrame, fp_radius: int = 2, fp_bits: int = 2048) -> pd.DataFrame:
    lip_records, fp_records, valid_mask = [], [], []
    for smi in df["smiles"]:
        lip = compute_lipinski_features(smi)
        fp = compute_morgan_fingerprints(smi, radius=fp_radius, n_bits=fp_bits)
        if lip is None or fp is None:
            valid_mask.append(False); lip_records.append(None); fp_records.append(None)
        else:
            valid_mask.append(True); lip_records.append(lip); fp_records.append(fp)

    df_valid = df[valid_mask].copy().reset_index(drop=True)
    lip_df = pd.DataFrame([r for r in lip_records if r is not None])
    fp_df = pd.DataFrame(np.array([r for r in fp_records if r is not None]),
                         columns=[f"fp_{i}" for i in range(fp_bits)])
    df_valid["murcko_scaffold"] = [compute_murcko_scaffold(s) for s in df_valid["smiles"]]
    return pd.concat([df_valid.reset_index(drop=True), lip_df.reset_index(drop=True),
                      fp_df.reset_index(drop=True)], axis=1)
