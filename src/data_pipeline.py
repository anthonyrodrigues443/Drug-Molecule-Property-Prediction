"""
Data pipeline for Drug Molecule Property Prediction.
Downloads ESOL and Lipophilicity from MoleculeNet via DeepChem.
Computes RDKit molecular descriptors and Morgan fingerprints.
"""
import os
import json
import numpy as np
import pandas as pd
from pathlib import Path

import deepchem as dc
from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors, AllChem, Draw
from rdkit.Chem.Scaffolds import MurckoScaffold

DATA_DIR = Path(__file__).parent.parent / "data"


def download_esol() -> pd.DataFrame:
    """Download ESOL dataset (log solubility in mols/L) from MoleculeNet."""
    tasks, datasets, transformers = dc.molnet.load_delaney(
        featurizer="ECFP",
        splitter="scaffold",
        reload=True,
        data_dir=str(DATA_DIR / "raw"),
    )
    train, val, test = datasets

    def _dc_dataset_to_df(ds, split):
        smiles = ds.ids
        y = ds.y.flatten()
        return pd.DataFrame({"smiles": smiles, "log_solubility": y, "split": split})

    df = pd.concat(
        [
            _dc_dataset_to_df(train, "train"),
            _dc_dataset_to_df(val, "val"),
            _dc_dataset_to_df(test, "test"),
        ],
        ignore_index=True,
    )
    return df


def download_lipophilicity() -> pd.DataFrame:
    """Download Lipophilicity dataset (logD at pH 7.4) from MoleculeNet."""
    tasks, datasets, transformers = dc.molnet.load_lipo(
        featurizer="ECFP",
        splitter="scaffold",
        reload=True,
        data_dir=str(DATA_DIR / "raw"),
    )
    train, val, test = datasets

    def _dc_dataset_to_df(ds, split):
        smiles = ds.ids
        y = ds.y.flatten()
        return pd.DataFrame({"smiles": smiles, "exp": y, "split": split})

    df = pd.concat(
        [
            _dc_dataset_to_df(train, "train"),
            _dc_dataset_to_df(val, "val"),
            _dc_dataset_to_df(test, "test"),
        ],
        ignore_index=True,
    )
    return df


def compute_lipinski_features(smiles: str) -> dict:
    """Compute Lipinski Rule of 5 features + ADMET-relevant descriptors."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return {
        "mol_weight": Descriptors.MolWt(mol),
        "logp": Descriptors.MolLogP(mol),
        "hbd": rdMolDescriptors.CalcNumHBD(mol),
        "hba": rdMolDescriptors.CalcNumHBA(mol),
        "tpsa": rdMolDescriptors.CalcTPSA(mol),
        "rotatable_bonds": rdMolDescriptors.CalcNumRotatableBonds(mol),
        "aromatic_rings": rdMolDescriptors.CalcNumAromaticRings(mol),
        "ring_count": rdMolDescriptors.CalcNumRings(mol),
        "heavy_atom_count": mol.GetNumHeavyAtoms(),
        "fraction_csp3": rdMolDescriptors.CalcFractionCSP3(mol),
        "num_stereocenters": len(Chem.FindMolChiralCenters(mol, includeUnassigned=True)),
        "molar_refractivity": Descriptors.MolMR(mol),
        "bertz_ct": Descriptors.BertzCT(mol),
        "ipc": Descriptors.Ipc(mol),
        "qed": Descriptors.qed(mol),
        # Lipinski compliance flags
        "lip_mw_ok": int(Descriptors.MolWt(mol) <= 500),
        "lip_logp_ok": int(Descriptors.MolLogP(mol) <= 5),
        "lip_hbd_ok": int(rdMolDescriptors.CalcNumHBD(mol) <= 5),
        "lip_hba_ok": int(rdMolDescriptors.CalcNumHBA(mol) <= 10),
        "lipinski_violations": (
            int(Descriptors.MolWt(mol) > 500)
            + int(Descriptors.MolLogP(mol) > 5)
            + int(rdMolDescriptors.CalcNumHBD(mol) > 5)
            + int(rdMolDescriptors.CalcNumHBA(mol) > 10)
        ),
        "passes_lipinski": int(
            Descriptors.MolWt(mol) <= 500
            and Descriptors.MolLogP(mol) <= 5
            and rdMolDescriptors.CalcNumHBD(mol) <= 5
            and rdMolDescriptors.CalcNumHBA(mol) <= 10
        ),
    }


def compute_morgan_fingerprints(smiles: str, radius: int = 2, n_bits: int = 2048) -> np.ndarray:
    """Compute Morgan (ECFP) circular fingerprints."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=radius, nBits=n_bits)
    return np.array(fp)


def compute_murcko_scaffold(smiles: str) -> str:
    """Extract Murcko scaffold for clustering molecules by structural class."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    scaffold = MurckoScaffold.GetScaffoldForMol(mol)
    return Chem.MolToSmiles(scaffold)


def build_feature_matrix(df: pd.DataFrame, fp_radius: int = 2, fp_bits: int = 2048) -> pd.DataFrame:
    """Build full feature matrix: Lipinski descriptors + Morgan fingerprints."""
    lip_records = []
    fp_records = []
    valid_mask = []

    for smi in df["smiles"]:
        lip = compute_lipinski_features(smi)
        fp = compute_morgan_fingerprints(smi, radius=fp_radius, n_bits=fp_bits)
        if lip is None or fp is None:
            valid_mask.append(False)
            lip_records.append(None)
            fp_records.append(None)
        else:
            valid_mask.append(True)
            lip_records.append(lip)
            fp_records.append(fp)

    df_valid = df[valid_mask].copy().reset_index(drop=True)
    lip_df = pd.DataFrame([r for r in lip_records if r is not None])
    fp_arr = np.array([r for r in fp_records if r is not None])
    fp_df = pd.DataFrame(fp_arr, columns=[f"fp_{i}" for i in range(fp_bits)])

    df_valid["murcko_scaffold"] = [compute_murcko_scaffold(s) for s in df_valid["smiles"]]
    df_features = pd.concat([df_valid.reset_index(drop=True), lip_df.reset_index(drop=True), fp_df.reset_index(drop=True)], axis=1)
    return df_features


if __name__ == "__main__":
    print("Downloading ESOL dataset...")
    esol_df = download_esol()
    print(f"ESOL: {len(esol_df)} molecules")
    print(esol_df.head())

    print("\nBuilding feature matrix...")
    esol_features = build_feature_matrix(esol_df)
    esol_features.to_csv(DATA_DIR / "processed" / "esol_features.csv", index=False)
    print(f"Saved features: {esol_features.shape}")
