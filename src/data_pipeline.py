"""
Data pipeline for Drug Molecule Property Prediction.
Downloads ogbg-molhiv from Open Graph Benchmark.
Computes RDKit molecular descriptors and Morgan fingerprints.
"""
import numpy as np
import pandas as pd
from pathlib import Path

from ogb.graphproppred import GraphPropPredDataset
from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors, AllChem
from rdkit.Chem.Scaffolds import MurckoScaffold

DATA_DIR = Path(__file__).parent.parent / "data"


def download_ogbg_molhiv():
    """Download ogbg-molhiv dataset and return SMILES + labels + split indices."""
    dataset = GraphPropPredDataset(name="ogbg-molhiv", root=str(DATA_DIR / "raw"))
    split_idx = dataset.get_idx_split()

    smiles_df = pd.read_csv(DATA_DIR / "raw" / "ogbg_molhiv" / "mapping" / "mol.csv.gz")
    smiles_list = smiles_df["smiles"].tolist()
    labels = dataset.labels.flatten()

    train_set = set(split_idx["train"].tolist()) if hasattr(split_idx["train"], 'tolist') else set(split_idx["train"])
    val_set = set(split_idx["valid"].tolist()) if hasattr(split_idx["valid"], 'tolist') else set(split_idx["valid"])
    test_set = set(split_idx["test"].tolist()) if hasattr(split_idx["test"], 'tolist') else set(split_idx["test"])

    rows = []
    for i, (smi, label) in enumerate(zip(smiles_list, labels)):
        if i in train_set:
            split = "train"
        elif i in val_set:
            split = "val"
        elif i in test_set:
            split = "test"
        else:
            split = "train"
        rows.append({"smiles": smi, "hiv_active": int(label), "split": split})

    return pd.DataFrame(rows)


def compute_lipinski_features(smiles: str) -> dict:
    """Compute Lipinski Ro5 + ADMET-relevant molecular descriptors."""
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
        "molar_refractivity": Descriptors.MolMR(mol),
        "qed": Descriptors.qed(mol),
        "lipinski_violations": int(mw > 500) + int(logp > 5) + int(hbd > 5) + int(hba > 10),
        "passes_lipinski": int(mw <= 500 and logp <= 5 and hbd <= 5 and hba <= 10),
    }


def compute_morgan_fingerprints(smiles: str, radius: int = 2, n_bits: int = 1024) -> np.ndarray:
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


def build_feature_matrix(df: pd.DataFrame, fp_radius: int = 2, fp_bits: int = 1024) -> pd.DataFrame:
    """Build feature matrix: Lipinski descriptors + Morgan fingerprints."""
    lip_records, fp_records, valid_mask = [], [], []
    for smi in df["smiles"]:
        lip = compute_lipinski_features(smi)
        fp = compute_morgan_fingerprints(smi, radius=fp_radius, n_bits=fp_bits)
        if lip is None or fp is None:
            valid_mask.append(False)
        else:
            valid_mask.append(True)
            lip_records.append(lip)
            fp_records.append(fp)

    df_valid = df[valid_mask].copy().reset_index(drop=True)
    lip_df = pd.DataFrame(lip_records)
    fp_df = pd.DataFrame(np.array(fp_records), columns=[f"fp_{i}" for i in range(fp_bits)])

    return pd.concat([df_valid.reset_index(drop=True), lip_df.reset_index(drop=True),
                      fp_df.reset_index(drop=True)], axis=1)
