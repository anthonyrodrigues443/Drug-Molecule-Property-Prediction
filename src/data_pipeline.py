"""
Data pipeline for Drug Molecule Property Prediction (ogbg-molhiv).
Downloads from OGB, computes RDKit descriptors + Morgan fingerprints.
"""
import numpy as np
import pandas as pd
from pathlib import Path

from ogb.graphproppred import GraphPropPredDataset
from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors, AllChem

DATA_DIR = Path(__file__).parent.parent / "data"


def download_ogbg_molhiv():
    dataset = GraphPropPredDataset(name="ogbg-molhiv", root=str(DATA_DIR / "raw"))
    split_idx = dataset.get_idx_split()

    smiles_df = pd.read_csv(DATA_DIR / "raw" / "ogbg_molhiv" / "mapping" / "mol.csv.gz")
    smiles_list = smiles_df["smiles"].tolist()
    labels = dataset.labels.flatten()

    train_set = set(split_idx["train"].tolist())
    val_set = set(split_idx["valid"].tolist())
    test_set = set(split_idx["test"].tolist())

    rows = []
    for i, (smi, label) in enumerate(zip(smiles_list, labels)):
        split = "train" if i in train_set else "val" if i in val_set else "test" if i in test_set else "train"
        rows.append({"smiles": smi, "hiv_active": int(label), "split": split})
    return pd.DataFrame(rows)


def compute_lipinski_features(smiles):
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


def compute_morgan_fingerprints(smiles, radius=2, n_bits=1024):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return np.array(AllChem.GetMorganFingerprintAsBitVect(mol, radius=radius, nBits=n_bits))


def build_feature_matrix(df, fp_radius=2, fp_bits=1024):
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
