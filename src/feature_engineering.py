"""
Feature engineering for Drug Molecule Property Prediction.
Computes domain descriptors, Morgan fingerprints, MACCS keys, and fragment counts.
Supports MI-based feature selection.
"""
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors, AllChem, MACCSkeys, Fragments
from sklearn.feature_selection import mutual_info_classif

FRAG_FUNCS = [(n, getattr(Fragments, n)) for n in sorted(dir(Fragments)) if n.startswith('fr_')]


def compute_all_features(smiles: str) -> dict | None:
    """Compute full feature vector for a single SMILES string."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    feats = {
        'mol_weight': Descriptors.MolWt(mol),
        'logp': Descriptors.MolLogP(mol),
        'hbd': rdMolDescriptors.CalcNumHBD(mol),
        'hba': rdMolDescriptors.CalcNumHBA(mol),
        'tpsa': rdMolDescriptors.CalcTPSA(mol),
        'rotatable_bonds': rdMolDescriptors.CalcNumRotatableBonds(mol),
        'aromatic_rings': rdMolDescriptors.CalcNumAromaticRings(mol),
        'ring_count': rdMolDescriptors.CalcNumRings(mol),
        'heavy_atom_count': mol.GetNumHeavyAtoms(),
        'fraction_csp3': rdMolDescriptors.CalcFractionCSP3(mol),
        'num_heteroatoms': rdMolDescriptors.CalcNumHeteroatoms(mol),
        'lipinski_violations': sum([
            Descriptors.MolWt(mol) > 500, Descriptors.MolLogP(mol) > 5,
            rdMolDescriptors.CalcNumHBD(mol) > 5, rdMolDescriptors.CalcNumHBA(mol) > 10
        ]),
        'bertz_ct': Descriptors.BertzCT(mol),
        'labute_asa': Descriptors.LabuteASA(mol),
    }

    fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=1024)
    for j in range(1024):
        feats[f'mfp_{j}'] = fp.GetBit(j)

    mk = MACCSkeys.GenMACCSKeys(mol)
    for j in range(167):
        feats[f'maccs_{j}'] = mk.GetBit(j)

    for fname, func in FRAG_FUNCS:
        try:
            feats[fname] = func(mol)
        except Exception:
            feats[fname] = 0

    return feats


def compute_features_batch(smiles_list: list[str]) -> pd.DataFrame:
    """Compute features for a list of SMILES. Drops invalid molecules."""
    rows = []
    valid_indices = []
    for i, smi in enumerate(smiles_list):
        f = compute_all_features(smi)
        if f is not None:
            rows.append(f)
            valid_indices.append(i)
    df = pd.DataFrame(rows)
    df.index = valid_indices
    return df


def select_features_mi(X_train: np.ndarray, y_train: np.ndarray,
                        k: int = 400, seed: int = 42) -> np.ndarray:
    """Select top-k features by mutual information. Returns index array."""
    mi = mutual_info_classif(X_train, y_train, random_state=seed, n_neighbors=5)
    return np.argsort(mi)[-k:]


def get_feature_columns(df: pd.DataFrame) -> list[str]:
    """Return feature column names (excludes metadata columns)."""
    exclude = {'smiles', 'hiv_active', 'split', 'idx', 'y'}
    return [c for c in df.columns if c not in exclude]
