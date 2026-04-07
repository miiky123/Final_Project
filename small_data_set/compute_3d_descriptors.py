import os
import numpy as np
import pandas as pd

from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem
from rdkit.Chem import rdMolDescriptors
from rdkit.Chem.MolStandardize import rdMolStandardize

RDLogger.DisableLog("rdApp.error")

DATA_DIR = "small_data_set/data"
INPUT_DATA = os.path.join(DATA_DIR, "tables1_4_consolidated.csv")
OUTPUT_DATA = os.path.join(DATA_DIR, "tables1_4_with_3d.csv")
OUTPUT_3D_ONLY = os.path.join(DATA_DIR, "tables1_4_3d_only.csv")

REIONIZER = rdMolStandardize.Reionizer()

SOLVENT_CONFIGS = {
    "water": {"num_confs": 30, "seed": 11},
    "chloroform": {"num_confs": 30, "seed": 22},
    "octanol": {"num_confs": 30, "seed": 33},
}

BASE_3D_COLUMNS = [
    "RadiusOfGyration",
    "Asphericity",
    "Eccentricity",
    "InertialShapeFactor",
    "NPR1",
    "NPR2",
    "PMI1",
    "PMI2",
    "PMI3",
    "SpherocityIndex",
    "ChargeMean",
    "ChargeMax",
    "ChargeMin",
    "AbsChargeMean",
    "AbsChargeMax",
]

THREED_FEATURE_COLUMNS = [
    f"{solvent}_{name}"
    for solvent in SOLVENT_CONFIGS
    for name in BASE_3D_COLUMNS
]


def _prepare_mol(smiles: str):
    """
    Build a protonated/standardized RDKit molecule from a SMILES string.

    Note:
    RDKit does not natively run true solvent-dependent molecular dynamics.
    The code below generates separate conformer ensembles that are labeled by
    solvent name, but the conformer generation itself is still an RDKit/MMFF
    approximation rather than an explicit solvent simulation.
    """
    if not isinstance(smiles, str) or not smiles.strip():
        return None

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    mol = rdMolStandardize.Cleanup(mol)
    mol = REIONIZER.reionize(mol)
    mol = Chem.AddHs(mol)

    return mol


def _embed_and_optimize_conformers(mol, num_confs: int, random_seed: int):
    """
    Generate a conformer ensemble using ETKDGv3 and optimize it with MMFF94.
    """
    work_mol = Chem.Mol(mol)

    params = AllChem.ETKDGv3()
    params.randomSeed = random_seed
    params.pruneRmsThresh = 0.25
    params.useSmallRingTorsions = True
    params.useMacrocycleTorsions = True
    params.enforceChirality = True
    params.maxIterations = 1000

    conf_ids = list(AllChem.EmbedMultipleConfs(work_mol, numConfs=num_confs, params=params))
    if not conf_ids:
        return work_mol, []

    mmff_props = AllChem.MMFFGetMoleculeProperties(work_mol, mmffVariant="MMFF94")
    if mmff_props is not None:
        try:
            AllChem.MMFFOptimizeMoleculeConfs(
                work_mol,
                mmffVariant="MMFF94",
                maxIters=500,
                numThreads=0,
            )
        except Exception:
            pass
    else:
        try:
            AllChem.UFFOptimizeMoleculeConfs(work_mol, maxIters=500, numThreads=0)
        except Exception:
            pass

    return work_mol, conf_ids


def _compute_charge_summary(mol):
    """
    Compute atom-level Gasteiger charge summary statistics.
    These are identical across conformers of the same protonation state, but we
    keep them in the 3D feature block because the mentor asked to enrich the 3D
    stage with electrostatic information as well.
    """
    try:
        AllChem.ComputeGasteigerCharges(mol)
    except Exception:
        return {
            "ChargeMean": np.nan,
            "ChargeMax": np.nan,
            "ChargeMin": np.nan,
            "AbsChargeMean": np.nan,
            "AbsChargeMax": np.nan,
        }

    charges = []
    for atom in mol.GetAtoms():
        try:
            charge = float(atom.GetProp("_GasteigerCharge"))
        except Exception:
            charge = np.nan
        if np.isfinite(charge):
            charges.append(charge)

    if not charges:
        return {
            "ChargeMean": np.nan,
            "ChargeMax": np.nan,
            "ChargeMin": np.nan,
            "AbsChargeMean": np.nan,
            "AbsChargeMax": np.nan,
        }

    charges = np.asarray(charges, dtype=float)
    abs_charges = np.abs(charges)
    return {
        "ChargeMean": float(np.mean(charges)),
        "ChargeMax": float(np.max(charges)),
        "ChargeMin": float(np.min(charges)),
        "AbsChargeMean": float(np.mean(abs_charges)),
        "AbsChargeMax": float(np.max(abs_charges)),
    }


def _compute_3d_descriptors_for_conf(mol, conf_id: int):
    """
    Compute a compact set of 3D shape descriptors for one conformer.
    """
    values = {}
    funcs = {
        "RadiusOfGyration": rdMolDescriptors.CalcRadiusOfGyration,
        "Asphericity": rdMolDescriptors.CalcAsphericity,
        "Eccentricity": rdMolDescriptors.CalcEccentricity,
        "InertialShapeFactor": rdMolDescriptors.CalcInertialShapeFactor,
        "NPR1": rdMolDescriptors.CalcNPR1,
        "NPR2": rdMolDescriptors.CalcNPR2,
        "PMI1": rdMolDescriptors.CalcPMI1,
        "PMI2": rdMolDescriptors.CalcPMI2,
        "PMI3": rdMolDescriptors.CalcPMI3,
        "SpherocityIndex": rdMolDescriptors.CalcSpherocityIndex,
    }

    for name, func in funcs.items():
        try:
            values[name] = float(func(mol, confId=conf_id))
        except Exception:
            values[name] = np.nan

    values.update(_compute_charge_summary(mol))
    return values


def _average_descriptor_dicts(descriptor_rows, column_names):
    """
    Average descriptor values over all conformers in one ensemble.
    """
    if not descriptor_rows:
        return {name: np.nan for name in column_names}

    frame = pd.DataFrame(descriptor_rows)
    frame = frame.reindex(columns=column_names)
    return frame.mean(axis=0, skipna=True).to_dict()


def compute_3d_descriptor_block(smiles: str):
    """
    For one molecule, build three conformer ensembles (water/chloroform/octanol),
    compute descriptors for each conformer, and average within each ensemble.
    """
    base_mol = _prepare_mol(smiles)
    if base_mol is None:
        return {col: np.nan for col in THREED_FEATURE_COLUMNS}

    output = {}
    for solvent_name, cfg in SOLVENT_CONFIGS.items():
        mol_with_confs, conf_ids = _embed_and_optimize_conformers(
            base_mol,
            num_confs=cfg["num_confs"],
            random_seed=cfg["seed"],
        )

        rows = []
        for conf_id in conf_ids:
            rows.append(_compute_3d_descriptors_for_conf(mol_with_confs, conf_id))

        averaged = _average_descriptor_dicts(rows, BASE_3D_COLUMNS)
        for key, value in averaged.items():
            output[f"{solvent_name}_{key}"] = value

    return output


def add_3d_descriptors_to_dataframe(df: pd.DataFrame, smiles_col: str = "SMILES"):
    """
    Compute 3D descriptors for every molecule in the dataframe and append them.
    """
    if smiles_col not in df.columns:
        raise ValueError(f"Missing required SMILES column: {smiles_col}")

    rows = []
    total = len(df)
    for idx, smiles in enumerate(df[smiles_col].tolist(), start=1):
        print(f"[3D] Processing molecule {idx}/{total}")
        rows.append(compute_3d_descriptor_block(smiles))

    desc_df = pd.DataFrame(rows)
    desc_df = desc_df.reindex(columns=THREED_FEATURE_COLUMNS)

    return pd.concat([df.reset_index(drop=True), desc_df.reset_index(drop=True)], axis=1)


def main():
    if not os.path.exists(INPUT_DATA):
        raise FileNotFoundError(
            f"Could not find input file: {INPUT_DATA}\n"
            "Run spliting.py first so tables1_4_consolidated.csv is created."
        )

    df = pd.read_csv(INPUT_DATA)
    print("=" * 60)
    print("Computing 3D descriptor ensembles")
    print("=" * 60)
    print("Input rows:", len(df))
    print("Solvents:", ", ".join(SOLVENT_CONFIGS.keys()))
    print("3D features per solvent:", len(BASE_3D_COLUMNS))
    print("Total new 3D features:", len(THREED_FEATURE_COLUMNS))

    out = add_3d_descriptors_to_dataframe(df, smiles_col="SMILES")

    out.to_csv(OUTPUT_DATA, index=False)
    out[THREED_FEATURE_COLUMNS].to_csv(OUTPUT_3D_ONLY, index=False)

    print("\nSaved full dataset with 3D descriptors:", OUTPUT_DATA)
    print("Saved 3D-only descriptor table:", OUTPUT_3D_ONLY)
    print("Final rows:", len(out))


if __name__ == "__main__":
    main()
