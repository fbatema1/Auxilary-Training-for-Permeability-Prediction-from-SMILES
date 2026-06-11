"""
build_permeability_dataset.py
=============================
Cleans the raw ChEMBL Caco-2 export into a single-assay training table for the
permeability auxiliary model, with a Bemis-Murcko scaffold split.

Endpoint decision (data-driven, see README):
  - Caco-2 cell line ONLY (drop PAMPA / MDCK / RRCK / LLC-PK1 — different scales)
  - apical→basolateral OR undirected (drop explicit basolateral→apical efflux)
  - units harmonised to ×10⁻⁶ cm/s
  - median Papp per RDKit-canonical compound
  - target = log10(Papp)

Outputs (data/):
  - permeability_clean.csv   columns: smiles, papp, log10_papp, split
  - permeability_split_summary.txt

Run:
    python scripts/build_permeability_dataset.py
"""
import pandas as pd, numpy as np
from pathlib import Path
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold
from rdkit import RDLogger; RDLogger.DisableLog("rdApp.*")

DATA = Path(__file__).resolve().parents[1] / "data"
RANDOM_STATE = 42
TEST_FRAC = 0.20

UNIT_FACTOR = {"ucm/s":1.0, "10'-6 cm/s":1.0, "10'-6cm/s":1.0,
               "10'6cm/s":1.0, "10'-5cm/s":10.0}


def canon(s):
    m = Chem.MolFromSmiles(str(s));  return Chem.MolToSmiles(m) if m else None

def scaffold(s):
    try:
        return MurckoScaffold.MurckoScaffoldSmiles(mol=Chem.MolFromSmiles(s))
    except Exception:
        return None


def run():
    df = pd.read_csv(DATA / "permeability_chembl_raw.csv", low_memory=False)
    desc = df["description"].fillna("").str.lower()
    ct   = df["assay_cell_type"].fillna("")

    is_caco = ct.str.contains("Caco-2", case=False)
    is_ab   = desc.str.contains("apical to basolateral")
    is_ba   = desc.str.contains("basolateral to apical")

    keep = is_caco & (is_ab | (~is_ab & ~is_ba))   # Caco-2, A→B or undirected
    df = df[keep].copy()

    df["papp"] = df["standard_value"] * df["standard_units"].map(UNIT_FACTOR)
    df = df.dropna(subset=["papp"])
    df = df[(df["papp"] > 0) & (df["papp"] < 1000)]

    df["smiles"] = df["canonical_smiles"].map(canon)
    df = df.dropna(subset=["smiles"])

    # median Papp per canonical compound
    g = df.groupby("smiles")["papp"].median().reset_index()
    g["log10_papp"] = np.log10(g["papp"])
    print(f"Clean Caco-2 compounds: {len(g)}")

    # ── Bemis-Murcko scaffold split ──────────────────────────────────────────
    g["scaffold"] = g["smiles"].map(scaffold)
    rng = np.random.default_rng(RANDOM_STATE)
    scs = g.groupby("scaffold").size().sort_values(ascending=False)
    test_smiles, n_target = set(), int(len(g) * TEST_FRAC)
    # fill test set from smallest scaffolds up (keeps big common scaffolds in train)
    n_test = 0
    for sc in scs.index[::-1]:
        members = g.loc[g["scaffold"] == sc, "smiles"].tolist()
        if n_test + len(members) <= n_target:
            test_smiles.update(members); n_test += len(members)
    g["split"] = np.where(g["smiles"].isin(test_smiles), "test", "train")

    out = g[["smiles", "papp", "log10_papp", "split"]]
    out.to_csv(DATA / "permeability_clean.csv", index=False)

    ntr = (out["split"]=="train").sum(); nte = (out["split"]=="test").sum()
    summary = "\n".join([
        "PERMEABILITY CLEAN DATASET",
        "="*45,
        f"Endpoint: Caco-2 A→B + undirected, ×10⁻⁶ cm/s, log10",
        f"Compounds:  {len(out)}",
        f"  train:    {ntr}",
        f"  test:     {nte}  ({100*nte/len(out):.1f}%)",
        f"Unique scaffolds: {g['scaffold'].nunique()}",
        f"log10_papp  mean={out['log10_papp'].mean():.3f}  std={out['log10_papp'].std():.3f}"
        f"  range=[{out['log10_papp'].min():.2f}, {out['log10_papp'].max():.2f}]",
        f"Papp (×10⁻⁶ cm/s) median={out['papp'].median():.2f}",
    ])
    print("\n"+summary)
    (DATA / "permeability_split_summary.txt").write_text(summary)
    print(f"\nSaved: {DATA/'permeability_clean.csv'}")


if __name__ == "__main__":
    run()
