# Auxiliary Training — Permeability Prediction from SMILES

Trains `SMILES → Caco-2 permeability (Papp)` models as an **auxiliary predictor**
for the [Wadhams human-PK platform](https://github.com/fbatema1/FHB_Human_PK_From_Structure_RShiny).

## Why

Human CL/Vd training data is small (~1,140 approved drugs). Permeability has a
much larger public dataset. We train a robust structure→permeability model here,
then use its **predictions** as a dense feature for the human-PK model (transfer
learning). Permeability is most directly linked to absorption/distribution; we
expect it to help **Vd** more than CL.

## Workflow

1. **Train + benchmark standalone** — RF, XGB, GNN on the Caco-2 data, with RDKit
   descriptors so the model captures how molecular features drive permeability.
   Report accuracy (R², GMFE, etc.) — we only trust it as a feature if it's solid.
2. **Predict** Papp for all approved drugs in the main PK dataset.
3. **Ablate** as a CL/Vd feature on the scaffold split — must beat physchem-only.

## Data (`data/`)

- `permeability_chembl_raw.csv` — 8,463 ChEMBL Caco-2 Papp records, 5,840 compounds.
  Columns include `canonical_smiles`, `standard_value`, `standard_units`,
  `description` (apical→basolateral vs basolateral→apical), `assay_organism`.

**Open prep decisions (settle before training):**
- Direction: keep apical→basolateral (absorptive) only, or also include
  ~2,977 direction-ambiguous "Papp" records? (more data vs cleaner endpoint)
- Units: harmonize to ×10⁻⁶ cm/s (`ucm/s`/`10'-6 cm/s` ×1, `10'-5cm/s` ×10).
- Duplicates: median Papp per canonical compound.
- Model target: log10(Papp) regression.

## Leakage rule
Exclude the main project's scaffold-**test** compounds from training before
generating permeability features for them.

## Dependencies
`rdkit`, `scikit-learn`, `xgboost`, `torch` + `torch_geometric`, `shap`, `optuna`.
