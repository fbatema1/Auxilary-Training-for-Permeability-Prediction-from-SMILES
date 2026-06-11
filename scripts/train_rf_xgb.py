"""
train_rf_xgb.py
===============
Trains RF + XGB to predict log10 Caco-2 Papp from structure, and benchmarks on
the scaffold-split test set. Self-contained featurization (RDKit 2D descriptors
+ Morgan fingerprint) so this repo stands alone.

Metrics mirror the main Wadhams PK models (log scale): R², RMSE, GMFE,
within-2-fold, within-3-fold.

Outputs (models/):
  - rf_permeability.pkl, xgb_permeability.pkl
  - permeability_rf_xgb_results.json

Run:
    python scripts/train_rf_xgb.py
"""
import json, pickle, warnings
import numpy as np, pandas as pd
from pathlib import Path

from rdkit import Chem
from rdkit.Chem import Descriptors
from rdkit.Chem import rdFingerprintGenerator
from rdkit import RDLogger; RDLogger.DisableLog("rdApp.*")

from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor

warnings.filterwarnings("ignore")
ROOT   = Path(__file__).resolve().parents[1]
DATA   = ROOT / "data"
MODELS = ROOT / "models"; MODELS.mkdir(exist_ok=True)
RANDOM_STATE = 42

_DESC = [name for name, _ in Descriptors._descList]
_calc = lambda m: [fn(m) for _, fn in Descriptors._descList]
_morgan = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)


def featurize(smiles_list):
    desc_rows, fp_rows, ok = [], [], []
    for s in smiles_list:
        m = Chem.MolFromSmiles(s)
        if m is None:
            ok.append(False); continue
        desc_rows.append(_calc(m))
        fp_rows.append(np.array(_morgan.GetFingerprint(m)))
        ok.append(True)
    X_desc = np.array(desc_rows, dtype=float)
    X_fp   = np.array(fp_rows,   dtype=float)
    return X_desc, X_fp, np.array(ok)


def metrics(y_obs_log, y_pred_log):
    res = y_pred_log - y_obs_log
    gmfe = 10 ** np.mean(np.abs(res))
    rmse = np.sqrt(np.mean(res**2))
    ss_res = np.sum(res**2); ss_tot = np.sum((y_obs_log - y_obs_log.mean())**2)
    r2 = 1 - ss_res/ss_tot
    fold = 10 ** np.abs(res)
    return dict(n=int(len(y_obs_log)), gmfe=round(float(gmfe),4), r2=round(float(r2),4),
                rmse=round(float(rmse),4),
                within_2fold=round(100*float(np.mean(fold<=2)),2),
                within_3fold=round(100*float(np.mean(fold<=3)),2))


def run():
    df = pd.read_csv(DATA / "permeability_clean.csv")
    tr = df[df.split=="train"].reset_index(drop=True)
    te = df[df.split=="test"].reset_index(drop=True)
    print(f"train {len(tr)}  test {len(te)}")

    print("Featurizing...")
    Xd_tr, Xf_tr, ok_tr = featurize(tr.smiles.tolist())
    Xd_te, Xf_te, ok_te = featurize(te.smiles.tolist())
    y_tr = tr.log10_papp.values[ok_tr]; y_te = te.log10_papp.values[ok_te]

    # Clean descriptors: replace inf, impute NaN with train medians
    Xd_tr[~np.isfinite(Xd_tr)] = np.nan; Xd_te[~np.isfinite(Xd_te)] = np.nan
    med = np.nanmedian(Xd_tr, axis=0); med[~np.isfinite(med)] = 0.0
    Xd_tr = np.where(np.isnan(Xd_tr), med, Xd_tr)
    Xd_te = np.where(np.isnan(Xd_te), med, Xd_te)
    # Cap extreme magnitudes (e.g. RDKit Ipc) that overflow float32
    Xd_tr = np.clip(Xd_tr, -1e12, 1e12)
    Xd_te = np.clip(Xd_te, -1e12, 1e12)

    X_tr = np.hstack([Xd_tr, Xf_tr]); X_te = np.hstack([Xd_te, Xf_te])
    print(f"features: {X_tr.shape[1]} ({Xd_tr.shape[1]} desc + {Xf_tr.shape[1]} FP)")

    results = {}

    print("\nRandom Forest...")
    rf = RandomForestRegressor(n_estimators=500, n_jobs=-1, random_state=RANDOM_STATE,
                               max_features="sqrt", min_samples_leaf=2)
    rf.fit(X_tr, y_tr)
    results["RF"] = metrics(y_te, rf.predict(X_te))
    pickle.dump(rf, open(MODELS/"rf_permeability.pkl","wb"))
    print("  ", results["RF"])

    print("\nXGBoost...")
    xgb = XGBRegressor(n_estimators=600, learning_rate=0.05, max_depth=6,
                       subsample=0.8, colsample_bytree=0.8, n_jobs=-1,
                       random_state=RANDOM_STATE)
    xgb.fit(X_tr, y_tr)
    results["XGB"] = metrics(y_te, xgb.predict(X_te))
    pickle.dump(xgb, open(MODELS/"xgb_permeability.pkl","wb"))
    print("  ", results["XGB"])

    json.dump(results, open(MODELS/"permeability_rf_xgb_results.json","w"), indent=2)
    print("\n=== PERMEABILITY BENCHMARK (scaffold test set) ===")
    for k,v in results.items():
        print(f"  {k}: R²={v['r2']:.3f}  GMFE={v['gmfe']:.3f}  "
              f"within-2fold={v['within_2fold']:.1f}%  within-3fold={v['within_3fold']:.1f}%")


if __name__ == "__main__":
    run()
