"""
train_optuna_rf_xgb.py
======================
Optuna-tuned RF + XGB for log10 Caco-2 Papp (scaffold-split benchmark).
Self-contained featurization (RDKit 2D descriptors + Morgan FP).

Tuning: TPE sampler, 5-fold CV on the TRAIN split, minimize log-RMSE.
Then refit best on full train, evaluate on the held-out scaffold test set.

Outputs (models/):
  - rf_permeability_tuned.pkl, xgb_permeability_tuned.pkl
  - permeability_tuned_results.json   (RF + XGB test metrics + best params)

Run (Longleaf):
    python scripts/train_optuna_rf_xgb.py
"""
import json, pickle, warnings
import numpy as np, pandas as pd
from pathlib import Path

from rdkit import Chem
from rdkit.Chem import Descriptors
from rdkit.Chem import rdFingerprintGenerator
from rdkit import RDLogger; RDLogger.DisableLog("rdApp.*")

import optuna
from optuna.samplers import TPESampler
from optuna.pruners import MedianPruner
from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor

warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)

ROOT   = Path(__file__).resolve().parents[1]
DATA   = ROOT / "data"
MODELS = ROOT / "models"; MODELS.mkdir(exist_ok=True)
RANDOM_STATE = 42
N_TRIALS = 200
N_FOLDS  = 5

_morgan = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
def _desc(m): return [fn(m) for _, fn in Descriptors._descList]

def featurize(smiles):
    d, f, ok = [], [], []
    for s in smiles:
        m = Chem.MolFromSmiles(s)
        if m is None: ok.append(False); continue
        d.append(_desc(m)); f.append(np.array(_morgan.GetFingerprint(m))); ok.append(True)
    return np.array(d, float), np.array(f, float), np.array(ok)

def clean_desc(Xtr, Xte):
    Xtr[~np.isfinite(Xtr)] = np.nan; Xte[~np.isfinite(Xte)] = np.nan
    med = np.nanmedian(Xtr, axis=0); med[~np.isfinite(med)] = 0.0
    Xtr = np.where(np.isnan(Xtr), med, Xtr); Xte = np.where(np.isnan(Xte), med, Xte)
    return np.clip(Xtr, -1e12, 1e12), np.clip(Xte, -1e12, 1e12)

def metrics(yo, yp):
    r = yp - yo; fold = 10**np.abs(r)
    return dict(n=int(len(yo)), gmfe=round(float(10**np.mean(np.abs(r))),4),
                r2=round(float(1-np.sum(r**2)/np.sum((yo-yo.mean())**2)),4),
                rmse=round(float(np.sqrt(np.mean(r**2))),4),
                within_2fold=round(100*float(np.mean(fold<=2)),2),
                within_3fold=round(100*float(np.mean(fold<=3)),2))

def cv_rmse(make_model, X, y):
    kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    sc = []
    for tr, va in kf.split(X):
        mdl = make_model(); mdl.fit(X[tr], y[tr])
        sc.append(np.sqrt(np.mean((mdl.predict(X[va]) - y[va])**2)))
    return float(np.mean(sc))


def run():
    df = pd.read_csv(DATA / "permeability_clean.csv")
    tr, te = df[df.split=="train"], df[df.split=="test"]
    Xd_tr, Xf_tr, ok_tr = featurize(tr.smiles.tolist())
    Xd_te, Xf_te, ok_te = featurize(te.smiles.tolist())
    Xd_tr, Xd_te = clean_desc(Xd_tr, Xd_te)
    X_tr = np.hstack([Xd_tr, Xf_tr]); X_te = np.hstack([Xd_te, Xf_te])
    y_tr = tr.log10_papp.values[ok_tr]; y_te = te.log10_papp.values[ok_te]
    print(f"train {X_tr.shape}  test {X_te.shape}")

    results = {}

    # ── RF ───────────────────────────────────────────────────────────────────
    def rf_obj(t):
        mk = lambda: RandomForestRegressor(
            n_estimators=t.suggest_int("n_estimators",200,1000,step=100),
            max_depth=t.suggest_int("max_depth",5,40),
            min_samples_leaf=t.suggest_int("min_samples_leaf",1,20),
            min_samples_split=t.suggest_int("min_samples_split",2,20),
            max_features=t.suggest_categorical("max_features",["sqrt","log2",0.3,0.5]),
            n_jobs=-1, random_state=RANDOM_STATE)
        return cv_rmse(mk, X_tr, y_tr)
    print("\nTuning RF...")
    s_rf = optuna.create_study(direction="minimize", sampler=TPESampler(seed=RANDOM_STATE))
    s_rf.optimize(rf_obj, n_trials=N_TRIALS, show_progress_bar=True)
    rf = RandomForestRegressor(**s_rf.best_params, n_jobs=-1, random_state=RANDOM_STATE)
    rf.fit(X_tr, y_tr)
    results["RF"] = metrics(y_te, rf.predict(X_te)); results["RF"]["best_params"] = s_rf.best_params
    pickle.dump(rf, open(MODELS/"rf_permeability_tuned.pkl","wb"))
    print("  RF:", {k:results["RF"][k] for k in ["r2","gmfe","within_2fold"]})

    # ── XGB ──────────────────────────────────────────────────────────────────
    def xgb_obj(t):
        mk = lambda: XGBRegressor(
            n_estimators=t.suggest_int("n_estimators",200,1500,step=100),
            learning_rate=t.suggest_float("learning_rate",0.005,0.3,log=True),
            max_depth=t.suggest_int("max_depth",3,12),
            min_child_weight=t.suggest_int("min_child_weight",1,20),
            subsample=t.suggest_float("subsample",0.5,1.0),
            colsample_bytree=t.suggest_float("colsample_bytree",0.3,1.0),
            gamma=t.suggest_float("gamma",0.0,5.0),
            reg_alpha=t.suggest_float("reg_alpha",1e-8,10.0,log=True),
            reg_lambda=t.suggest_float("reg_lambda",1e-8,10.0,log=True),
            n_jobs=-1, random_state=RANDOM_STATE)
        return cv_rmse(mk, X_tr, y_tr)
    print("\nTuning XGB...")
    s_xgb = optuna.create_study(direction="minimize", sampler=TPESampler(seed=RANDOM_STATE))
    s_xgb.optimize(xgb_obj, n_trials=N_TRIALS, show_progress_bar=True)
    xgb = XGBRegressor(**s_xgb.best_params, n_jobs=-1, random_state=RANDOM_STATE)
    xgb.fit(X_tr, y_tr)
    results["XGB"] = metrics(y_te, xgb.predict(X_te)); results["XGB"]["best_params"] = s_xgb.best_params
    pickle.dump(xgb, open(MODELS/"xgb_permeability_tuned.pkl","wb"))
    print("  XGB:", {k:results["XGB"][k] for k in ["r2","gmfe","within_2fold"]})

    json.dump(results, open(MODELS/"permeability_tuned_results.json","w"), indent=2)
    print("\n=== TUNED PERMEABILITY (scaffold test) ===")
    for k in ["RF","XGB"]:
        v=results[k]; print(f"  {k}: R²={v['r2']:.3f} GMFE={v['gmfe']:.3f} W2={v['within_2fold']:.1f}% W3={v['within_3fold']:.1f}%")


if __name__ == "__main__":
    run()
