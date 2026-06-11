"""
train_gnn.py
============
Optuna-tuned GNN (AttentiveFP) for log10 Caco-2 Papp, scaffold-split benchmark.

Reuses the main Wadhams project's graph builder (features/graph_builder.py) for
consistent atom/bond featurization, and PyG's built-in AttentiveFP with a single
regression output. Run on Longleaf (GPU).

Outputs (models/):
  - gnn_permeability.pt
  - permeability_gnn_results.json

Run (Longleaf, GPU):
    python scripts/train_gnn.py
"""
import json, warnings
import numpy as np, pandas as pd
from pathlib import Path
import sys

import torch
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
from torch_geometric.nn.models import AttentiveFP
import optuna
from optuna.samplers import TPESampler

warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)

ROOT      = Path(__file__).resolve().parents[1]            # .../permeability
PKROOT    = Path(__file__).resolve().parents[3]            # .../pk-predictor
DATA      = ROOT / "data"
MODELS    = ROOT / "models"; MODELS.mkdir(exist_ok=True)
sys.path.insert(0, str(PKROOT))
from features.graph_builder import MolGraphBuilder

RANDOM_STATE = 42
N_TRIALS = 60
MAX_EPOCHS = 120
PATIENCE = 15
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(RANDOM_STATE)


def build_graphs(df):
    b = MolGraphBuilder()
    graphs = []
    for smi, y in zip(df.smiles, df.log10_papp):
        g = b.smiles_to_graph(smi)
        if g is not None:
            g.y = torch.tensor([y], dtype=torch.float32)
            graphs.append(g)
    return graphs


def metrics(yo, yp):
    r = yp - yo; fold = 10**np.abs(r)
    return dict(n=int(len(yo)), gmfe=round(float(10**np.mean(np.abs(r))),4),
                r2=round(float(1-np.sum(r**2)/np.sum((yo-yo.mean())**2)),4),
                rmse=round(float(np.sqrt(np.mean(r**2))),4),
                within_2fold=round(100*float(np.mean(fold<=2)),2),
                within_3fold=round(100*float(np.mean(fold<=3)),2))


def make_model(in_ch, edge_dim, hp):
    return AttentiveFP(in_channels=in_ch, hidden_channels=hp["hidden"],
                       out_channels=1, edge_dim=edge_dim,
                       num_layers=hp["layers"], num_timesteps=hp["timesteps"],
                       dropout=hp["dropout"]).to(DEVICE)


def train_eval(model, tr_loader, va_loader, lr, max_epochs=MAX_EPOCHS, patience=PATIENCE):
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    best, best_state, wait = np.inf, None, 0
    for ep in range(max_epochs):
        model.train()
        for b in tr_loader:
            b = b.to(DEVICE); opt.zero_grad()
            out = model(b.x, b.edge_index, b.edge_attr, b.batch)
            loss = F.mse_loss(out.view(-1), b.y.view(-1))
            loss.backward(); opt.step()
        # val
        model.eval(); vs = []
        with torch.no_grad():
            for b in va_loader:
                b = b.to(DEVICE)
                out = model(b.x, b.edge_index, b.edge_attr, b.batch)
                vs.append(F.mse_loss(out.view(-1), b.y.view(-1), reduction="sum").item())
        v = np.sqrt(sum(vs)/len(va_loader.dataset))
        if v < best - 1e-4: best, best_state, wait = v, {k:t.cpu().clone() for k,t in model.state_dict().items()}, 0
        else:
            wait += 1
            if wait >= patience: break
    if best_state: model.load_state_dict(best_state)
    return model, best


def predict(model, loader):
    model.eval(); preds, ys = [], []
    with torch.no_grad():
        for b in loader:
            b = b.to(DEVICE)
            preds.append(model(b.x, b.edge_index, b.edge_attr, b.batch).view(-1).cpu().numpy())
            ys.append(b.y.view(-1).cpu().numpy())
    return np.concatenate(ys), np.concatenate(preds)


def run():
    df = pd.read_csv(DATA / "permeability_clean.csv")
    tr_df, te_df = df[df.split=="train"], df[df.split=="test"]
    print(f"Building graphs... (device={DEVICE})")
    tr_graphs = build_graphs(tr_df); te_graphs = build_graphs(te_df)
    print(f"  train {len(tr_graphs)}  test {len(te_graphs)}")
    in_ch = tr_graphs[0].x.shape[1]; edge_dim = tr_graphs[0].edge_attr.shape[1]

    # train/val split for tuning
    rng = np.random.default_rng(RANDOM_STATE)
    idx = rng.permutation(len(tr_graphs)); n_val = int(0.15*len(idx))
    val_g = [tr_graphs[i] for i in idx[:n_val]]; fit_g = [tr_graphs[i] for i in idx[n_val:]]

    def obj(t):
        hp = dict(hidden=t.suggest_categorical("hidden",[64,128,256]),
                  layers=t.suggest_int("layers",2,5),
                  timesteps=t.suggest_int("timesteps",1,3),
                  dropout=t.suggest_float("dropout",0.0,0.4))
        lr = t.suggest_float("lr",1e-4,1e-2,log=True)
        bs = t.suggest_categorical("batch",[32,64,128])
        m = make_model(in_ch, edge_dim, hp)
        _, v = train_eval(m, DataLoader(fit_g,batch_size=bs,shuffle=True),
                          DataLoader(val_g,batch_size=128), lr)
        return v

    print(f"\nTuning GNN ({N_TRIALS} trials)...")
    study = optuna.create_study(direction="minimize", sampler=TPESampler(seed=RANDOM_STATE))
    study.optimize(obj, n_trials=N_TRIALS, show_progress_bar=True)
    best = study.best_params
    print("  best:", best)

    # refit best on full train (with internal val for early stop), eval on test
    hp = {k:best[k] for k in ["hidden","layers","timesteps","dropout"]}
    model = make_model(in_ch, edge_dim, hp)
    model, _ = train_eval(model, DataLoader(fit_g,batch_size=best["batch"],shuffle=True),
                          DataLoader(val_g,batch_size=128), best["lr"])
    yo, yp = predict(model, DataLoader(te_graphs, batch_size=128))
    res = metrics(yo, yp); res["best_params"] = best
    torch.save({"state_dict":model.state_dict(),"hp":hp,"in_ch":in_ch,"edge_dim":edge_dim},
               MODELS/"gnn_permeability.pt")
    json.dump(res, open(MODELS/"permeability_gnn_results.json","w"), indent=2)
    print("\n=== TUNED GNN PERMEABILITY (scaffold test) ===")
    print(f"  R²={res['r2']:.3f} GMFE={res['gmfe']:.3f} W2={res['within_2fold']:.1f}% W3={res['within_3fold']:.1f}%")


if __name__ == "__main__":
    run()
