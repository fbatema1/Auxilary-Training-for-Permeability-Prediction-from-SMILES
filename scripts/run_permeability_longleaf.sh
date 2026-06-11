#!/bin/bash
#SBATCH --job-name=perm_aux
#SBATCH --partition=volta-gpu
#SBATCH --qos=gpu_access
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --gres=gpu:tesla_v100-sxm2-16gb:1
#SBATCH --time=12:00:00
#SBATCH --output=/nas/longleaf/home/fbatema1/pk-predictor/auxiliary_training/permeability/logs/perm_%j.out
#SBATCH --error=/nas/longleaf/home/fbatema1/pk-predictor/auxiliary_training/permeability/logs/perm_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=fbatema1@unc.edu

# =============================================================================
# Permeability auxiliary models — Optuna-tuned RF/XGB (CPU) + GNN (GPU).
# Scaffold-split benchmark on log10 Caco-2 Papp.
# =============================================================================
set -e
PYTHON=/nas/longleaf/home/fbatema1/.conda/envs/pkip-env/bin/python
cd /nas/longleaf/home/fbatema1/pk-predictor/auxiliary_training/permeability
mkdir -p logs models

echo "=== Permeability auxiliary training ==="
echo "Started: $(date)  Node: $(hostname)"
echo "Python:  $($PYTHON --version)"

echo ""
echo "[1/2] Optuna RF + XGB..."
$PYTHON scripts/train_optuna_rf_xgb.py

echo ""
echo "[2/2] Optuna GNN (AttentiveFP, GPU)..."
$PYTHON scripts/train_gnn.py

echo ""
echo "=== Done: $(date) ==="
echo "Results: models/permeability_tuned_results.json + permeability_gnn_results.json"
echo "Untuned baseline was: XGB R2=0.30 GMFE=3.35 / RF R2=0.28 GMFE=3.57"
