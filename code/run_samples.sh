#!/usr/bin/env bash
#SBATCH -A ada2_serv
#SBATCH -p cccmd
#SBATCH --array=1-100
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=5-00:00:00
#SBATCH --output=logs/%x-%A-%a.out
#SBATCH --error=logs/%x-%A-%a.err

set -euo pipefail

SAMPLES="${1:?Uso: sbatch samples.sh <1|2|4|8>}"

module load python/3.10.15
source ~/venvs/mesplus/bin/activate

cd ~/mesplus

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export TORCH_NUM_THREADS=1
export PYTHONUNBUFFERED=1

ROOT="experiments/synthetic_4d_samples/samples_${SAMPLES}"
EXP_ID="${SLURM_ARRAY_TASK_ID}"

is_complete() {
    local results="$1"

    [ -f "$results/objective_at_recommendations_post_mean.txt" ] \
        && [ -f "$results/objective_at_recommendations_obs_obs.txt" ] \
        && [ -f "$results/points_evaluated.txt" ] \
        && [ -f "$results/y_values_evaluated.txt" ] \
        && [ "$(wc -l < "$results/objective_at_recommendations_post_mean.txt")" -eq 100 ] \
        && [ "$(wc -l < "$results/objective_at_recommendations_obs_obs.txt")" -eq 100 ] \
        && [ "$(wc -l < "$results/points_evaluated.txt")" -eq 110 ] \
        && [ "$(wc -l < "$results/y_values_evaluated.txt")" -eq 110 ]
}

for METHOD in mes mes_plus; do
    CONFIG="${ROOT}/exp_${EXP_ID}/config_${METHOD}.json"

    if [ ! -f "$CONFIG" ]; then
        echo "No existe: $CONFIG"
        exit 1
    fi

    RESULTS=$(python -c \
        'import json,sys; print(json.load(open(sys.argv[1]))["file_results"])' \
        "$CONFIG")

    mkdir -p "$RESULTS"

    if is_complete "$RESULTS"; then
        echo "Ya estaba completo: $CONFIG"
        continue
    fi

    if find "$RESULTS" -maxdepth 1 -type f -size +0c -print -quit \
        | grep -q .; then
        echo "Hay resultados parciales en:"
        echo "$RESULTS"
        echo "No se sobrescribe ni se borra nada."
        exit 2
    fi

    echo "========================================"
    echo "Samples: $SAMPLES"
    echo "Experimento: $EXP_ID"
    echo "Método: $METHOD"
    echo "========================================"

    python my_loop_BO.py "$CONFIG"

    if ! is_complete "$RESULTS"; then
        echo "La ejecución ha terminado incompleta: $CONFIG"
        exit 3
    fi

    echo "Completado: $CONFIG"
done

echo "Experimento ${EXP_ID} terminado con ${SAMPLES} samples."