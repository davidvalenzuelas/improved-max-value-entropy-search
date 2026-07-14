#!/usr/bin/env bash
set -euo pipefail

ROOT="${1:-experiments/shekel}"
START_EXP="${2:-1}"
END_EXP="${3:-100}"
N_JOBS="${4:-4}"

PYTHON_BIN="${PYTHON_BIN:-$(command -v python)}"
LOOP_SCRIPT="${LOOP_SCRIPT:-my_loop_BO.py}"

METHODS=(mes mes_plus random)

if [ ! -f "$LOOP_SCRIPT" ]; then
    echo "Missing loop script: $LOOP_SCRIPT"
    exit 1
fi

if [ ! -d "$ROOT" ]; then
    echo "Missing experiments root: $ROOT"
    exit 1
fi

echo "Using Python: $PYTHON_BIN"
"$PYTHON_BIN" -c "import sys; print(sys.executable); print(sys.version)"
echo "Experiments root: $ROOT"
echo "Experiments: $START_EXP-$END_EXP"
echo "Parallel jobs: $N_JOBS"

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export TORCH_NUM_THREADS=1
export PYTHON_BIN
export LOOP_SCRIPT

CONFIG_LIST="$(mktemp)"
trap 'rm -f "$CONFIG_LIST"' EXIT

for EXP_ID in $(seq "$START_EXP" "$END_EXP"); do
    for METHOD in "${METHODS[@]}"; do
        CFG="$ROOT/exp_${EXP_ID}/config_${METHOD}.json"

        if [ ! -f "$CFG" ]; then
            echo "Missing config: $CFG"
            exit 1
        fi

        echo "$CFG" >> "$CONFIG_LIST"
    done
done

echo "Total configs: $(wc -l < "$CONFIG_LIST")"

xargs -a "$CONFIG_LIST" -P "$N_JOBS" -I {} bash -c '
    set -euo pipefail

    cfg="$1"

    file_results=$("$PYTHON_BIN" -c "import json, sys; cfg=json.load(open(sys.argv[1])); print(cfg[\"file_results\"])" "$cfg")
    bo_iters=$("$PYTHON_BIN" -c "import json, sys; cfg=json.load(open(sys.argv[1])); print(int(cfg[\"BO_iters\"]))" "$cfg")
    num_initial_obs=$("$PYTHON_BIN" -c "import json, sys; cfg=json.load(open(sys.argv[1])); print(int(cfg[\"num_initial_obs\"]))" "$cfg")

    post_mean_file="$file_results/objective_at_recommendations_post_mean.txt"
    obs_obs_file="$file_results/objective_at_recommendations_obs_obs.txt"
    points_file="$file_results/points_evaluated.txt"
    y_values_file="$file_results/y_values_evaluated.txt"

    expected_points=$((num_initial_obs + bo_iters))

    if [ -f "$post_mean_file" ] && [ -f "$obs_obs_file" ] \
        && [ -f "$points_file" ] && [ -f "$y_values_file" ] \
        && [ "$(wc -l < "$post_mean_file")" -eq "$bo_iters" ] \
        && [ "$(wc -l < "$obs_obs_file")" -eq "$bo_iters" ] \
        && [ "$(wc -l < "$points_file")" -eq "$expected_points" ] \
        && [ "$(wc -l < "$y_values_file")" -eq "$expected_points" ]; then
        echo "Skipping completed: $cfg"
        exit 0
    fi

    rm -rf "$file_results"
    mkdir -p "$file_results"

    echo "Starting: $cfg"
    "$PYTHON_BIN" "$LOOP_SCRIPT" "$cfg"

    for file in "$post_mean_file" "$obs_obs_file" "$points_file" "$y_values_file"; do
        if [ ! -f "$file" ]; then
            echo "Missing output after run: $file"
            exit 1
        fi
    done

    n_post_mean=$(wc -l < "$post_mean_file")
    n_obs_obs=$(wc -l < "$obs_obs_file")
    n_points=$(wc -l < "$points_file")
    n_y_values=$(wc -l < "$y_values_file")

    if [ "$n_post_mean" -ne "$bo_iters" ]; then
        echo "Incomplete post-mean file: $cfg has $n_post_mean values, expected $bo_iters"
        exit 1
    fi

    if [ "$n_obs_obs" -ne "$bo_iters" ]; then
        echo "Incomplete best-observed file: $cfg has $n_obs_obs values, expected $bo_iters"
        exit 1
    fi

    if [ "$n_points" -ne "$expected_points" ]; then
        echo "Incomplete points file: $cfg has $n_points points, expected $expected_points"
        exit 1
    fi

    if [ "$n_y_values" -ne "$expected_points" ]; then
        echo "Incomplete values file: $cfg has $n_y_values values, expected $expected_points"
        exit 1
    fi

    echo "Finished: $cfg"
' _ {}

echo "All requested experiments finished."