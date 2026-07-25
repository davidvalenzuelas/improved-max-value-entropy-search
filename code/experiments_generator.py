"""
Generates the configuration files for the experiments.
Both synthetic and benchmark problems are supported.

Author: David Valenzuela Sánchez
"""
import os
import json
import copy

PROBLEM_TYPE = "synthetic"
BENCHMARK = "michalewicz"
DIMENSIONS = [6]
START_EXP = 1
END_EXP = 100
BO_ITERS = 100
NUM_SAMPLES_SOLUTION = 4

methods = {
    "mes": "MES",
    "mes_plus": "MES+"
}

if PROBLEM_TYPE not in {"synthetic", "benchmark"}:
    raise ValueError(f"Unknown problem type: {PROBLEM_TYPE}")

with open("config.json", "r") as f:
    base_config = json.load(f)

for D in DIMENSIONS:
    if PROBLEM_TYPE == "synthetic":
        root = (f"experiments/synthetic_{D}d_samples/"
            f"samples_{NUM_SAMPLES_SOLUTION}")
        results_folder = "results_synthetic_problem"
        experiment_name = (f"synthetic_{D}d_samples_{NUM_SAMPLES_SOLUTION}")
    else:
        root = f"experiments/{BENCHMARK}"
        results_folder = "results_benchmark_problem"
        experiment_name = f"{BENCHMARK}{D}d"

    os.makedirs(root, exist_ok=True)

    for exp_id in range(START_EXP, END_EXP + 1):
        exp_dir = f"{root}/exp_{exp_id}"
        os.makedirs(exp_dir, exist_ok=True)

        for folder_name, acquisition_name in methods.items():
            results_dir = f"{exp_dir}/{folder_name}/{results_folder}"
            os.makedirs(results_dir, exist_ok=True)

            cfg = copy.deepcopy(base_config)
            cfg["problem_type"] = PROBLEM_TYPE
            cfg["random_seed"] = exp_id
            cfg["num_dims"] = D
            cfg["BO_iters"] = BO_ITERS
            cfg["num_initial_obs"] = 10
            cfg["num_samples_solution"] = NUM_SAMPLES_SOLUTION
            cfg["M_extra"] = 100
            cfg["num_repetitions"] = 1
            cfg["acquisitions"] = [acquisition_name]
            cfg["acquisition"] = acquisition_name
            cfg["file_results"] = results_dir
            cfg["experiment-name"] = f"{experiment_name}_exp_{exp_id}_{folder_name}"

            if PROBLEM_TYPE == "synthetic":
                cfg.pop("benchmark", None)
            else:
                cfg["benchmark"] = BENCHMARK
                cfg.pop("lenghtscale_model_synthetic_problem", None)

            cfg["variables"] = {
                f"X{i}": {"type": "FLOAT", "size": 1, "min": 0, "max": 1}
                for i in range(1, D + 1)
            }

            with open(f"{exp_dir}/config_{folder_name}.json", "w") as f:
                json.dump(cfg, f, indent=4)

print("Configs generated.")