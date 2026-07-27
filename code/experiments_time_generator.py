"""
Generates the configuration files for the synthetic 4D experiments,
when we measure the time taken by each method.

Author: David Valenzuela Sánchez
"""
import copy
import json
import os


ROOT = "experiments/time4d"

START_EXP = 1
END_EXP = 10

# Uses the 4d synthetic problem with 1 optimum sample
NUM_DIMS = 4
NUM_INITIAL_OBS = 10
BO_ITERS = 100
NUM_SAMPLES = 1
M_EXTRA = 100

METHODS = {
    "random": "RANDOM",
    "mes": "MES",
    "mes_plus": "MES+",
}


with open("config.json", "r") as file:
    base_config = json.load(file)


for exp_id in range(START_EXP, END_EXP + 1):
    exp_dir = f"{ROOT}/exp_{exp_id}"
    os.makedirs(exp_dir, exist_ok=True)

    for folder_name, acquisition_name in METHODS.items():
        results_dir = (
            f"{exp_dir}/{folder_name}/"
            "results_synthetic_problem"
        )

        os.makedirs(results_dir, exist_ok=True)

        config = copy.deepcopy(base_config)

        config["problem_type"] = "synthetic"
        config["random_seed"] = exp_id
        config["num_dims"] = NUM_DIMS
        config["num_initial_obs"] = NUM_INITIAL_OBS
        config["BO_iters"] = BO_ITERS
        config["num_samples_solution"] = NUM_SAMPLES
        config["M_extra"] = M_EXTRA
        config["num_repetitions"] = 1
        # We want to measure times
        config["measure_time"] = True
        config["acquisitions"] = [acquisition_name]

        config["file_results"] = results_dir
        config["experiment-name"] = (
            f"time4d_exp_{exp_id}_{folder_name}"
        )

        config.pop("benchmark", None)
        config.pop("dataset", None)

        config["variables"] = {
            f"X{i}": {
                "type": "FLOAT",
                "size": 1,
                "min": 0,
                "max": 1,
            }
            for i in range(1, NUM_DIMS + 1)
        }

        config_path = (
            f"{exp_dir}/config_{folder_name}.json"
        )

        with open(config_path, "w") as file:
            json.dump(
                config,
                file,
                indent=4,
            )


print("Timing configurations generated!")
