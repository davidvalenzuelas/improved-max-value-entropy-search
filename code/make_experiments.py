import os
import json
import copy

D = 6
NUM_EXPERIMENTS = 10
BO_ITERS = 50
ROOT = f"{D}d_experiments"

methods = {
    "mes": "MES",
    "mes_plus": "MES+",
}

with open("config.json", "r") as f:
    base_config = json.load(f)

for exp_id in range(1, NUM_EXPERIMENTS + 1):
    exp_dir = f"{ROOT}/exp_{exp_id}"
    os.makedirs(exp_dir, exist_ok=True)

    for folder_name, acquisition_name in methods.items():
        cfg = copy.deepcopy(base_config)

        cfg["random_seed"] = exp_id
        cfg["num_dims"] = D
        cfg["BO_iters"] = BO_ITERS
        cfg["num_repetitions"] = 1
        cfg["acquisitions"] = [acquisition_name]
        cfg["file_results"] = f"{exp_dir}/{folder_name}"
        cfg["experiment-name"] = f"synthetic_{D}d_exp_{exp_id}_{folder_name}"

        cfg["variables"] = {
            f"X{i}": {"type": "FLOAT", "size": 1, "min": 0, "max": 1}
            for i in range(1, D + 1)
        }

        os.makedirs(cfg["file_results"], exist_ok=True)

        with open(f"{exp_dir}/config_{folder_name}.json", "w") as f:
            json.dump(cfg, f, indent=4)