import os
import json
import copy

DIMENSIONS = [4, 6]
START_EXP = 1
END_EXP = 10
BO_ITERS = 20
ROOT_TEMPLATE = "{D}d_experiments"

methods = {
    "mes": "MES",
    "mes_plus": "MES+",
    "random": "RANDOM",
}

with open("config.json", "r") as f:
    base_config = json.load(f)

for D in DIMENSIONS:
    root = ROOT_TEMPLATE.format(D=D)

    for exp_id in range(START_EXP, END_EXP + 1):
        exp_dir = f"{root}/exp_{exp_id}"
        os.makedirs(exp_dir, exist_ok=True)

        for folder_name, acquisition_name in methods.items():
            cfg = copy.deepcopy(base_config)

            cfg["random_seed"] = exp_id
            cfg["num_dims"] = D
            cfg["BO_iters"] = BO_ITERS
            cfg["num_repetitions"] = 1
            cfg["acquisitions"] = [acquisition_name]
            cfg["acquisition"] = acquisition_name
            cfg["file_results"] = f"{exp_dir}/{folder_name}"
            cfg["experiment-name"] = f"synthetic_{D}d_exp_{exp_id}_{folder_name}"

            cfg["variables"] = {
                f"X{i}": {"type": "FLOAT", "size": 1, "min": 0, "max": 1}
                for i in range(1, D + 1)
            }

            os.makedirs(cfg["file_results"], exist_ok=True)

            with open(f"{exp_dir}/config_{folder_name}.json", "w") as f:
                json.dump(cfg, f, indent=4)

print("Configs generated.")
