#!/usr/bin/env python3
# coding: utf-8
"""Generates configuration files for real MLP experiments."""

import copy
import json
import os


DATASETS = ["australian", "ionosphere"]

START_EXP = 1
END_EXP = 1

methods = {
    "mes": "MES",
    "mes_plus": "MES+",
    "random": "RANDOM",
}

with open("config_mlp.json", "r") as f:
    base_config = json.load(f)

for dataset in DATASETS:
    root = f"experiments/real/{dataset}"
    os.makedirs(root, exist_ok=True)

    for exp_id in range(START_EXP, END_EXP + 1):
        exp_dir = f"{root}/exp_{exp_id}"
        os.makedirs(exp_dir, exist_ok=True)

        for folder_name, acquisition_name in methods.items():
            results_dir = f"{exp_dir}/{folder_name}/results_real_problem"
            os.makedirs(results_dir, exist_ok=True)

            cfg = copy.deepcopy(base_config)
            cfg["problem_type"] = "real"
            cfg["dataset"] = dataset
            cfg["random_seed"] = exp_id
            cfg["num_dims"] = 5
            cfg["num_repetitions"] = 1
            cfg["acquisitions"] = [acquisition_name]
            cfg["acquisition"] = acquisition_name
            cfg["file_results"] = results_dir
            cfg["experiment-name"] = f"{dataset}_exp_{exp_id}_{folder_name}"

            with open(f"{exp_dir}/config_{folder_name}.json", "w") as f:
                json.dump(cfg, f, indent=4)

print("Real experiment configs generated.")