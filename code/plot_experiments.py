import os
import numpy as np
import matplotlib.pyplot as plt

ROOT = "6d_experiments"
NUM_EXPERIMENTS = 10
BO_ITERS = 50

methods = [
    ("mes_plus", "MES+"),
    ("mes", "MES"),
]

os.makedirs(f"{ROOT}/summary", exist_ok=True)

iterations = np.arange(BO_ITERS)

plt.figure(figsize=(9, 6))

for folder_name, label in methods:
    runs = []

    for exp_id in range(1, NUM_EXPERIMENTS + 1):
        path = f"{ROOT}/exp_{exp_id}/{folder_name}/metric_post_mean.txt"

        if not os.path.exists(path):
            print(f"Missing file: {path}")
            continue

        vals = np.loadtxt(path)
        vals = np.atleast_1d(vals)

        if vals.shape[0] < BO_ITERS:
            print(f"Incomplete file: {path}, only {vals.shape[0]} values")
            continue

        runs.append(vals[:BO_ITERS])

    runs = np.vstack(runs)

    mean = runs.mean(axis=0)
    sem = runs.std(axis=0, ddof=1) / np.sqrt(runs.shape[0])

    plt.plot(iterations, mean, label=label)
    plt.fill_between(iterations, mean - sem, mean + sem, alpha=0.15)

plt.xlabel("BO iteration")
plt.ylabel("log(abs(y_recom - y_opt) / abs(y_opt) + 1e-6)")
plt.legend()
plt.tight_layout()
plt.savefig(f"{ROOT}/summary/bo_metric_mean_sem.png", dpi=200)
plt.show()