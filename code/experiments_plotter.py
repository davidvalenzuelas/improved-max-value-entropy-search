import os
import numpy as np
import matplotlib.pyplot as plt

DIMENSIONS = [4, 6]
NUM_EXPERIMENTS = 10
BO_ITERS = 20
N_BOOTSTRAP = 200
ROOT_TEMPLATE = "{D}d_experiments"
RNG_SEED = 12345

methods = [
    ("mes", "MES"),
    ("mes_plus", "MES+"),
    ("random", "Random"),
]

reference_files = [
    "objective_at_recommendations_obs.txt",
    "objective_at_recommendations_post_mean.txt",
    "objective_at_recommendations_obs_obs.txt",
    "y_optimum_problem.txt",
    "y_values_evaluated.txt",
]

plot_targets = [
    ("post_mean", "objective_at_recommendations_post_mean.txt", "Post. Mean Recommendation"),
    ("obs", "objective_at_recommendations_obs_obs.txt", "Best Observed Recommendation"),
]


def candidate_paths(root, exp_id, folder_name, filename):
    base = f"{root}/exp_{exp_id}/{folder_name}"
    return [
        f"{base}/{filename}",
        f"{base}/results_synthetic_problem/{filename}",
    ]


def load_values(root, exp_id, folder_name, filename):
    for path in candidate_paths(root, exp_id, folder_name, filename):
        if os.path.exists(path):
            values = np.loadtxt(path)
            return np.atleast_1d(values).astype(float).reshape(-1)
    return None


def get_reference_value(root, exp_id):
    value_ref = -np.inf

    for folder_name, _ in methods:
        for filename in reference_files:
            values = load_values(root, exp_id, folder_name, filename)
            if values is None or values.size == 0:
                continue
            values = values[np.isfinite(values)]
            if values.size > 0:
                value_ref = max(value_ref, np.max(values))

    if not np.isfinite(value_ref):
        return None

    return value_ref


def compute_metric(values, value_ref):
    values = values[:BO_ITERS]
    return np.log(np.maximum(0.0, np.abs(value_ref - values)) / np.abs(value_ref + 1.0) + 1e-6)


def bootstrap_std_of_mean(runs, n_bootstrap=N_BOOTSTRAP, seed=RNG_SEED):
    rng = np.random.default_rng(seed)
    n_runs = runs.shape[0]
    bootstrap_means = np.zeros((n_bootstrap, runs.shape[1]))

    for b in range(n_bootstrap):
        idx = rng.choice(n_runs, size=n_runs, replace=True)
        bootstrap_means[b] = runs[idx].mean(axis=0)

    return bootstrap_means.std(axis=0, ddof=1)


def load_runs(root, target_file):
    runs_by_method = {label: [] for _, label in methods}
    valid_experiments = []

    for exp_id in range(1, NUM_EXPERIMENTS + 1):
        value_ref = get_reference_value(root, exp_id)
        if value_ref is None:
            print(f"Skipping exp_{exp_id}: no valid reference value")
            continue

        current_runs = {}
        correct = True

        for folder_name, label in methods:
            values = load_values(root, exp_id, folder_name, target_file)

            if values is None:
                print(f"Skipping exp_{exp_id}: missing {folder_name}/{target_file}")
                correct = False
                break

            if values.shape[0] < BO_ITERS:
                print(f"Skipping exp_{exp_id}: incomplete {folder_name}/{target_file} ({values.shape[0]} values)")
                correct = False
                break

            current_runs[label] = compute_metric(values, value_ref)

        if correct:
            for label in current_runs:
                runs_by_method[label].append(current_runs[label])
            valid_experiments.append(exp_id)

    for label in runs_by_method:
        if len(runs_by_method[label]) > 0:
            runs_by_method[label] = np.vstack(runs_by_method[label])
        else:
            runs_by_method[label] = np.empty((0, BO_ITERS))

    return runs_by_method, valid_experiments


def make_plot(root, D, target_name, target_file, title_suffix):
    os.makedirs(f"{root}/summary", exist_ok=True)

    runs_by_method, valid_experiments = load_runs(root, target_file)
    print(f"{D}D - {target_name}: using {len(valid_experiments)} valid experiments")

    if len(valid_experiments) == 0:
        print(f"No valid experiments for {root} / {target_name}")
        return

    iterations = np.arange(1, BO_ITERS + 1)

    plt.figure(figsize=(9, 5))

    for _, label in methods:
        runs = runs_by_method[label]
        if runs.shape[0] == 0:
            continue

        mean = runs.mean(axis=0)
        sd_mean = bootstrap_std_of_mean(runs)

        plt.plot(iterations, mean, label=label)
        plt.fill_between(iterations, mean - sd_mean, mean + sd_mean, alpha=0.15)

    plt.title(f"{D}D. Noiseless. {title_suffix}")
    plt.xlabel("Number of Function Evaluations")
    plt.ylabel("Log. Rel. Diff. w.r.t Max")
    plt.legend()
    plt.tight_layout()

    plt.savefig(f"{root}/summary/plot_{target_name}.png", dpi=200)
    plt.savefig(f"{root}/summary/plot_{target_name}.pdf")
    plt.close()


def main():
    for D in DIMENSIONS:
        root = ROOT_TEMPLATE.format(D=D)
        for target_name, target_file, title_suffix in plot_targets:
            make_plot(root, D, target_name, target_file, title_suffix)


if __name__ == "__main__":
    main()
