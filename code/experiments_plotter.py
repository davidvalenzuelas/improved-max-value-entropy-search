import os
import numpy as np
import matplotlib.pyplot as plt

DIMENSIONS = [4]
NUM_EXPERIMENTS = 100
BO_ITERS = 100
N_BOOTSTRAP = 200
ROOT_TEMPLATE = "{D}d_experiments"

method_names = ["MES", "MES+", "Random"]
method_folders = ["mes", "mes_plus", "random"]
method_colors = ["green", "pink", "blue"]

reference_files = [
    "objective_at_recommendations_obs.txt",
    "objective_at_recommendations_post_mean.txt",
    "objective_at_recommendations_obs_obs.txt",
    "y_optimum_problem.txt",
]

plot_targets = [
    ("post_mean", "objective_at_recommendations_post_mean.txt"),
    ("obs", "objective_at_recommendations_obs_obs.txt"),
]


def load_vector(path):
    values = np.loadtxt(path)
    return np.atleast_1d(values).astype(float).reshape(-1)


def result_path(root, exp_id, method_folder, filename):
    return f"{root}/exp_{exp_id}/{method_folder}/results_synthetic_problem/{filename}"


def load_results(root, target_file):
    results = [np.zeros((NUM_EXPERIMENTS, BO_ITERS)) for _ in method_names]
    counter = 0

    for exp_id in range(1, NUM_EXPERIMENTS + 1):
        correct = True
        value_solution = -np.inf

        for method_index, method_folder in enumerate(method_folders):
            path = result_path(root, exp_id, method_folder, target_file)

            if not os.path.exists(path):
                correct = False
                break

            current_results = load_vector(path)[:BO_ITERS]

            if len(current_results) != BO_ITERS:
                correct = False
                break

            results[method_index][counter, :] = current_results

            for reference_file in reference_files:
                ref_path = result_path(root, exp_id, method_folder, reference_file)
                if not os.path.exists(ref_path):
                    correct = False
                    break
                ref_values = load_vector(ref_path)
                value_solution = max(value_solution, np.max(ref_values))

            if not correct:
                break

        if correct:
            for method_index in range(len(method_names)):
                value = np.log(
                    np.maximum(0.0, np.abs(value_solution - results[method_index][counter, :]))
                    / np.abs(value_solution)
                    + 1e-6
                )
                results[method_index][counter, :] = value
            counter += 1

        print(exp_id)

    results = [r[:counter, :] for r in results]
    return results


def bootstrap_sd_of_mean(runs):
    bootstrap_estimator = np.zeros((N_BOOTSTRAP, BO_ITERS))

    for b in range(N_BOOTSTRAP):
        idx = np.random.choice(np.arange(runs.shape[0]), size=runs.shape[0], replace=True)
        bootstrap_estimator[b, :] = runs[idx, :].mean(axis=0)
        print(b + 1)

    finite = np.isfinite(bootstrap_estimator.sum(axis=1))
    return bootstrap_estimator[finite, :].std(axis=0, ddof=1)


def make_plot(root, D, target_name, target_file):
    output_dir = f"{root}/generate_plot"
    os.makedirs(output_dir, exist_ok=True)

    results = load_results(root, target_file)

    n_valid = results[0].shape[0]
    print(f"{D}D - {target_name}: {n_valid} valid experiments")

    if n_valid == 0:
        print(f"No valid experiments for {root}/{target_name}")
        return

    mean_value = np.zeros((len(method_names), BO_ITERS))
    sd_value = np.zeros((len(method_names), BO_ITERS))

    for i in range(len(method_names)):
        mean_value[i, :] = results[i].mean(axis=0)
        sd_value[i, :] = bootstrap_sd_of_mean(results[i])

    iterations = np.arange(1, BO_ITERS + 1)

    plt.figure(figsize=(9, 5))

    for i, method_name in enumerate(method_names):
        plt.errorbar(
            iterations,
            mean_value[i, :],
            yerr=sd_value[i, :],
            label=method_name,
            color=method_colors[i],
            linewidth=0.75,
            markersize=2,
            marker="o",
            elinewidth=0.75,
            capsize=1.5,
        )

    plt.ylabel("Log. Rel. Diff. w.r.t Max")
    plt.xlabel("Number of Function Evaluations")
    plt.title(f"{D} Dimensions. Noiseless. Post Mean.")
    plt.legend(title="Methods", ncol=2)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/plot_{target_name}.pdf")
    plt.close()


def main():
    for D in DIMENSIONS:
        root = ROOT_TEMPLATE.format(D=D)
        for target_name, target_file in plot_targets:
            make_plot(root, D, target_name, target_file)


if __name__ == "__main__":
    main()
