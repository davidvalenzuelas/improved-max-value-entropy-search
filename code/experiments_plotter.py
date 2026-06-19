import os
import numpy as np
import matplotlib.pyplot as plt

DIMENSIONS = [4, 6]
NUM_EXPERIMENTS = 100
BO_ITERS = 100
N_BOOTSTRAP = 200
ROOT_TEMPLATE = "{D}d_experiments"

method_names = ["MES", "MES+", "Random"]
method_folders = ["mes", "mes_plus", "random"]

method_colors = {
    "MES":"#2563EB",
    "MES+": "#DC2626",
    "Random": "#16A34A",
}

reference_files = [
    "objective_at_recommendations_obs.txt",
    "objective_at_recommendations_post_mean.txt",
    "objective_at_recommendations_obs_obs.txt",
    "y_optimum_problem.txt",
]

plot_targets = [
    ("post_mean", "objective_at_recommendations_post_mean.txt", "Posterior mean recommendation"),
    ("obs", "objective_at_recommendations_obs_obs.txt", "Best observed value recommendation"),
]

plt.rcParams.update({
    "font.size": 12,
    "axes.titlesize": 15,
    "axes.labelsize": 13,
    "legend.fontsize": 11,
    "legend.title_fontsize": 11,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "grid.linestyle": "--",
    "lines.linewidth": 2.2,
    "figure.dpi": 120,
    "savefig.dpi": 300,
})


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
                    np.maximum(
                        0.0,
                        np.abs(value_solution - results[method_index][counter, :])
                    )
                    / np.abs(value_solution)
                    + 1e-6
                )
                results[method_index][counter, :] = value

            counter += 1

    results = [r[:counter, :] for r in results]
    return results


def bootstrap_sd_of_mean(runs):
    bootstrap_estimator = np.zeros((N_BOOTSTRAP, BO_ITERS))

    for b in range(N_BOOTSTRAP):
        idx = np.random.choice(
            np.arange(runs.shape[0]),
            size=runs.shape[0],
            replace=True,
        )
        bootstrap_estimator[b, :] = runs[idx, :].mean(axis=0)

    finite = np.isfinite(bootstrap_estimator.sum(axis=1))
    return bootstrap_estimator[finite, :].std(axis=0, ddof=1)


def make_plot(root, D, target_name, target_file, target_title):
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

    fig, ax = plt.subplots(figsize=(10.5, 6.2))

    for i, method_name in enumerate(method_names):
        color = method_colors[method_name]

        ax.plot(
            iterations,
            mean_value[i, :],
            label=method_name,
            color=color,
            linewidth=2.4,
        )

        ax.fill_between(
            iterations,
            mean_value[i, :] - sd_value[i, :],
            mean_value[i, :] + sd_value[i, :],
            color=color,
            alpha=0.16,
            linewidth=0,
        )

    ax.set_xlabel("Number of BO iterations")
    ax.set_ylabel("Log relative difference to the optimum")

    ax.set_title(
        f"{D}D synthetic (noiseless) problem\n{target_title}",
        pad=12,
        fontweight="bold",
    )

    ax.legend(
        title="Method",
        loc="upper right",
        frameon=True,
        fancybox=True,
        framealpha=0.95,
    )

    ax.set_xlim(1, BO_ITERS)

    y_values = np.concatenate([
        mean_value[i, :] for i in range(len(method_names))
    ])
    y_min, y_max = np.nanmin(y_values), np.nanmax(y_values)
    margin = 0.08 * max(1e-8, y_max - y_min)
    ax.set_ylim(y_min - margin, y_max + margin)

    fig.tight_layout()

    pdf_path = f"{output_dir}/plot_{D}d_{target_name}.pdf"
    png_path = f"{output_dir}/plot_{D}d_{target_name}.png"

    fig.savefig(pdf_path, bbox_inches="tight")
    fig.savefig(png_path, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved: {pdf_path}")
    print(f"Saved: {png_path}")


def main():
    np.random.seed(123)

    for D in DIMENSIONS:
        root = ROOT_TEMPLATE.format(D=D)

        for target_name, target_file, target_title in plot_targets:
            make_plot(root, D, target_name, target_file, target_title)


if __name__ == "__main__":
    main()