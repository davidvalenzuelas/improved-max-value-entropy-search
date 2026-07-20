"""
Generates the plots for both synthetic and benchmark
experiments.

Author: David Valenzuela Sánchez
"""
import os
import numpy as np
import matplotlib.pyplot as plt

# Info about the experiment
PROBLEM_TYPE = "benchmark"
BENCHMARK = "shekel"

BO_ITERS = 100
N_BOOTSTRAP = 200

if PROBLEM_TYPE == "synthetic":
    DIMENSIONS = [4, 6]
    NUM_EXPERIMENTS = 100
    ROOT_TEMPLATE = "experiments/synthetic_{D}d"
    RESULTS_FOLDER = "results_synthetic_problem"

    method_names = ["MES", "MES+", "Random"]

    method_colors = {
        "MES":"#2563EB",
        "MES+": "#DC2626",
        "Random": "#16A34A",
    }

    plot_targets = [
        ("post_mean", "objective_at_recommendations_post_mean.txt", "Posterior mean recommendation"),
        ("obs", "objective_at_recommendations_obs_obs.txt", "Best observed value recommendation"),
    ]

elif PROBLEM_TYPE == "benchmark":
    DIMENSIONS = [4]
    NUM_EXPERIMENTS = 200
    ROOT_TEMPLATE = f"experiments/{BENCHMARK}"
    RESULTS_FOLDER = "results_benchmark_problem"

    method_names = ["MES", "MES+"]

    method_colors = {
        "MES": "#2563EB",
        "MES+": "#DC2626",
        #"RANDOM": "#16A34A"
    }

    plot_targets = [
        (
            "post_mean",
            "objective_at_recommendations_post_mean.txt",
            "Posterior mean recommendation",
        ),
        (
            "obs",
            "objective_at_recommendations_obs_obs.txt",
            "Best observed recommendation",
        ),
    ]

else:
    raise ValueError(f"Unknown problem type: {PROBLEM_TYPE}")

method_folders = ["mes", "mes_plus"]

reference_files = [
    "objective_at_recommendations_obs.txt",
    "objective_at_recommendations_post_mean.txt",
    "objective_at_recommendations_obs_obs.txt",
    "y_optimum_problem.txt",
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
    return f"{root}/exp_{exp_id}/{method_folder}/{RESULTS_FOLDER}/{filename}"


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

        if PROBLEM_TYPE == "benchmark":
            print(exp_id)

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

        if PROBLEM_TYPE == "synthetic":
            ax.errorbar(
                iterations,
                mean_value[i, :],
                yerr=sd_value[i, :],
                fmt="o-",
                label=method_name,
                color=color,
                linewidth=1.8,
                markersize=3.2,
                markeredgewidth=0.8,
                elinewidth=1.0,
                capsize=3.0,
                capthick=1.0,
            )
        else:
            ax.errorbar(
                iterations,
                mean_value[i, :],
                yerr=sd_value[i, :],
                fmt="o-",
                label=method_name,
                color=color,
                linewidth=1.8,
                markersize=3.0,
                markeredgewidth=0.8,
                elinewidth=0.9,
                capsize=2.8,
                capthick=0.9,
            )

    ax.set_xlabel("Number of BO iterations")
    ax.set_ylabel("Log relative difference to the optimum")

    if PROBLEM_TYPE == "synthetic":
        plot_title = f"{D}D synthetic (noiseless) problem\n{target_title}"
    else:
        plot_title = f"{BENCHMARK.capitalize()} {D}D benchmark\n{target_title}"

    ax.set_title(
        plot_title,
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

    y_low_values = np.concatenate([
        mean_value[i, :] - sd_value[i, :] for i in range(len(method_names))
    ])

    y_high_values = np.concatenate([
        mean_value[i, :] + sd_value[i, :] for i in range(len(method_names))
    ])

    y_min = np.nanmin(y_low_values)
    y_max = np.nanmax(y_high_values)

    margin = 0.08 * max(1e-8, y_max - y_min)
    ax.set_ylim(y_min - margin, y_max + margin)

    fig.tight_layout()

    if PROBLEM_TYPE == "synthetic":
        pdf_path = f"{output_dir}/plot_{D}d_{target_name}.pdf"
        png_path = f"{output_dir}/plot_{D}d_{target_name}.png"
    else:
        pdf_path = f"{output_dir}/plot_{BENCHMARK}{D}d_{target_name}.pdf"
        png_path = f"{output_dir}/plot_{BENCHMARK}{D}d_{target_name}.png"

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