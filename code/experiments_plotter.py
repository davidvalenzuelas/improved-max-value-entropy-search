"""
Generates the plots for the synthetic 4D and 6D experiments
with different numbers of optimum samples.

Author: David Valenzuela Sánchez
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


DIMENSIONS = [4, 6]
SAMPLE_COUNTS = [1, 2, 4, 8]

NUM_EXPERIMENTS = 100
BO_ITERS = 100
N_BOOTSTRAP = 200

RESULTS_FOLDER = "results_synthetic_problem"

TARGETS = {
    "post_mean": (
        "objective_at_recommendations_post_mean.txt",
        "Posterior mean recommendation",
    ),
    "obs": (
        "objective_at_recommendations_obs_obs.txt",
        "Best observed value recommendation",
    ),
}

METHODS = {
    "MES": "mes",
    "MES+": "mes_plus",
}

METHOD_COLORS = {
    "MES": {
        1: "#FCA5A5",
        2: "#F87171",
        4: "#DC2626",
        8: "#7F1D1D",
    },
    "MES+": {
        1: "#93C5FD",
        2: "#60A5FA",
        4: "#2563EB",
        8: "#1E3A8A",
    },
}

REFERENCE_FILES = [
    "objective_at_recommendations_obs.txt",
    "objective_at_recommendations_post_mean.txt",
    "objective_at_recommendations_obs_obs.txt",
    "y_optimum_problem.txt",
]


plt.rcParams.update({
    "font.size": 12,
    "axes.titlesize": 15,
    "axes.labelsize": 13,
    "legend.fontsize": 10,
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


def result_directory(root, sample_count, exp_id, method_folder):
    return (
        root
        / f"samples_{sample_count}"
        / f"exp_{exp_id}"
        / method_folder
        / RESULTS_FOLDER
    )


def experiment_is_complete(root, exp_id):
    for sample_count in SAMPLE_COUNTS:
        for method_folder in METHODS.values():
            folder = result_directory(root, sample_count, exp_id, method_folder)

            for target_file, _ in TARGETS.values():
                target_path = folder / target_file

                if not target_path.exists():
                    return False

                if load_vector(target_path).size < BO_ITERS:
                    return False

            for filename in REFERENCE_FILES:
                if not (folder / filename).exists():
                    return False

    return True


def get_complete_experiment_ids(root):
    return [
        exp_id
        for exp_id in range(1, NUM_EXPERIMENTS + 1)
        if experiment_is_complete(root, exp_id)
    ]


def get_reference_value(root, exp_id):
    values = []

    # We use the same reference value for all methods and sample counts
    for sample_count in SAMPLE_COUNTS:
        for method_folder in METHODS.values():
            folder = result_directory(root, sample_count, exp_id, method_folder)

            for filename in REFERENCE_FILES:
                current_values = load_vector(folder / filename)
                current_values = current_values[np.isfinite(current_values)]
                values.extend(current_values.tolist())

    if not values:
        raise RuntimeError(f"No finite reference values found for exp_{exp_id}.")

    return float(np.max(values))


def load_results(root, sample_count, method_folder, target_file,
    experiment_ids, references):

    results = np.zeros((len(experiment_ids), BO_ITERS))

    for row, exp_id in enumerate(experiment_ids):
        path = (
            result_directory(root, sample_count, exp_id, method_folder)
            / target_file
        )

        current_results = load_vector(path)[:BO_ITERS]
        value_solution = references[exp_id]

        # Obtains log regret to the optimum
        results[row, :] = np.log(
            np.maximum(0.0, np.abs(value_solution - current_results))
            / np.abs(value_solution)
            + 1e-6
        )

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


def make_plot(D, target_name, target_file, target_title):
    root = Path(f"experiments/synthetic_{D}d_samples")
    output_dir = root / "generate_plot"
    output_dir.mkdir(parents=True, exist_ok=True)

    experiment_ids = get_complete_experiment_ids(root)

    print(f"{D}D - {target_name}: {len(experiment_ids)} valid experiments")

    if not experiment_ids:
        print(f"No valid experiments for {D}D")
        return

    references = {
        exp_id: get_reference_value(root, exp_id)
        for exp_id in experiment_ids
    }

    iterations = np.arange(1, BO_ITERS + 1)

    fig, ax = plt.subplots(figsize=(11, 6.5))

    y_low_values = []
    y_high_values = []

    for method_name, method_folder in METHODS.items():
        for sample_count in SAMPLE_COUNTS:
            results = load_results(
                root, sample_count, method_folder, target_file,
                experiment_ids, references
            )

            mean_value = results.mean(axis=0)
            sd_value = bootstrap_sd_of_mean(results)

            if sample_count == 1:
                sample_label = "1 sample"
            else:
                sample_label = f"{sample_count} samples"

            ax.errorbar(
                iterations,
                mean_value,
                yerr=sd_value,
                fmt="o-",
                label=f"{method_name} - {sample_label}",
                color=METHOD_COLORS[method_name][sample_count],
                linewidth=1.8,
                markersize=3.2,
                markeredgewidth=0.8,
                elinewidth=1.0,
                capsize=3.0,
                capthick=1.0,
            )

            y_low_values.append(mean_value - sd_value)
            y_high_values.append(mean_value + sd_value)

    ax.set_xlabel("Number of BO iterations")
    ax.set_ylabel("Log relative difference to the optimum")

    ax.set_title(
        f"{D}D synthetic (noiseless) problem\n{target_title}",
        pad=12,
        fontweight="bold",
    )

    ax.legend(
        title="Method / optimum samples",
        loc="upper right",
        ncol=2,
        frameon=True,
        fancybox=True,
        framealpha=0.95,
    )

    ax.set_xlim(1, BO_ITERS)

    y_low_values = np.concatenate(y_low_values)
    y_high_values = np.concatenate(y_high_values)

    y_min = np.nanmin(y_low_values)
    y_max = np.nanmax(y_high_values)

    margin = 0.08 * max(1e-8, y_max - y_min)
    ax.set_ylim(y_min - margin, y_max + margin)

    fig.tight_layout()

    pdf_path = output_dir / f"plot_{D}d_samples_{target_name}.pdf"
    png_path = output_dir / f"plot_{D}d_samples_{target_name}.png"

    fig.savefig(pdf_path, bbox_inches="tight")
    fig.savefig(png_path, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved: {pdf_path}")
    print(f"Saved: {png_path}")


def main():
    np.random.seed(123)

    for D in DIMENSIONS:
        for target_name, (target_file, target_title) in TARGETS.items():
            make_plot(D, target_name, target_file, target_title)


if __name__ == "__main__":
    main()