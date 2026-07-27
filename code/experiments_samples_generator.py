"""
Generates the configuration files for the synthetic 4D experiments,
when sampling different numbers of the optimum.

Author: David Valenzuela Sánchez
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path("experiments/synthetic_4d_samples")

# Sample counts used in the experiments
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
    "obs_obs": (
        "objective_at_recommendations_obs_obs.txt",
        "Best observed recommendation",
    ),
}

METHODS = {
    "MES": "mes",
    "MES+": "mes_plus",
}

METHOD_COLORS = {
    "MES": "#2563EB",
    "MES+": "#DC2626",
}

SAMPLE_COLORS = {
    1: "#DE25EB",
    2: "#26BEDC",
    4: "#16A34A",
    8: "#EA9233",
}

REFERENCE_FILES = [
    "objective_at_recommendations_obs.txt",
    "objective_at_recommendations_post_mean.txt",
    "objective_at_recommendations_obs_obs.txt",
    "y_optimum_problem.txt",
]

OUTPUT_DIR = ROOT / "generate_plot"

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


def load_vector(path: Path) -> np.ndarray:
    values = np.loadtxt(path)

    return (np.atleast_1d(values).astype(float).reshape(-1))


def result_directory(sample_count: int, exp_id: int, method_folder: str) -> Path:
    return (
        ROOT
        / f"samples_{sample_count}"
        / f"exp_{exp_id}"
        / method_folder
        / RESULTS_FOLDER
    )


def experiment_is_complete(exp_id: int) -> bool:
    for sample_count in SAMPLE_COUNTS:
        for method_folder in METHODS.values():
            folder = result_directory(sample_count=sample_count, exp_id=exp_id, method_folder=method_folder)

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


def get_complete_experiment_ids() -> list[int]:
    return [
        exp_id
        for exp_id in range(1, NUM_EXPERIMENTS + 1)
        if experiment_is_complete(exp_id)
    ]


def get_reference_value(exp_id: int) -> float:
    values = []

    # We use the same reference values for all methods and sample counts
    for sample_count in SAMPLE_COUNTS:
        for method_folder in METHODS.values():
            folder = result_directory(sample_count=sample_count, exp_id=exp_id, method_folder=method_folder)

            for filename in REFERENCE_FILES:
                current_values = load_vector(folder / filename)
                current_values = current_values[np.isfinite(current_values)]

                values.extend(current_values.tolist())

    if not values:
        raise RuntimeError(
            f"No finite reference values found for exp_{exp_id}."
        )

    return float(np.max(values))


def load_metric_runs(sample_count: int, method_folder: str, target_file: str, experiment_ids: list[int],
    references: dict[int, float]) -> np.ndarray:
    runs = np.zeros((len(experiment_ids), BO_ITERS),dtype=float)

    for row, exp_id in enumerate(experiment_ids):
        path = (
            result_directory(
                sample_count=sample_count,
                exp_id=exp_id,
                method_folder=method_folder,
            )
            / target_file
        )

        values = load_vector(path)[:BO_ITERS]

        reference = references[exp_id]
        denominator = max(abs(reference), 1e-12)

        # obtains log regret to the optimum
        runs[row] = np.log(np.abs(reference - values) / denominator + 1e-6)

    return runs


def bootstrap_sd_of_mean(runs: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    if runs.shape[0] <= 1:
        return np.zeros(BO_ITERS)

    bootstrap_estimator = np.zeros((N_BOOTSTRAP, BO_ITERS),dtype=float)

    for bootstrap_index in range(N_BOOTSTRAP):
        indices = rng.choice(np.arange(runs.shape[0]),size=runs.shape[0],replace=True)

        bootstrap_estimator[bootstrap_index] = (runs[indices].mean(axis=0))

    finite = np.isfinite(bootstrap_estimator.sum(axis=1))

    return bootstrap_estimator[finite].std(axis=0, ddof=1)


def save_errorbar_plot(curves: list[dict], title: str, legend_title: str, filename: str) -> None:
    iterations = np.arange(1, BO_ITERS + 1)

    fig, ax = plt.subplots(figsize=(10.5, 6.2))

    y_low_values = []
    y_high_values = []

    for curve in curves:
        mean = curve["mean"]
        sd = curve["sd"]

        ax.errorbar(
            iterations,
            mean,
            yerr=sd,
            fmt="o-",
            label=curve["label"],
            color=curve["color"],
            linewidth=1.8,
            markersize=3.2,
            markeredgewidth=0.8,
            elinewidth=1.0,
            capsize=3.0,
            capthick=1.0,
        )

        y_low_values.append(mean - sd)
        y_high_values.append(mean + sd)

    ax.set_xlabel("Number of BO iterations")
    ax.set_ylabel(
        "Log relative difference to the optimum"
    )
    ax.set_title(
        title,
        pad=12,
        fontweight="bold",
    )

    ax.legend(
        title=legend_title,
        loc="upper right",
        frameon=True,
        fancybox=True,
        framealpha=0.95,
    )

    ax.set_xlim(1, BO_ITERS)

    y_low_values = np.concatenate(y_low_values)
    y_high_values = np.concatenate(y_high_values)

    y_min = np.nanmin(y_low_values)
    y_max = np.nanmax(y_high_values)

    margin = 0.08 * max(
        1e-8,
        y_max - y_min,
    )

    ax.set_ylim(
        y_min - margin,
        y_max + margin,
    )

    fig.tight_layout()

    png_path = OUTPUT_DIR / f"{filename}.png"
    pdf_path = OUTPUT_DIR / f"{filename}.pdf"

    fig.savefig(
        png_path,
        bbox_inches="tight",
    )
    fig.savefig(
        pdf_path,
        bbox_inches="tight",
    )

    plt.close(fig)

    print(f"Saved: {png_path}")
    print(f"Saved: {pdf_path}")


def plot_mes_vs_mes_plus(sample_count: int, target_name: str, target_file: str, target_title: str,
    experiment_ids: list[int], references: dict[int, float]) -> None:
    # Different seed for each sample count
    rng = np.random.default_rng(123 + sample_count)

    curves = []

    for method_name, method_folder in METHODS.items():
        runs = load_metric_runs(sample_count=sample_count, method_folder=method_folder,
            target_file=target_file, experiment_ids=experiment_ids, references=references)

        curves.append({
            "label": method_name,
            "color": METHOD_COLORS[method_name],
            "mean": runs.mean(axis=0),
            "sd": bootstrap_sd_of_mean(
                runs,
                rng,
            ),
        })

    if sample_count == 1:
        sample_label = "1 optimum sample"
    else:
        sample_label = (
            f"{sample_count} optimum samples"
        )

    save_errorbar_plot(
        curves=curves,
        title=(
            f"MES vs MES+ with {sample_label}\n"
            f"4D synthetic problem — {target_title}"
        ),
        legend_title="Method",
        filename=(
            f"plot_samples_{sample_count}_"
            f"mes_vs_mes_plus_{target_name}"
        ),
    )


def plot_samples_for_method(method_name: str, method_folder: str, target_name: str, target_file: str,
    target_title: str, experiment_ids: list[int], references: dict[int, float]) -> None:
    if method_name == "MES":
        seed = 321
    else:
        seed = 654

    rng = np.random.default_rng(seed)
    curves = []

    for sample_count in SAMPLE_COUNTS:
        runs = load_metric_runs(
            sample_count=sample_count,
            method_folder=method_folder,
            target_file=target_file,
            experiment_ids=experiment_ids,
            references=references,
        )

        if sample_count == 1:
            sample_label = "1 sample"
        else:
            sample_label = (
                f"{sample_count} samples"
            )

        curves.append({
            "label": sample_label,
            "color": SAMPLE_COLORS[sample_count],
            "mean": runs.mean(axis=0),
            "sd": bootstrap_sd_of_mean(
                runs,
                rng,
            ),
        })

    if method_name == "MES":
        filename_method = "mes"
    else:
        filename_method = "mes_plus"

    save_errorbar_plot(
        curves=curves,
        title=(
            f"{method_name} with different numbers "
            f"of optimum samples\n"
            f"4D synthetic problem — {target_title}"
        ),
        legend_title="Optimum samples",
        filename=(
            f"plot_{filename_method}_"
            f"samples_{target_name}"
        ),
    )


def main() -> None:
    for sample_count in SAMPLE_COUNTS:
        sample_root = (
            ROOT
            / f"samples_{sample_count}"
        )

        if not sample_root.is_dir():
            raise FileNotFoundError(
                f"Missing experiment folder: "
                f"{sample_root}"
            )

    experiment_ids = (get_complete_experiment_ids())

    if not experiment_ids:
        raise RuntimeError(
            "There are no complete experiments "
            "shared by all methods, sample counts "
            "and target metrics."
        )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    references = {
        exp_id: get_reference_value(exp_id)
        for exp_id in experiment_ids
    }

    print(
        f"Sample counts: {SAMPLE_COUNTS}"
    )
    print(
        "Complete paired experiments used: "
        f"{len(experiment_ids)}"
    )

    for (
        target_name,
        (target_file, target_title),
    ) in TARGETS.items():

        for sample_count in SAMPLE_COUNTS:
            plot_mes_vs_mes_plus(
                sample_count=sample_count,
                target_name=target_name,
                target_file=target_file,
                target_title=target_title,
                experiment_ids=experiment_ids,
                references=references,
            )

        plot_samples_for_method(
            method_name="MES",
            method_folder=METHODS["MES"],
            target_name=target_name,
            target_file=target_file,
            target_title=target_title,
            experiment_ids=experiment_ids,
            references=references,
        )

        plot_samples_for_method(
            method_name="MES+",
            method_folder=METHODS["MES+"],
            target_name=target_name,
            target_file=target_file,
            target_title=target_title,
            experiment_ids=experiment_ids,
            references=references,
        )


if __name__ == "__main__":
    main()
