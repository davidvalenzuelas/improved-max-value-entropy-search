from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path("experiments/synthetic_6d_samples")

SAMPLE_COUNTS = [1, 2, 4, 8]

NUM_EXPERIMENTS = 100
BO_ITERS = 100
N_BOOTSTRAP = 200

RESULTS_FOLDER = "results_synthetic_problem"
TARGET_FILE = "objective_at_recommendations_post_mean.txt"

METHODS = {
    "MES": "mes",
    "MES+": "mes_plus",
}

METHOD_COLORS = {
    "MES":"#2563EB",
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
    return np.atleast_1d(values).astype(float).reshape(-1)


def result_directory(
    sample_count: int,
    exp_id: int,
    method_folder: str,
) -> Path:
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
            folder = result_directory(
                sample_count,
                exp_id,
                method_folder,
            )

            target_path = folder / TARGET_FILE

            if not target_path.exists():
                return False

            target_values = load_vector(target_path)

            if target_values.size < BO_ITERS:
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

    # Utilizamos la misma referencia para MES, MES+ y todos los números
    # de muestras comparados en esta repetición.
    for sample_count in SAMPLE_COUNTS:
        for method_folder in METHODS.values():
            folder = result_directory(
                sample_count,
                exp_id,
                method_folder,
            )

            for filename in REFERENCE_FILES:
                current_values = load_vector(folder / filename)
                current_values = current_values[
                    np.isfinite(current_values)
                ]
                values.extend(current_values.tolist())

    if not values:
        raise RuntimeError(
            f"No finite reference values found for exp_{exp_id}."
        )

    return float(np.max(values))


def load_metric_runs(
    sample_count: int,
    method_folder: str,
    experiment_ids: list[int],
    references: dict[int, float],
) -> np.ndarray:
    runs = np.zeros(
        (len(experiment_ids), BO_ITERS),
        dtype=float,
    )

    for row, exp_id in enumerate(experiment_ids):
        path = (
            result_directory(
                sample_count,
                exp_id,
                method_folder,
            )
            / TARGET_FILE
        )

        values = load_vector(path)[:BO_ITERS]

        reference = references[exp_id]
        denominator = max(abs(reference), 1e-12)

        runs[row] = np.log(
            np.abs(reference - values) / denominator
            + 1e-6
        )

    return runs


def bootstrap_sd_of_mean(
    runs: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    if runs.shape[0] <= 1:
        return np.zeros(BO_ITERS)

    bootstrap_estimator = np.zeros(
        (N_BOOTSTRAP, BO_ITERS),
        dtype=float,
    )

    for bootstrap_index in range(N_BOOTSTRAP):
        indices = rng.choice(
            np.arange(runs.shape[0]),
            size=runs.shape[0],
            replace=True,
        )

        bootstrap_estimator[bootstrap_index] = (
            runs[indices].mean(axis=0)
        )

    finite = np.isfinite(
        bootstrap_estimator.sum(axis=1)
    )

    return bootstrap_estimator[finite].std(
        axis=0,
        ddof=1,
    )


def save_errorbar_plot(
    curves: list[dict],
    title: str,
    legend_title: str,
    filename: str,
) -> None:
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
    ax.set_ylabel("Log relative difference to the optimum")

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

    margin = 0.08 * max(1e-8, y_max - y_min)
    ax.set_ylim(y_min - margin, y_max + margin)

    fig.tight_layout()

    png_path = OUTPUT_DIR / f"{filename}.png"
    pdf_path = OUTPUT_DIR / f"{filename}.pdf"

    fig.savefig(png_path, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved: {png_path}")
    print(f"Saved: {pdf_path}")


def plot_mes_vs_mes_plus(
    sample_count: int,
    experiment_ids: list[int],
    references: dict[int, float],
) -> None:
    rng = np.random.default_rng(123 + sample_count)
    curves = []

    for method_name, method_folder in METHODS.items():
        runs = load_metric_runs(
            sample_count=sample_count,
            method_folder=method_folder,
            experiment_ids=experiment_ids,
            references=references,
        )

        curves.append({
            "label": method_name,
            "color": METHOD_COLORS[method_name],
            "mean": runs.mean(axis=0),
            "sd": bootstrap_sd_of_mean(runs, rng),
        })

    sample_label = (
        "1 optimum sample"
        if sample_count == 1
        else f"{sample_count} optimum samples"
    )

    save_errorbar_plot(
        curves=curves,
        title=(
            f"MES vs MES+ with {sample_label}\n"
            "6D synthetic problem — "
            "posterior mean recommendation"
        ),
        legend_title="Method",
        filename=(
            f"plot_samples_{sample_count}_"
            "mes_vs_mes_plus_post_mean"
        ),
    )


def plot_samples_for_method(
    method_name: str,
    method_folder: str,
    experiment_ids: list[int],
    references: dict[int, float],
) -> None:
    rng = np.random.default_rng(
        321 if method_name == "MES" else 654
    )

    curves = []

    for sample_count in SAMPLE_COUNTS:
        runs = load_metric_runs(
            sample_count=sample_count,
            method_folder=method_folder,
            experiment_ids=experiment_ids,
            references=references,
        )

        sample_label = (
            "1 sample"
            if sample_count == 1
            else f"{sample_count} samples"
        )

        curves.append({
            "label": sample_label,
            "color": SAMPLE_COLORS[sample_count],
            "mean": runs.mean(axis=0),
            "sd": bootstrap_sd_of_mean(runs, rng),
        })

    filename_method = (
        "mes" if method_name == "MES" else "mes_plus"
    )

    save_errorbar_plot(
        curves=curves,
        title=(
            f"{method_name} with different numbers "
            "of optimum samples\n"
            "6D synthetic problem — "
            "posterior mean recommendation"
        ),
        legend_title="Optimum samples",
        filename=(
            f"plot_{filename_method}_"
            "samples_post_mean"
        ),
    )


def main() -> None:
    for sample_count in SAMPLE_COUNTS:
        sample_root = ROOT / f"samples_{sample_count}"

        if not sample_root.is_dir():
            raise FileNotFoundError(
                f"Missing experiment folder: {sample_root}"
            )

    experiment_ids = get_complete_experiment_ids()

    if not experiment_ids:
        raise RuntimeError(
            "There are no complete experiments shared by "
            "all methods and sample counts."
        )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    references = {
        exp_id: get_reference_value(exp_id)
        for exp_id in experiment_ids
    }

    print(f"Sample counts: {SAMPLE_COUNTS}")
    print(
        "Complete paired experiments used: "
        f"{len(experiment_ids)}"
    )

    # Una gráfica MES vs MES+ para cada número de muestras.
    for sample_count in SAMPLE_COUNTS:
        plot_mes_vs_mes_plus(
            sample_count=sample_count,
            experiment_ids=experiment_ids,
            references=references,
        )

    # Una gráfica comparando todos los números de muestras para MES.
    plot_samples_for_method(
        method_name="MES",
        method_folder=METHODS["MES"],
        experiment_ids=experiment_ids,
        references=references,
    )

    # Una gráfica comparando todos los números de muestras para MES+.
    plot_samples_for_method(
        method_name="MES+",
        method_folder=METHODS["MES+"],
        experiment_ids=experiment_ids,
        references=references,
    )


if __name__ == "__main__":
    main()
