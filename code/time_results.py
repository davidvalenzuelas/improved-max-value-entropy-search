"""
Obtains the time results of the synthetic 4D experiments, when we measure the
time taken by each method.

Author: David Valenzuela Sánchez
"""
from pathlib import Path

import numpy as np


ROOT = Path("experiments/time4d")
NUM_MEASUREMENTS = 10
EXPECTED_TIMES = 100

METHODS = {
    "RANDOM": "random",
    "MES": "mes",
    "MES+": "mes_plus",
}


def load_mean_time(exp_id: int, method_folder: str) -> float:
    path = (
        ROOT
        / f"exp_{exp_id}"
        / method_folder
        / "results_synthetic_problem"
        / "times.txt"
    )

    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path}")

    times = np.atleast_1d(
        np.loadtxt(path, dtype=float)
    ).reshape(-1)

    if times.size != EXPECTED_TIMES:
        raise ValueError(
            f"{path} contains {times.size} times; "
            f"expected {EXPECTED_TIMES}."
        )

    if not np.all(np.isfinite(times)):
        raise ValueError(
            f"{path} contains non-finite values."
        )

    return float(times.mean())


mean_times = {
    method_name: np.array(
        [
            load_mean_time(exp_id, method_folder)
            for exp_id in range(1, NUM_MEASUREMENTS + 1)
        ],
        dtype=float,
    )
    for method_name, method_folder in METHODS.items()
}

corrected_mes = mean_times["MES"] - mean_times["RANDOM"]
corrected_mes_plus = mean_times["MES+"] - mean_times["RANDOM"]


print("\nTiempo medio por iteración en cada medición (segundos)\n")

header = (
    f"{'Medición':>8}"
    f"{'RANDOM':>14}"
    f"{'MES':>14}"
    f"{'MES+':>14}"
    f"{'MES-RANDOM':>16}"
    f"{'MES+-RANDOM':>17}"
)

print(header)
print("-" * len(header))

for index in range(NUM_MEASUREMENTS):
    print(
        f"{index + 1:>8d}"
        f"{mean_times['RANDOM'][index]:>14.6f}"
        f"{mean_times['MES'][index]:>14.6f}"
        f"{mean_times['MES+'][index]:>14.6f}"
        f"{corrected_mes[index]:>16.6f}"
        f"{corrected_mes_plus[index]:>17.6f}"
    )


results = {
    "RANDOM": mean_times["RANDOM"],
    "MES": mean_times["MES"],
    "MES+": mean_times["MES+"],
    "MES - RANDOM": corrected_mes,
    "MES+ - RANDOM": corrected_mes_plus,
}


print("\nResumen de las 10 mediciones\n")

summary_header = (
    f"{'Método':<16}"
    f"{'Media (s)':>14}"
    f"{'Desv. estándar (s)':>22}"
)

print(summary_header)
print("-" * len(summary_header))

for name, values in results.items():
    print(
        f"{name:<16}"
        f"{values.mean():>14.6f}"
        f"{values.std(ddof=1):>22.6f}"
    )
