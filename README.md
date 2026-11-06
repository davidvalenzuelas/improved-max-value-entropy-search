# Improved Max-value Entropy Search

This repository contains the code developed for the Bachelor's Thesis "Optimización Bayesiana mediante Entropía del Máximo Valor Mejorada", authored by David Ángel Valenzuela Sánchez under the supervision of Daniel Hernández-Lobato at the Universidad Autónoma de Madrid.

The project implements and compares Bayesian optimization (BO) acquisition based on Max-value entropy. In particular, it includes:

* **MES**, based on a Gaussian truncation approximation.
* **MES+**, our proposed method, which replaces the local Gaussian truncation approximation with a modified variational sparse GP.
* **Random search**, used as a baseline for comparison.

The main contribution of MES+ is the use of a VFE sparse GP trained with a modified ELBO that incorporates a soft probabilistic constraint. This constraint encourages the model predictions to remain below a sampled maximum value, providing a more global approximation to the corresponding conditional distribution.

The experiments are performed on synthetic black box optimization problems sampled from GP priors, benchmark functions and real MLP hyperparameter optimization problems. Our objective is to compare the behaviour and computational cost of these methods within a BO loop.

## Repository structure

The repository is organized as follows:

```text
improved-max-value-entropy-search/
├── code/
│   ├── my_loop_BO.py
│   ├── my_acquisition_mes.py
│   ├── my_acquisition_mes_plus.py
│   ├── modified_vfe_sparse_gp.py
│   ├── my_utils.py
│   ├── util.py
│   ├── synthetic_problem.py
│   ├── benchmark_problem.py
│   ├── mlp_real_problem.py
│   ├── experiments_generator.py
│   ├── experiments_mlp_generator.py
│   ├── experiments_time_generator.py
│   ├── experiments_plotter.py
│   ├── experiments_mlp_plotter.py
│   ├── experiments_samples_generator.py
│   ├── time_results.py
│   ├── config.json
│   ├── config_mlp.json
│   └── tests/
├── requirements.txt
├── README.md
├── LICENSE
├── CITATION.cff
└── .gitignore
```

The main files are:

* `code/my_loop_BO.py`: main BO loop for synthetic, benchmark and real problems.
* `code/my_acquisition_mes.py`: MES acquisition function, based on local Gaussian truncation.
* `code/my_acquisition_mes_plus.py`: MES+ acquisition function, based on our more global proposed variational approximation.
* `code/modified_vfe_sparse_gp.py`: implementation of the modified VFE sparse GP used by MES+.
* `code/my_utils.py`: utility functions for fitting GPs, sampling candidate optima, conditioning models and computing predictive quantities.
* `code/util.py`: general utilities for reading configurations, managing paths, setting random seeds and transforming MLP hyperparameters.
* `code/synthetic_problem.py`: generation of synthetic objective functions sampled from a GP prior.
* `code/benchmark_problem.py`: implementation of the Shekel and Michalewicz benchmark problems.
* `code/mlp_real_problem.py`: definition of the real MLP hyperparameter optimization problems using the Australian Credit Approval and Ionosphere datasets.
* `code/experiments_generator.py`: generation of configurations for synthetic and benchmark experiments.
* `code/experiments_mlp_generator.py`: generation of configurations for the real MLP experiments.
* `code/experiments_time_generator.py`: generation of configurations for the 4D timing experiments.
* `code/experiments_plotter.py`: aggregation and plotting of synthetic and benchmark results.
* `code/experiments_mlp_plotter.py`: aggregation and plotting of the real MLP experiment results.
* `code/experiments_samples_generator.py`: comparison and plotting of MES and MES+ with different numbers of posterior optimum samples.
* `code/time_results.py`: summary of the results obtained in the timing experiments.
* `code/tests/`: 1D validation scripts for the MES and MES+ approximations and acquisition functions.

External libraries such as BoTorch and GPyTorch are **not included** in the repository. They are installed through `requirements.txt`.

## Requirements

The code requires:

* Python >= 3.10
* PyTorch
* GPyTorch
* BoTorch
* NumPy
* SciPy
* Matplotlib
* Pandas
* Scikit-learn

These dependencies are listed in `requirements.txt`.

## Installation

From the repository root, create a virtual environment:

```bash
python3 -m venv venv
```

Activate the environment. On Linux or macOS:

```bash
source venv/bin/activate
```

Then install the dependencies:

```bash
python3 -m pip install --upgrade pip setuptools wheel
pip install --no-cache-dir -r requirements.txt
```

If disk space is limited and GPU support is not needed, the CPU version of PyTorch can be installed first:

```bash
pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu
pip install --no-cache-dir gpytorch botorch numpy scipy matplotlib pandas scikit-learn
```

The `venv/` is now ready to be used.

## Running tests

The 1D validation scripts are included in `code/tests/`. They are useful for checking the behaviour of the MES and MES+ acquisitions and the approximations used to compute them.

Run the required test script from the `code/` directory:

```bash
cd code
python3 tests/<test_file>.py
```

## Configuration files

The BO loop is controlled through JSON configuration files. A typical synthetic configuration contains:

```json
{
  "problem_type": "synthetic",
  "random_seed": 1,
  "lenghtscale_model_synthetic_problem": 0.25,
  "num_samples_solution": 1,
  "num_initial_obs": 10,
  "num_dims": 4,
  "BO_iters": 100,
  "M_extra": 100,
  "num_repetitions": 10,
  "acquisitions": ["MES+", "MES", "RANDOM"],
  "file_results": "experiments/example_experiment"
}
```

The most relevant parameters are:

* `problem_type`: type of problem. The supported values are `"synthetic"`, `"benchmark"` and `"real"`.
* `random_seed`: seed used to initialize the problem and the optimization process.
* `lenghtscale_model_synthetic_problem`: lengthscale used to sample the synthetic objective function.
* `num_samples_solution`: number of posterior optimum samples used to approximate the acquisition function.
* `num_initial_obs`: number of initial observations before the BO loop starts.
* `num_dims`: input dimension of the problem.
* `BO_iters`: number of BO iterations.
* `M_extra`: number of extra inducing points used by the sparse approximation.
* `num_repetitions`: number of independent repetitions.
* `acquisitions`: list of acquisition strategies to compare. The supported values are `"MES+"`, `"MES"` and `"RANDOM"`.
* `file_results`: directory where the results will be stored.
* `measure_time`: optional Boolean value indicating whether the execution time of every BO iteration should be recorded.

Benchmark configurations also contain the `benchmark` parameter, while real MLP configurations contain the `dataset`, `n_folds` and `n_repeats` parameters.

## Generating batches of experiments

The script `code/experiments_generator.py` creates folders and configuration files for synthetic and benchmark experiments. It reads a base configuration from `code/config.json` and generates one configuration per experiment and method.

Before running it, check the constants at the top of the file, as in this example:

```python
PROBLEM_TYPE = "synthetic"
BENCHMARK = "michalewicz"
DIMENSIONS = [6]
START_EXP = 1
END_EXP = 100
BO_ITERS = 100
NUM_SAMPLES_SOLUTION = 1
```

These constants control:

* `PROBLEM_TYPE`: whether synthetic or benchmark problems are generated.
* `BENCHMARK`: benchmark to use, when `PROBLEM_TYPE` is `"benchmark"`.
* `DIMENSIONS`: dimensions used in the experiments.
* `START_EXP` and `END_EXP`: range of experiment identifiers.
* `BO_ITERS`: number of BO iterations.
* `NUM_SAMPLES_SOLUTION`: number of posterior optimum samples used by the acquisition functions.

E.g., a synthetic experiment can have the following structure:

```text
experiments/synthetic_6d_samples/samples_1/
├── exp_1/
│   ├── config_mes.json
│   ├── config_mes_plus.json
│   ├── mes/
│   │   └── results_synthetic_problem/
│   └── mes_plus/
│       └── results_synthetic_problem/
├── exp_2/
│   └── ...
└── ...
```

To generate the experiment configurations, run:

```bash
cd code
python3 experiments_generator.py
```

After generating the configurations, the experiments can be launched individually or in batches.

The script `code/experiments_mlp_generator.py` generates the configurations for the Australian Credit Approval and Ionosphere experiments:

```bash
cd code
python3 experiments_mlp_generator.py
```

## Running BO experiments

The main BO loop is implemented in `code/my_loop_BO.py`. As explained, it takes a JSON configuration file as an argument and runs the specified acquisition method on the selected optimization problem.

For example, after generating the configurations for the 6D synthetic experiments, a single MES experiment can be launched from the `code/` directory with:

```bash
cd code
python3 my_loop_BO.py experiments/synthetic_6d_samples/samples_1/exp_1/config_mes.json
```

To run MES+ instead, use the corresponding `config_mes_plus.json` file.

The results are saved in the directory specified by the `file_results` parameter. Multiple experiments can be executed independently using their respective configuration files.

## Timing experiments

The script `code/experiments_time_generator.py` generates configurations for 10 independent 4D synthetic experiments, comparing the execution times of MES, MES+ and random search. These configurations can be generated with:

```bash
cd code
python3 experiments_time_generator.py
```

When `measure_time` is enabled, `my_loop_BO.py` records the time spent fitting the GP and selecting the next candidate, excluding objective evaluations and recommendation computations, in a file named:

```text
times.txt
```

The script `time_results.py` reads these files and prints the corresponding timing results.

```bash
cd code
python3 time_results.py
```

## Plotting experiment results

The script `code/experiments_plotter.py` aggregates the results generated by synthetic and benchmark experiments and produces comparison plots.

Before running it, check the constants at the top of the file. These constants must match the experiments that were actually run.

To generate the plots, run:

```bash
cd code
python3 experiments_plotter.py
```

The real MLP results are plotted with:

```bash
cd code
python3 experiments_mlp_plotter.py
```

The experiments comparing different numbers of posterior optimum samples are plotted with:

```bash
cd code
python3 experiments_samples_generator.py
```

Before running this script, check that its configuration matches the experiments to be plotted. The plots are saved in a `generate_plot/` folder inside the corresponding experiment directory.

The synthetic and benchmark plotters compute the log relative difference with respect to the best available value and use bootstrap estimates to represent the uncertainty of the mean curves. The real MLP plotter represents validation accuracy.

## Output files

During each run, the code stores several files with recommendations and objective values, including:

* `recommendations_obs.txt`
* `objective_at_recommendations_obs.txt`
* `recommendations_obs_obs.txt`
* `objective_at_recommendations_obs_obs.txt`
* `recommendations_post_mean.txt`
* `objective_at_recommendations_post_mean.txt`
* `points_evaluated.txt`
* `y_values_evaluated.txt`
* `x_optimum_problem.txt`
* `y_optimum_problem.txt`

Three recommendation strategies are considered: selecting the observed point with the highest GP posterior mean (`recommendations_obs`), selecting the best observed point (`recommendations_obs_obs`), and maximizing the GP posterior mean over the search domain (`recommendations_post_mean`). The corresponding objective values are also stored for performance evaluation.

For benchmark problems, the original optimum location is also stored in:

* `x_optimum_original_problem.txt`

When timing is enabled, the following file is also generated:

* `times.txt`

The optimum files are not generated for real MLP problems because their global optimum is unknown. The exact directory structure depends on the value of `file_results` and on whether multiple repetitions and acquisition functions are used.

## Citation

If you use this repository, please cite it as:

```bibtex
@software{valenzuelahernandezlobato2026improvedmes,
  author = {Valenzuela S{\'a}nchez, D. A. and Hern{\'a}ndez-Lobato, D.},
  title = {Improved Max-value Entropy Search},
  year = {2026},
  version = {1.0.0},
  url = {https://github.com/davidvalenzuelas/improved-max-value-entropy-search},
  note = {Code repository associated with the Bachelor's Thesis ``Optimización Bayesiana mediante Entropía del Máximo Valor Mejorada''}
}
```

## Authors

Daniel Hernández-Lobato and David A. Valenzuela-Sánchez

## Acknowledgements

We acknowledge support from projects PID2022-139856NB-I00 and PID2025-167768NB-I00, funded by MCIN, and from project IDEA-CM (TEC-2024/COM-89) and from the ELLIS Unit Madrid, funded by the Autonomous Community of Madrid. We also acknowledge support from Centro de Computación Cientı́fica-Universidad Autónoma de Madrid (CCC-UAM).

We would also like to thank Daniel Fernández-Sánchez for his implementation of the MLP problem generator, which was used as the basis for the real MLP experiments included in this repository.

## License

See the `LICENSE` file for details.
