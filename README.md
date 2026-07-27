# Improved Max-value Entropy Search

This repository contains the code developed for the Bachelor's Thesis "Optimización Bayesiana mediante Entropía del Máximo Valor Mejorada", authored by David Ángel Valenzuela Sánchez under the supervision of Daniel Hernández-Lobato at the Universidad Autónoma de Madrid.

The project implements and compares BO strategies based on max-value entropy. In particular, it includes:

* **MES**, based on a Gaussian truncation approximation.
* **MES+**, the proposed method, which replaces the local Gaussian truncation approximation with a modified variational sparse GP.
* **Random search**.

The experiments are performed on synthetic black-box optimization problems sampled from GP priors, benchmark functions and real MLP hyperparameter optimization problems. The objective is to compare the behaviour of these acquisition functions inside a BO loop.

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
* `code/my_acquisition_mes.py`: MES acquisition function based on Gaussian truncation.
* `code/my_acquisition_mes_plus.py`: MES+ acquisition function based on the proposed variational approximation.
* `code/modified_vfe_sparse_gp.py`: implementation of the modified VFE sparse Gaussian process used by MES+.
* `code/my_utils.py`: utility functions for fitting Gaussian processes, sampling candidate optima, conditioning models and computing predictive quantities.
* `code/util.py`: general utilities for reading configurations, managing paths, setting random seeds and transforming MLP hyperparameters.
* `code/synthetic_problem.py`: generation of synthetic objective functions sampled from a GP prior.
* `code/benchmark_problem.py`: implementation of the Shekel and Michalewicz benchmark problems.
* `code/mlp_real_problem.py`: definition of the real MLP hyperparameter optimization problems using the Australian Credit Approval and Ionosphere datasets.
* `code/experiments_generator.py`: generation of configurations for synthetic and benchmark experiments.
* `code/experiments_mlp_generator.py`: generation of configurations for the real MLP experiments.
* `code/experiments_time_generator.py`: generation of configurations for the 4D timing experiments.
* `code/experiments_plotter.py`: aggregation and plotting of synthetic and benchmark results.
* `code/experiments_mlp_plotter.py`: aggregation and plotting of the real MLP experiment results.
* `code/experiments_samples_generator.py`: aggregation and plotting of experiments with different numbers of optimum samples.
* `code/time_results.py`: summary of the results obtained in the timing experiments.
* `code/tests/`: one-dimensional validation scripts for the MES and MES+ approximations and acquisition functions.

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

The dependencies are listed in `requirements.txt`.

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

The virtual environment `venv/` should not be committed to the repository.

## Running tests

The one-dimensional validation scripts are included in `code/tests/`. They are useful for checking the behaviour of the MES and MES+ acquisition functions and the approximations used to compute them.

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

Benchmark configurations also contain the `benchmark` field, while real MLP configurations contain the `dataset`, `n_folds` and `n_repeats` fields.

## Generating batches of experiments

The script `code/experiments_generator.py` creates folders and configuration files for synthetic and benchmark experiments. It reads a base configuration from `code/config.json` and generates one configuration per experiment and method.

Before running it, check the constants at the top of the file:

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
* `BENCHMARK`: benchmark to use when `PROBLEM_TYPE` is `"benchmark"`.
* `DIMENSIONS`: dimensions used in the experiments.
* `START_EXP` and `END_EXP`: range of experiment identifiers.
* `BO_ITERS`: number of BO iterations.
* `NUM_SAMPLES_SOLUTION`: number of posterior optimum samples used by the acquisition functions.

For example, a synthetic experiment can have the following structure:

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

## Timing experiments

The script `code/experiments_time_generator.py` generates ten configurations for timing MES, MES+ and random search on a 4D synthetic problem.

Generate the configurations with:

```bash
cd code
python3 experiments_time_generator.py
```

When `measure_time` is enabled, `my_loop_BO.py` stores the execution time of each BO iteration in a file named:

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

The plots are saved in a `generate_plot/` folder inside the corresponding experiment directory.

The synthetic and benchmark plotters compute the log-relative difference with respect to the best available value and use bootstrap estimates to represent the uncertainty of the mean curves. The real MLP plotter represents validation accuracy.

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

Daniel Hernández-Lobato and David Ángel Valenzuela Sánchez

## Acknowledgements

We would like to thank Daniel Fernández Sánchez for his implementation of the MLP problem generator, which was used as the basis for the real MLP experiments included in this repository.

## License

See the `LICENSE` file for details.
