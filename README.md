# Improved Max-value Entropy Search

This repository contains the code developed for the Bachelor's Thesis **"Optimización Bayesiana mediante Entropía del Máximo Valor Mejorada"**, authored by **David Ángel Valenzuela Sánchez** and **Daniel Hernández-Lobato** at the **Universidad Autónoma de Madrid**.

The project implements and compares Bayesian optimization strategies based on max-value entropy. In particular, it includes:

* **MES**, based on a Gaussian truncation approximation.
* **MES+**, the proposed method, which replaces the local Gaussian truncation approximation with a modified variational sparse Gaussian process.
* **Random search**, used as a baseline strategy.

The experiments are performed on synthetic black-box optimization problems sampled from Gaussian process priors. The objective is to compare the behaviour of these acquisition functions inside a Bayesian optimization loop.

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
│   ├── synthetic_problem.py
│   ├── experiments_generator.py
│   ├── experiments_plotter.py
│   └── ...
├── requirements.txt
├── README.md
├── LICENSE
├── CITATION.cff
└── .gitignore
```

The main files are:

* `code/my_loop_BO.py`: main BO loop for the synthetic problem.
* `code/my_acquisition_mes.py`: MES acquisition function based on Gaussian truncation.
* `code/my_acquisition_mes_plus.py`: MES+ acquisition function based on the proposed variational approximation.
* `code/modified_vfe_sparse_gp.py`: implementation of the modified VFE sparse Gaussian process used by MES+.
* `code/my_utils.py`: utility functions for fitting Gaussian processes, sampling candidate optima, conditioning models and computing predictive quantities.
* `code/synthetic_problem.py`: generation of synthetic objective functions sampled from a Gaussian process prior.
* `code/experiments_generator.py`: script used to generate experiment folders and configuration files for several repetitions, dimensions and acquisition functions.
* `code/experiments_plotter.py`: script used to aggregate experiment results and generate comparison plots.

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
* Pytest, only for running tests or import checks

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

If disk space is limited and GPU support is not needed, it is recommended to install the CPU version of PyTorch first:

```bash
pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu
pip install --no-cache-dir gpytorch botorch numpy scipy matplotlib pytest
```

The virtual environment `venv/` should not be committed to the repository.


## Running tests

Some tests are included inside `code/`. They are useful for checking the behaviour of the acquisition functions and the approximations used by MES and MES+. Run them from there:

```bash
cd code
python3 test_1d_5obs_acquisition_mes.py
```

or:

```bash
cd code
python3 test_1d_5obs_jes.py
```



## Configuration files

The BO loop is controlled through JSON configuration files. A typical configuration contains:

```json
{
  "random_seed": 123,
  "lenghtscale_model_synthetic_problem": 0.25,
  "num_initial_obs": 10,
  "num_dims": 4,
  "BO_iters": 20,
  "M_extra": 100,
  "num_repetitions": 5,
  "acquisitions": ["MES+", "MES", "RANDOM"],
  "file_results": "../results/example_experiment"
}
```

The most relevant parameters are:

* `random_seed`: seed used to initialize the synthetic problem and the optimization process.
* `lenghtscale_model_synthetic_problem`: lengthscale used to sample the synthetic objective function.
* `num_initial_obs`: number of initial observations before the BO loop starts.
* `num_dims`: input dimension of the synthetic problem.
* `BO_iters`: number of BO iterations.
* `M_extra`: number of extra inducing points used by the sparse approximation.
* `num_repetitions`: number of independent repetitions.
* `acquisitions`: list of acquisition strategies to compare. The supported values are `"MES+"`, `"MES"` and `"RANDOM"`.
* `file_results`: directory where the results will be stored.

## Generating batches of experiments

The script `code/experiments_generator.py` creates folders and configuration files for repeated experiments. It reads a base configuration from `code/config.json` and then generates one configuration per experiment and method.

Before running it, check the constants at the top of `experiments_generator.py`:

```python
DIMENSIONS = [4]
START_EXP = 1
END_EXP = 100
BO_ITERS = 100
ROOT_TEMPLATE = "{D}d_experiments"
```

These constants control:

* `DIMENSIONS`: dimensions to run, for example `[4]` or `[4, 6]`.
* `START_EXP` and `END_EXP`: range of experiment identifiers.
* `BO_ITERS`: number of BO iterations.
* `ROOT_TEMPLATE`: name of the root results folder for each dimension.

The script creates directories of the form:

```text
4d_experiments/
├── exp_1/
│   ├── config_mes.json
│   ├── config_mes_plus.json
│   ├── config_random.json
│   ├── mes/
│   │   └── results_synthetic_problem/
│   ├── mes_plus/
│   │   └── results_synthetic_problem/
│   └── random/
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

After generating the configurations, the experiments can be launched one by one. For example, for experiments 1 to 100 in 4 dimensions:

```bash
cd code

for EXP_ID in $(seq 1 100); do
    python my_loop_BO.py 4d_experiments/exp_${EXP_ID}/config_mes.json
    python my_loop_BO.py 4d_experiments/exp_${EXP_ID}/config_mes_plus.json
    python my_loop_BO.py 4d_experiments/exp_${EXP_ID}/config_random.json
done
```


## Plotting experiment results

The script `code/experiments_plotter.py` aggregates the results generated by the experiments and produces comparison plots.

Before running it, check the constants at the top of `experiments_plotter.py`:

```python
DIMENSIONS = [4]
NUM_EXPERIMENTS = 100
BO_ITERS = 100
N_BOOTSTRAP = 200
ROOT_TEMPLATE = "{D}d_experiments"
```

These constants must match the experiments that were actually run.

The plotter expects a directory structure like:

```text
4d_experiments/
├── exp_1/
│   ├── mes/
│   │   └── results_synthetic_problem/
│   ├── mes_plus/
│   │   └── results_synthetic_problem/
│   └── random/
│       └── results_synthetic_problem/
├── exp_2/
│   └── ...
└── ...
```

To generate plots, run:

```bash
cd code
python3 experiments_plotter.py
```

The plots will be saved in:

```text
4d_experiments/generate_plot/
```

or in the corresponding folder for the selected dimension.

The plotter computes the log-relative difference with respect to the best available value and uses bootstrap estimates to represent the uncertainty of the mean curves.

## Output files

During each run, the code stores several files with recommendations and objective values, including:

* `recommendations_obs.txt`
* `objective_at_recommendations_obs.txt`
* `recommendations_obs_obs.txt`
* `objective_at_recommendations_obs_obs.txt`
* `recommendations_post_mean.txt`
* `objective_at_recommendations_post_mean.txt`
* `metric_post_mean.txt`
* `points_evaluated.txt`
* `y_values_evaluated.txt`
* `x_optimum_problem.txt`
* `y_optimum_problem.txt`

The exact directory structure depends on the value of `file_results` and on whether multiple repetitions and acquisition functions are used.

## Citation

If you use this repository, please cite it as:

```bibtex
@software{valenzuelahernandezlobato2026improvedmes,
  author = {Valenzuela Sánchez, David Ángel and Hernández-Lobato, Daniel},
  title = {{Improved Max-value Entropy Search}},
  year = {2026},
  version = {1.0.0},
  url = {https://github.com/davidvalenzuelas/improved-max-value-entropy-search},
  note = {Code repository associated with the Bachelor's Thesis ``Optimización Bayesiana mediante Entropía del Máximo Valor Mejorada''}
}
```

## Authors

**David Ángel Valenzuela Sánchez**
**Daniel Hernández-Lobato**

## Thesis information

**Bachelor's Thesis:** *Optimización Bayesiana mediante Entropía del Máximo Valor Mejorada*
**Institution:** Universidad Autónoma de Madrid
**Year:** 2026

## License

See the `LICENSE` file for details.
