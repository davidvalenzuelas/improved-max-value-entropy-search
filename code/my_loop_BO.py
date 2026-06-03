#!/usr/bin/env python3
# coding: utf-8

import os
import sys
import gpytorch
import matplotlib.pyplot as plt
import torch
import numpy as np
from botorch.models.transforms.outcome import Standardize
from botorch.models.gp_regression import SingleTaskGP
from gpytorch.mlls.exact_marginal_log_likelihood import ExactMarginalLogLikelihood
from botorch.fit import fit_gpytorch_mll
from botorch.acquisition.utils import get_optimal_samples
from botorch.optim import optimize_acqf

from util import reset_random_state, preprocess_outputs, read_config, create_path
from plotter import Plotter
import scipy as sp

from synthetic_problem import Synthetic_problem
from my_acquisition_mes_plus import MyAcquisition as MyAcquisitionMESPlus
from my_acquisition_mes import MyAcquisition as MyAcquisitionMES

SCALE_ACQ_VALS = True
RESOLUTION = 20
SIZE_GRID = 10000
# num_restarts_opt = 1
# raw_samples_opt_acq = 200
num_restarts_opt = 5
raw_samples_opt_acq = 512

# This function finds the optimal solution of the problem

def get_maximum_problem(num_dims, problem):

    grid = torch.rand(SIZE_GRID * num_dims, num_dims)

    vals = problem(grid)

    input_sol = grid[ vals.argmax() ]
    output_sol = vals.max()

    # We find the actual minimum using LBFGS with diff gradient approximation. 

    def f(x):
        if len(x.shape) == 1:
            x = x.reshape((1, x.shape[ 0 ]))
        return -1.0 * problem(torch.from_numpy(x).double()).numpy()[0,0]

    result = sp.optimize.fmin_l_bfgs_b(f, input_sol.numpy(), None, bounds = [ (0,1) ] * num_dims, approx_grad = True)

    input_sol = torch.from_numpy(result[ 0 ]).double()
    output_sol = problem(input_sol.reshape((1, num_dims)))[0,0]

    return input_sol, output_sol

# This function optimizes the posterior mean

def optimize_posterior_mean(model, num_dims):

    def f(x):
        if len(x.shape) == 1:
            x = x.reshape((1, x.shape[ 0 ]))
        X = torch.from_numpy(x).double()
        means = []
        model.eval()
        model.likelihood.eval()
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            for i in range(0, X.shape[ 0 ], 1024):
                post = model.posterior(X[ i : i + 1024 ], observation_noise = False)
                means.append(post.mean.detach().reshape(-1).cpu())
        vals = torch.cat(means, dim = 0).numpy()
        if vals.shape[ 0 ] == 1:
            return -1.0 * float(vals[ 0 ])
        return -1.0 * vals


    grid = torch.rand(SIZE_GRID * num_dims, num_dims)
    vals = f(grid.numpy())

    input_sol = grid.numpy()[ vals.argmin() ]
    output_sol = vals.max()

    # We find the actual minimum using LBFGS with diff gradient approximation. 

    result = sp.optimize.fmin_l_bfgs_b(f, input_sol, None, bounds = [ (0,1) ] * num_dims, approx_grad = True)

    return torch.from_numpy(result[ 0 ])

# This function fits the model using an early fit if available

def fit_model(train_X, train_Y, state_dict = None, likelihood_exp: str = "GAUSSIAN", median_for_ls: bool = False):

    if likelihood_exp == "GAUSSIAN":
        model = SingleTaskGP(train_X, train_Y, outcome_transform=None)

    else:
        model = SingleTaskGP(train_X, train_Y, outcome_transform=None)
        model.likelihood.noise = 1e-6
        model.likelihood.noise_covar.raw_noise.requires_grad_(False)

    if median_for_ls:
        model.covar_module.base_kernel.lengthscale = get_init_lengthscale(train_X)

    # We use the previously trained model, if available

    if state_dict is not None:
        model.load_state_dict(state_dict)

    mll = ExactMarginalLogLikelihood(model.likelihood, model)
    fit_gpytorch_mll(mll)

    return model

def main():
    if len(sys.argv) != 2:
        print("Usage: python synthetic_problem.py <config.json>")
        sys.exit(1)
    
    config = read_config(sys.argv[1])
    
    seed = int(config["random_seed"])
    ls_model = float(config["lenghtscale_model_synthetic_problem"])
    num_initial_obs = int(config["num_initial_obs"])
    num_samples = 1
    num_dims = int(config.get("num_dims", 4))
    BO_iters =  int(config["BO_iters"])
    M_extra = int(config.get("M_extra", 100))
    num_repetitions = int(config.get("num_repetitions", 10))
    acquisition_names = config.get("acquisitions", ["MES+", "MES", "RANDOM"])

    create_path(config["file_results"])

    # XXX We asume 0, 1 as the box where optimization takes place

    bounds = torch.tensor([[0.0] * num_dims, [1.0] * num_dims]).double()

    all_metrics = np.zeros((len(acquisition_names), num_repetitions, BO_iters))

    for rep in range(num_repetitions):

        rep_seed = seed + rep
        reset_random_state(rep_seed)

        if num_dims == 2:
            plotter = Plotter(num_dims=num_dims, bounds=bounds, resolution=RESOLUTION, path=config["file_results"])

        # We now generate some data and then fit the Gaussian process model.

        synthetic_problem = Synthetic_problem(num_dims=num_dims, lengthscale_model=ls_model, seed=rep_seed)
        problem = synthetic_problem.f
        problem_noiseless = synthetic_problem.f
        x_max_val_problem, max_val_problem = get_maximum_problem(num_dims=num_dims, problem=problem)

        path_rep = f'{config["file_results"]}/rep_{rep}'
        os.makedirs(path_rep, exist_ok=True)

        # We save the optimum (x and y) to a file

        np.savetxt(path_rep + "/x_optimum_problem.txt", x_max_val_problem.detach().numpy().reshape((1, num_dims)))
        np.savetxt(path_rep + "/y_optimum_problem.txt", np.array([ max_val_problem.detach().numpy() ]))

        # XXX We asume 0, 1 as the box where optimization takes place. We generate initial observations there.

        x_observations_initial = torch.rand(num_initial_obs, num_dims)
        y_values_obs_initial = problem(x_observations_initial).T
        x_observations_initial = x_observations_initial.double()
        y_values_obs_initial = y_values_obs_initial.double()

        for acq_index, acquisition_name in enumerate(acquisition_names):

            reset_random_state(rep_seed + 1000 * (acq_index + 1))
            path_results = f'{path_rep}/{acquisition_name.lower().replace("+", "plus")}'
            os.makedirs(path_results, exist_ok=True)

            x_observations = x_observations_initial.clone()
            y_values_obs = y_values_obs_initial.clone()
            model = None

            # START THE BO LOOP

            for bo_iteration in range(BO_iters):
               
                print(f"Rep {rep} - {acquisition_name} - BO Iteration number: {bo_iteration}")

                # We fit the model using the previous solution

                print("Fitting the model")
                if model is not None:
                    model = fit_model(x_observations, y_values_obs.detach(), model.state_dict(), likelihood_exp = "NOISELESS")
                else:
                    model = fit_model(x_observations, y_values_obs.detach(), likelihood_exp = "NOISELESS")
               
               # Find the maximum of the acq

                print("Optimizing Acquisition")

                if acquisition_name == "MES":
                    optimal_inputs, optimal_outputs = get_optimal_samples(model, bounds=bounds, num_optima=num_samples)
                    x_star = optimal_inputs.reshape(num_samples, num_dims)[0:1].double().detach()
                    y_star = optimal_outputs.reshape(num_samples)[0].double().detach()
                    acq = MyAcquisitionMES(model=model, x_star=x_star, y_star=y_star, M=M_extra,
                        lower_bound=bounds[0], upper_bound=bounds[1])
                    candidate, acq_value = optimize_acqf(acq_function=acq, bounds=bounds, q=1,
                        num_restarts=num_restarts_opt, raw_samples=raw_samples_opt_acq)

                elif acquisition_name == "MES+":
                    optimal_inputs, optimal_outputs = get_optimal_samples(model, bounds=bounds, num_optima=num_samples)
                    x_star = optimal_inputs.reshape(num_samples, num_dims)[0:1].double().detach()
                    y_star = optimal_outputs.reshape(num_samples)[0].double().detach()
                    acq = MyAcquisitionMESPlus(model=model, x_star=x_star, y_star=y_star, M=M_extra,
                        lower_bound=bounds[0], upper_bound=bounds[1])
                    candidate, acq_value = optimize_acqf(acq_function=acq, bounds=bounds, q=1,
                        num_restarts=num_restarts_opt, raw_samples=raw_samples_opt_acq)

                elif acquisition_name == "RANDOM":
                    candidate = torch.rand(1, num_dims).double()
                    acq_value = torch.tensor(float("nan"))

                else:
                    raise ValueError(f"Unknown acquisition: {acquisition_name}")

                print(f"{acquisition_name}: candidate={candidate}, acq_value={acq_value}")

                # We obtain the recommendation removing noise from the observations
                # The recommendation is the best observed value

                print("Computing recommendations.")

                model.eval()
                model.likelihood.eval()
                with torch.no_grad(), gpytorch.settings.fast_pred_var():
                    posterior_mean_obs = model.posterior(x_observations, observation_noise = False).mean.reshape(-1)
                recommendation = x_observations[ posterior_mean_obs.argmax() : (posterior_mean_obs.argmax() + 1), : ]
                objective_value_at_recommendation = problem_noiseless(recommendation.double())

                file_recommendations = open(f'{path_results}/recommendations_obs.txt', 'a')
                file_vals_rec = open(f'{path_results}/objective_at_recommendations_obs.txt', 'a')

                np.savetxt(file_recommendations, recommendation.detach().numpy())
                np.savetxt(file_vals_rec, objective_value_at_recommendation.detach().numpy())

                file_recommendations.close()
                file_vals_rec.close()

                # We obtain the recommendation from the observations

                recommendation = x_observations[ y_values_obs.argmax() : (y_values_obs.argmax() + 1), : ]
                objective_value_at_recommendation = problem_noiseless(recommendation.double())

                file_recommendations = open(f'{path_results}/recommendations_obs_obs.txt', 'a')
                file_vals_rec = open(f'{path_results}/objective_at_recommendations_obs_obs.txt', 'a')

                np.savetxt(file_recommendations, recommendation.detach().numpy()) 
                np.savetxt(file_vals_rec, objective_value_at_recommendation.detach().numpy())

                file_recommendations.close()
                file_vals_rec.close()

                # We obtain the recommendation by optimizing the posterior mean

                recommendation = optimize_posterior_mean(model, num_dims).reshape((1, num_dims))
                objective_value_at_recommendation = problem_noiseless(recommendation.double())

                metric = torch.log(torch.abs(objective_value_at_recommendation.reshape(-1)[0] - max_val_problem)
                    / torch.abs(max_val_problem).clamp_min(1e-12) + 1e-6)
                all_metrics[acq_index, rep, bo_iteration] = metric.detach().cpu().item()

                file_recommendations = open(f'{path_results}/recommendations_post_mean.txt', 'a')
                file_vals_rec = open(f'{path_results}/objective_at_recommendations_post_mean.txt', 'a')
                file_metric = open(f'{path_results}/metric_post_mean.txt', 'a')

                np.savetxt(file_recommendations, recommendation.detach().numpy()) 
                np.savetxt(file_vals_rec, objective_value_at_recommendation.detach().numpy())
                np.savetxt(file_metric, np.array([ metric.detach().cpu().item() ]))

                file_recommendations.close()
                file_vals_rec.close()
                file_metric.close()

                # We evaluate the objective at the selected point and add that to the training set

                print("Evaluating objective")
                value_cand = problem(candidate)[ 0 ]
                x_observations = torch.cat((x_observations, candidate), dim=0)
                y_values_obs = torch.cat((y_values_obs, value_cand[ None, : ]), dim=0)

                # We save the points evaluated so far

                with open(f'{path_results}/points_evaluated.txt', "w") as f:
                    np.savetxt(f, x_observations.numpy())

                with open(f'{path_results}/y_values_evaluated.txt', "w") as f:
                    np.savetxt(f, y_values_obs.numpy())

                sys.stdout.flush()

    mean_metrics = all_metrics.mean(axis=1)
    std_metrics = all_metrics.std(axis=1)
    np.savez(f'{config["file_results"]}/summary_metrics.npz',
        acquisition_names=np.array(acquisition_names), metrics=all_metrics,
        mean_metrics=mean_metrics, std_metrics=std_metrics)

    iterations = np.arange(BO_iters)
    plt.figure(figsize=(9, 6))
    for acq_index, acquisition_name in enumerate(acquisition_names):
        mean = mean_metrics[acq_index]
        std = std_metrics[acq_index]
        plt.plot(iterations, mean, label=acquisition_name)
        plt.fill_between(iterations, mean - std, mean + std, alpha=0.15)
    plt.xlabel("BO iteration")
    plt.ylabel("log(abs(y_recom - y_opt) / abs(y_opt) + 1e-6)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'{config["file_results"]}/bo_metric_mean.png', dpi=200)
    plt.show()

    print("End. Have a nice day!")


if __name__ == "__main__":

    tkwargs = {"dtype": torch.double, "device": "cpu"}

    main()
