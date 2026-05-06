#!/usr/bin/env python3
# coding: utf-8

import os
import sys
import json
import gpytorch
import matplotlib.pyplot as plt
import torch
import numpy as np
from botorch.utils.sampling import draw_sobol_samples
from botorch.models.transforms.outcome import Standardize
from botorch.models.gp_regression import SingleTaskGP
from gpytorch.mlls.exact_marginal_log_likelihood import ExactMarginalLogLikelihood
from botorch.fit import fit_gpytorch_mll
from botorch.sampling.pathwise.prior_samplers import draw_kernel_feature_paths
import math
from botorch.sampling.pathwise.utils import GetTrainInputs
from typing import Any, Callable, Iterable, List, Optional, overload, Tuple, Union
from botorch.utils.dispatcher import Dispatcher
from torch import Tensor
from botorch.acquisition.utils import get_optimal_samples
from botorch.acquisition.predictive_entropy_search import qPredictiveEntropySearch
from botorch.acquisition.max_value_entropy_search import ( qLowerBoundMaxValueEntropy,)
from botorch.acquisition.joint_entropy_search import qJointEntropySearch
from renyi_entropy_search import qRenyiEntropySearch
from renyi_entropy_search_ensemble import qRenyiEntropySearchEnsemble
from renyi_entropy_search_hedge import qRenyiEntropySearchHedge
from botorch.optim import optimize_acqf

from util import reset_random_state, preprocess_outputs, read_config, create_path
from plotter import Plotter
import scipy as sp

from synthetic_problem import Synthetic_problem
from my_acquisition import MyAcquisition

SCALE_ACQ_VALS = True
RESOLUTION = 20
SIZE_GRID = 10000
num_restarts_opt = 1
raw_samples_opt_acq = 200

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
        model = SingleTaskGP(train_X, train_Y, outcome_transform=Standardize(m=1))

    else:
        model = SingleTaskGP(train_X, train_Y, outcome_transform=Standardize(m=1))
        model.likelihood.noise = 1e-4
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
    num_dims = 4#int(len(config["variables"].keys()))
    BO_iters =  int(config["BO_iters"])
    acquisition_name = "MY_ACQ"
    M_extra = int(config.get("M_extra_inducing", config.get("M", 100)))
    my_acq_style = str(config.get("my_acquisition_style", config.get("acquisition", "jes"))).lower()
    if my_acq_style not in ("mes", "jes"):
        my_acq_style = "jes"

    create_path(config["file_results"])

    reset_random_state(seed)

    # XXX We asume 0, 1 as the box where optimization takes place

    bounds = torch.tensor([[0.0] * num_dims, [1.0] * num_dims]).double()
 
    if num_dims == 2:
        plotter = Plotter(num_dims=num_dims, bounds=bounds, resolution=RESOLUTION, path=config["file_results"])

    # We now generate some data and then fit the Gaussian process model.

    synthetic_problem = Synthetic_problem(num_dims=num_dims, lengthscale_model=ls_model, seed=seed)
    problem = synthetic_problem.f
    problem_noiseless = synthetic_problem.f
    x_max_val_problem, max_val_problem = get_maximum_problem(num_dims=num_dims, problem=problem)

    # We save the optimum (x and y) to a file

    np.savetxt(config["file_results"] + "/x_optimum_problem.txt", x_max_val_problem.detach().numpy().reshape((1, num_dims)))
    np.savetxt(config["file_results"] + "/y_optimum_problem.txt", np.array([ max_val_problem.detach().numpy() ]))

    # XXX We asume 0, 1 as the box where optimization takes place. We generate initial observations there.

    x_observations = torch.rand(num_initial_obs, num_dims)
    y_values_obs = problem(x_observations).T
    
    x_observations = x_observations.double()
    y_values_obs = y_values_obs.double()

    model = None

    import os
    if os.path.exists("points_evaluated.txt"):
        x_observations = torch.from_numpy(np.loadtxt("points_evaluated.txt", ndmin = 2)).double()
        y_values_obs = torch.from_numpy(np.loadtxt("y_values_evaluated.txt", ndmin = 2)).double()
        BO_iters = BO_iters - (x_observations.shape[ 0 ] - num_initial_obs)

    # START THE BO LOOP

    for bo_iteration in range(BO_iters):
       
        print(f"BO Iteration number: {bo_iteration}")

        # We fit the model using the previous solution

        print("Fitting the model")
        if model is not None:
            model = fit_model(x_observations, y_values_obs.detach(), model.state_dict(), likelihood_exp = "NOISELESS")
        else:
            model = fit_model(x_observations, y_values_obs.detach(), likelihood_exp = "NOISELESS")
       
       # Find the maximum of the acq

        print("Optimizing Acquisition")

        optimal_inputs, optimal_outputs = get_optimal_samples(model, bounds=bounds, num_optima=num_samples)
        x_star = optimal_inputs.reshape(num_samples, num_dims)[0:1].double().detach()
        y_star = optimal_outputs.reshape(num_samples)[0].double().detach()

        acq = MyAcquisition(model=model, x_star=x_star, y_star=y_star, M=M_extra, style=my_acq_style, lower_bound=bounds[0], upper_bound=bounds[1])
        candidate, acq_value = optimize_acqf(acq_function=acq, bounds=bounds, q=1, num_restarts=num_restarts_opt, raw_samples=raw_samples_opt_acq)

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

        file_recommendations = open(f'{config["file_results"]}/recommendations_obs.txt', 'a')
        file_vals_rec = open(f'{config["file_results"]}/objective_at_recommendations_obs.txt', 'a')

        np.savetxt(file_recommendations, recommendation.detach().numpy())
        np.savetxt(file_vals_rec, objective_value_at_recommendation.detach().numpy())

        file_recommendations.close()
        file_vals_rec.close()

        # We obtain the recommendation from the observations

        recommendation = x_observations[ y_values_obs.argmax() : (y_values_obs.argmax() + 1), : ]
        objective_value_at_recommendation = problem_noiseless(recommendation.double())

        file_recommendations = open(f'{config["file_results"]}/recommendations_obs_obs.txt', 'a')
        file_vals_rec = open(f'{config["file_results"]}/objective_at_recommendations_obs_obs.txt', 'a')

        np.savetxt(file_recommendations, recommendation.detach().numpy()) 
        np.savetxt(file_vals_rec, objective_value_at_recommendation.detach().numpy())

        file_recommendations.close()
        file_vals_rec.close()

        # We obtain the recommendation by optimizing the posterior mean

        recommendation = optimize_posterior_mean(model, num_dims).reshape((1, num_dims))
        objective_value_at_recommendation = problem_noiseless(recommendation.double())

        file_recommendations = open(f'{config["file_results"]}/recommendations_post_mean.txt', 'a')
        file_vals_rec = open(f'{config["file_results"]}/objective_at_recommendations_post_mean.txt', 'a')

        np.savetxt(file_recommendations, recommendation.detach().numpy()) 
        np.savetxt(file_vals_rec, objective_value_at_recommendation.detach().numpy())

        file_recommendations.close()
        file_vals_rec.close()

        # We evaluate the objective at the selected point and add that to the training set

        print("Evaluating objective")
        value_cand = problem(candidate)[ 0 ]
        x_observations = torch.cat((x_observations, candidate), dim=0)
        y_values_obs = torch.cat((y_values_obs, value_cand[ None, : ]), dim=0)

        # We save the points evaluated so far

        with open("points_evaluated.txt", "w") as f:
            np.savetxt(f, x_observations.numpy())

        with open("y_values_evaluated.txt", "w") as f:
            np.savetxt(f, y_values_obs.numpy())

        sys.stdout.flush()

    print("End. Have a nice day!")


if __name__ == "__main__":

    tkwargs = {"dtype": torch.double, "device": "cpu"}

    main()

