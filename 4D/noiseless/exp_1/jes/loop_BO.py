#!/usr/bin/env python3
# coding: utf-8

# # Information-theoretic acquisition functions

# We present a simple example in one-dimension with one objective
# to illustrate the use of these acquisition functions.
# We first define the objective function.


# Arguments.
# First argument: Seed of the problem. Default=42
# Second argument: GP Lengthscale. Default=0.25
# Third argument: Alpha. Default=0.9999
# Fourth argument: Num_samples. Default=512

#Example of invocation.
# python res_synthetic_problem.py 42 0.25 0.9999 512


import os
import sys
import json
import gpytorch
import matplotlib.pyplot as plt
import torch
import numpy as np
import scipy as sp
import math

from botorch.utils.sampling import draw_sobol_samples
from botorch.models.transforms.outcome import Standardize
from botorch.models.gp_regression import SingleTaskGP
from gpytorch.mlls.exact_marginal_log_likelihood import ExactMarginalLogLikelihood
from botorch.fit import fit_gpytorch_mll
from botorch.sampling.pathwise.prior_samplers import draw_kernel_feature_paths
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
from synthetic_problem import Synthetic_problem
from vfe_sparse_gp import fit_model_vfe_sparse, pack_state_dict
from vfe_sparse_gp import as_botorch_model

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
        return -1.0 * model(torch.from_numpy(x).double()).mean.detach().numpy()


    grid = torch.rand(SIZE_GRID * num_dims, num_dims)
    vals = f(grid.numpy())

    input_sol = grid.numpy()[ vals.argmin() ]
    output_sol = vals.max()

    # We find the actual minimum using LBFGS with diff gradient approximation. 

    result = sp.optimize.fmin_l_bfgs_b(f, input_sol, None, bounds = [ (0,1) ] * num_dims, approx_grad = True)

    return torch.from_numpy(result[ 0 ])


# This function fits the model to the data, using a variational sparse GP with Adam optimizer
# It allows warm-starting from previous state dict and also allows to choose between a Gaussian
# likelihood (with noise) and a noiseless likelihood (fixed noise)
def fit_model(train_X, train_Y, state_dict=None, median_for_ls: bool = False):
    
    # Initial hyperparameters for the GP model
    M = 64 # inducing points
    training_iter = 500 # ADAM steps
    lr = 0.01
    
    # Fixed tiny noise for numerical stability in the noiseless likelihood
    noise_eps = 1e-6
    
    # Calls the VFE sparse GP fitting function, which returns the trained model and likelihood
    model, likelihood = fit_model_vfe_sparse(train_X=train_X, train_Y=train_Y, state_dict=state_dict,
        M=M, training_iter=training_iter, lr=lr, noise_eps=noise_eps, verbose=False,)
    
    # Attaches likelihood to the model
    model.likelihood = likelihood
    return model


def get_optimal_samples_grid_mc(model, bounds: torch.Tensor, num_optima: int, num_grid: int = 4096):
    d = bounds.shape[-1]

    # Importante: que bounds sea double
    bounds = bounds.double()

    # Grid Sobol en double
    X = draw_sobol_samples(bounds=bounds, n=num_grid, q=1).squeeze(1).double()

    model.eval()
    if hasattr(model, "likelihood"):
        model.likelihood.eval()

    with torch.no_grad():
        post = model(X)

        # --- FIX dtype mismatch ---
        # base_samples debe tener el mismo dtype/device que la media del posterior
        base_samples = torch.randn(
            num_optima, X.shape[0],
            device=post.mean.device,
            dtype=post.mean.dtype,
        )

        # Si le pasas base_samples, rsample ya no crea float por dentro
        samples = post.rsample(base_samples=base_samples)  # (num_optima, num_grid)
        # --- FIN FIX ---

        idx = samples.argmax(dim=-1)
        optimal_inputs = X[idx, :]
        optimal_outputs = samples[torch.arange(num_optima, device=X.device), idx].unsqueeze(-1)

    return optimal_inputs, optimal_outputs


def main():
    if len(sys.argv) != 2:
        print("Usage: python synthetic_problem.py <config.json>")
        sys.exit(1)
    
    config = read_config(sys.argv[1])
    seed = int(config["random_seed"])
    ls_model = float(config["lenghtscale_model_synthetic_problem"])
    num_initial_obs = int(config["num_initial_obs"])
    num_samples = int(config["num_samples_solution"])
    num_dims = 4 #int(len(config["variables"].keys()))
    BO_iters =  int(config["BO_iters"])
    acquisition_name = config["acquisition"]
    
    create_path(config["file_results"])
    
    reset_random_state(seed)
    
    # We asume [0,1]^4 as the box where optimization takes place
    bounds = torch.tensor([[0.0] * num_dims, [1.0] * num_dims])
    bounds = bounds.double()
    
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
    
    # We asume [0,1]^4 as the box where optimization takes place. We generate initial observations there.
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
            prev_state = pack_state_dict(model, model.likelihood)
            model = fit_model(x_observations, y_values_obs.detach(), state_dict=prev_state)
        else:
            model = fit_model(x_observations, y_values_obs.detach(), state_dict=None)
        
        print("Sanity check posterior (latent):")
        with torch.no_grad():
            post = model(x_observations)
            print("  mean[:5] =", post.mean[:5].cpu().numpy())
            print("  var[:5]  =", post.variance[:5].cpu().numpy())
            
        print("Sanity check posterior (observational via likelihood):")
        with torch.no_grad():
            post_obs = model.likelihood(model(x_observations))
            print("  obs mean[:5] =", post_obs.mean[:5].cpu().numpy())
            print("  obs var[:5]  =", post_obs.variance[:5].cpu().numpy())
            
        # Find the maximum of the acq
        print("Optimizing Acquisition")
        optimal_inputs, optimal_outputs = get_optimal_samples_grid_mc(model, bounds=bounds, num_optima=num_samples, num_grid=4096)
        # optimal_inputs, optimal_outputs = get_optimal_samples(model, bounds=bounds, num_optima=num_samples)
        
        if acquisition_name == "JES":
            alpha = 0.0
            
            model_botorch = as_botorch_model(model)
            acq = qJointEntropySearch(
                model=model_botorch,
                optimal_inputs=optimal_inputs.double(),
                optimal_outputs=optimal_outputs.double(),
                estimation_type="LB",
                condition_noiseless=True,
            )
            
            candidate, acq_value = optimize_acqf(acq_function=acq, bounds=bounds, q=1, num_restarts=num_restarts_opt, raw_samples=raw_samples_opt_acq)
            
        elif acquisition_name == "RES":
            alpha = float(config["alpha"])
            acq = qRenyiEntropySearch(model=model, optimal_inputs=optimal_inputs.double(), \
                    optimal_outputs=optimal_outputs.double(), alpha=alpha)
            
            candidate, acq_value = optimize_acqf(acq_function=acq, bounds=bounds, q=1, num_restarts=num_restarts_opt, raw_samples=raw_samples_opt_acq)
            
        elif acquisition_name == "RES_ENS":
            
            l_max_acqs = []
            alphas = [0.001, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.999] # config["alphas"]
            for alpha in alphas:
                acq = qRenyiEntropySearch(model=model, optimal_inputs=optimal_inputs.double(), \
                        optimal_outputs=optimal_outputs.double(), alpha=alpha)
                
                candidate, acq_value = optimize_acqf(acq_function=acq, bounds=bounds, q=1, num_restarts=num_restarts_opt, raw_samples=raw_samples_opt_acq)
                
                l_max_acqs.append(acq_value)
                
            acq = qRenyiEntropySearchEnsemble(model=model, optimal_inputs=optimal_inputs.double(), \
                    optimal_outputs=optimal_outputs.double(), alphas=alphas, weights_alphas=l_max_acqs)
            
            candidate, acq_value = optimize_acqf(acq_function=acq, bounds=bounds, q=1, num_restarts=num_restarts_opt, raw_samples=raw_samples_opt_acq)
            
        elif acquisition_name == "RES_Hedge":
            alphas = [0.001, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.999] # config["alphas"]
            
            hres = qRenyiEntropySearchHedge(model=model,
                                        filename_rewards="vector_of_rewards_hres.txt", \
                                        filename_candidates="d_candidates.dat", \
                                        filename_selected_alphas="selected_alphas.txt", \
                                        optimal_inputs=optimal_inputs.double(), \
                                        optimal_outputs=optimal_outputs.double(), \
                                        alphas=alphas, num_iters=BO_iters)
            
            candidate, acq_value = hres.optimize_acqf(bounds=bounds, q=1, num_restarts_opt=num_restarts_opt, raw_samples_opt_acq=raw_samples_opt_acq)
            
        print(f"{acquisition_name}: candidate={candidate}, acq_value={acq_value}")
        
        # We obtain the recommendation removing noise from the observations
        # The recommendation is the best observed value
        
        print("Computing recommendations.")
        
        recommendation = x_observations[ model(x_observations).mean.argmax() : (model(x_observations).mean.argmax() + 1), : ]
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

