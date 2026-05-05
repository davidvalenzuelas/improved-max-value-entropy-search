#!/usr/bin/env python3
# coding: utf-8
"""
This file includes utilities, useful when working with GPs.

It implements helper functions for fitting the base GP,
sampling and conditioning on candidate optima, computing predictive
moments and comparing gaussian approximations used across experiments.
It also includes mathematical utilities for working with the normal
distribution.

Authors: Daniel Hernández Lobato, David Valenzuela Sánchez
"""
import torch
import math

from botorch.models.gp_regression import SingleTaskGP
from botorch.fit import fit_gpytorch_mll
from botorch.acquisition.utils import get_optimal_samples
from gpytorch.mlls.exact_marginal_log_likelihood import ExactMarginalLogLikelihood


# Mathematical utilities, useful when working with the normal distribution
def normal_cdf(z: torch.Tensor) -> torch.Tensor:
    """ This function computes the CDF of the standard normal distribution at z, using
    the error function."""
    # phi(z) = 0.5 * (1 + erf(z/sqrt(2)))
    return 0.5 * (1.0 + torch.erf(z / math.sqrt(2.0)))


def normal_pdf(z: torch.Tensor) -> torch.Tensor:
    """This function computes the standard normal pdf"""
    # phi(z) = exp(-0.5 * z^2) / sqrt(2*pi)
    return torch.exp(-0.5 * z.pow(2)) / math.sqrt(2.0 * math.pi)


# Helpers for fitting, sampling and conditioning GPs
def fit_singletask_gp(train_X: torch.Tensor, train_Y: torch.Tensor,
    init_noise: float) -> SingleTaskGP:
    """ This function fits a SingleTaskGP used as base GP for initializing
    the modified VFE sparse GP and for sampling y* from its posterior"""
    # Ensures correct dimensions and types for the training data
    train_X = train_X.double()
    train_Y = train_Y.double()
    if train_X.ndim == 1:
        train_X = train_X.unsqueeze(-1)
    if train_Y.ndim == 1:
        train_Y = train_Y.unsqueeze(-1)
    
    # Builds and fits the SingleTaskGP on the training data
    base_gp = SingleTaskGP(train_X, train_Y, outcome_transform=None)
    base_gp = base_gp.to(dtype=train_X.dtype, device=train_X.device)
    
    # Uses a tiny fixed observation noise
    base_gp.likelihood.noise = torch.as_tensor(init_noise,
        dtype=train_X.dtype, device=train_X.device)
    base_gp.likelihood.noise_covar.raw_noise.requires_grad_(False)
    
    # Exact marginal log likelihood for fitting the model
    mll = ExactMarginalLogLikelihood(base_gp.likelihood, base_gp)
    fit_gpytorch_mll(mll)
    
    # Sets model and likelihood to eval mode for posterior sampling
    base_gp.eval()
    base_gp.likelihood.eval()
    # Returns the fitted base GP
    return base_gp


def sample_solution_outputs_from_model(base_gp, bounds,
    num_samples: int = 512, seed_posterior_samples: int = 1):
    """This function samples posterior optimal pairs (x*, y*) from the
    GP model."""
    torch.manual_seed(seed_posterior_samples)
    
    # Samples optimal inputs/outputs from the posterior of the base GP 
    # within the given bounds
    optimal_inputs, optimal_outputs = get_optimal_samples(model=base_gp,
        bounds=bounds,num_optima=num_samples)
    
    # Reshapes and detaches the sampled optima
    sampled_x_stars = optimal_inputs.reshape(num_samples, -1).squeeze(-1).detach()
    sampled_y_stars = optimal_outputs.reshape(num_samples).detach()
    # Returns the sampled x* and y* values
    return sampled_x_stars, sampled_y_stars


@torch.no_grad()
def choose_y_star(sampled_x_stars: torch.Tensor, sampled_y_stars: torch.Tensor,
    seed_star_selection: int = 4):
    """ This function selects one y* value from the sampled posterior optima"""
    # Number of sampled optima
    n = sampled_y_stars.numel()
    
    # Makes the selection of the pair (x*,y*) reproducible
    g = torch.Generator(device=sampled_y_stars.device)
    g.manual_seed(seed_star_selection)
    chosen_idx = torch.randint(low=0, high=n, size=(1,), generator=g).item()
    
    return {
        "chosen_idx": int(chosen_idx), 
        "x_star": float(sampled_x_stars[chosen_idx].item()),
        "y_star": float(sampled_y_stars[chosen_idx].item()),    
        "num_samples": int(n)
    }


@torch.no_grad()
def condition_base_gp_on_optimum(base_gp: SingleTaskGP, x_star: float, y_star: float):
    """This function conditions the base GP on the selected optimal pair (x*,y*),
    obtaining a new GP posterior which incorporates this informtion"""
    dtype = next(base_gp.parameters()).dtype
    device = next(base_gp.parameters()).device
    
    # Converts x_star and y_star to tensors with correct dtype/device
    x_star_t = torch.tensor([[x_star]], dtype=dtype, device=device)
    y_star_t = torch.tensor([[y_star]], dtype=dtype, device=device)
    
    # Sets model and likelihood to eval mode
    base_gp.eval()
    base_gp.likelihood.eval()
    
    # Computes the posterior of the base GP at x_star, which will be used for conditioning
    _ = base_gp.posterior(x_star_t, observation_noise=False)
    
    # Conditions the base GP on the observation (x*,y*)
    # We want to  work with p(f|D,x*,y*) which is the same as p(f|D) conditioned on (x*,y*)
    conditioned_gp = base_gp.condition_on_observations(X=x_star_t, Y=y_star_t)
    conditioned_gp = conditioned_gp.to(dtype=dtype, device=device)
    # Sets the conditioned GP to eval mode
    conditioned_gp.eval()
    conditioned_gp.likelihood.eval()
    
    # Returns the conditioned GP and the tensors for x* and y*
    return conditioned_gp, x_star_t, y_star_t


# Helper for computing the predictive distribution and moments of a model
@torch.no_grad()
def get_predictive_distribution(model, likelihood, grid: torch.Tensor,
    observation_noise: bool = False):
    """ This function computes the predictive distribution of a model
    on the given grid"""
    # If the model has a posterior method, uses it to compute the predictive
    # distribution on the grid
    if hasattr(model, "posterior"):
        return model.posterior(grid, observation_noise=observation_noise)
    
    from modified_vfe_sparse_gp import sparse_predictive_distribution
    # Otherwise, uses the helper function for computing it
    return sparse_predictive_distribution(
        model, likelihood, grid, observation_noise=observation_noise)


@torch.no_grad()
def marginal_mean_variance(model, likelihood, grid: torch.Tensor,
    observation_noise: bool = False):
    """This function returns the pointwise predictive mean and variance
    of a model on the given grid"""
    # Computes the predictive distribution on the grid
    post = get_predictive_distribution(model, likelihood, grid, observation_noise=observation_noise)
    
    # Extracts the mean and the variance, and returns them
    mean = post.mean
    if mean.ndim > 1 and mean.shape[-1] == 1:
        mean = mean.squeeze(-1)
    variance = post.variance.clamp_min(1e-12)
    if variance.ndim > 1 and variance.shape[-1] == 1:
        variance = variance.squeeze(-1)
        
    return mean, variance


@torch.no_grad()
def extract_noise_variance(model, likelihood, grid: torch.Tensor) -> torch.Tensor:
    """This functions obtains the noise variance of a model on the given grid"""
    # Computes the predictive variance with and without observation noise
    _, var_f = marginal_mean_variance(model, likelihood, grid, observation_noise=False)
    _, var_y = marginal_mean_variance(model, likelihood, grid, observation_noise=True)
    # Returns the difference between those two variances
    return (var_y - var_f).clamp_min(1e-12)


# Helpers for computing truncated predictive moments and probabilities
@torch.no_grad()
def truncated_upper_normal_moments(mean: torch.Tensor, variance: torch.Tensor,
    upper: torch.Tensor):
    """This function computes the mean and variance of a truncated normal distribution
    with an upper truncation point, ie, calculates moments of X|X<upper where
    X~N(mean, variance)"""
    variance = variance.clamp_min(1e-12)
    std = variance.sqrt()   
    # Computes the standardized truncation point
    beta = (upper - mean) / std
    
    # Computes the pdf and cdf of the standard normal at the truncation point
    Phi = normal_cdf(beta).clamp_min(1e-12)
    phi = normal_pdf(beta)
    # Computes the lambda term used in the truncated normal moments formulas
    lam = phi / Phi
    
    # Computes the truncated mean and variance using the formulas for upper truncation
    mean_trunc = mean - std *lam
    var_trunc = variance * (1.0 - beta * lam - lam.pow(2)).clamp_min(1e-12)
    return mean_trunc, var_trunc


@torch.no_grad()
def upper_truncated_predictive_moments(model, likelihood, grid: torch.Tensor,
    y_star: float | torch.Tensor, observation_noise: bool = False):
    """ This function computes the predictive mean and variance after
    upper truncation at y*"""
    # Obtains the marginal predictive mean and variance of the model on the grid
    mean_f, var_f = marginal_mean_variance(model, likelihood, grid,
        observation_noise=False)
    
    # Converts y_star to a tensor with correct dtype/device
    y_star_t = torch.as_tensor(y_star, dtype=mean_f.dtype, device=mean_f.device)
    # Computes the truncated mean and variance using the formulas for upper truncation
    mean_trunc, var_trunc = truncated_upper_normal_moments(mean_f, var_f, y_star_t)
    
    # If there is not observation noise, returns the truncated mean and variance directly
    if not observation_noise:
        return mean_trunc, var_trunc
    
    # If not, adds the noise variance to the truncated variance before returning
    noise_var = extract_noise_variance(model, likelihood, grid)
    return mean_trunc, var_trunc + noise_var


# Helpers for computing and summarizing acquisition functions
@torch.no_grad()
def gaussian_entropy_reduction_acq(prior_variance: torch.Tensor, 
    conditioned_variance: torch.Tensor) -> torch.Tensor:
    """This function approximates the acquisition function by the
    reduction in the entropy of a normal distribution"""
    if prior_variance.ndim > 1 and prior_variance.shape[-1] == 1:
        prior_variance = prior_variance.squeeze(-1)
    if conditioned_variance.ndim > 1 and conditioned_variance.shape[-1] == 1:
        conditioned_variance = conditioned_variance.squeeze(-1)
        
    # Clamps both variances to avoid numerical issues
    prior_variance = prior_variance.clamp_min(1e-12)
    conditioned_variance = conditioned_variance.clamp_min(1e-12)
    
    # Calculates the reduction in the entropy of a normal distribution
    # This is 1/2 * log(prior_variance/conditioned_variance))
    return 0.5 * (torch.log(prior_variance) - torch.log(conditioned_variance))


@torch.no_grad()
def summarize_acquisition_curve(name: str, x_grid: torch.Tensor,
    acq: torch.Tensor):
    """This function summarizes the statistics of an acquisition curve"""
    # Finds the mean and max of the acquisition values, and the x value where the maximum is
    # obtained
    
    x_flat = x_grid.squeeze(-1).reshape(-1)

    if acq.ndim > 1 and acq.shape[-1] == 1:
        acq = acq.squeeze(-1)
    acq_flat = acq.reshape(-1)
    
    idx = torch.argmax(acq_flat)
    
    print(
        f"{name}: "
        f"max={acq_flat.max().item():.6f}, "
        f"x_argmax={x_flat[idx].item():.6f}"
    )


# Useful helpers for comparing probabilities and showing differences between approximations
@torch.no_grad()
def gaussian_prob_less_than(mean: torch.Tensor, variance: torch.Tensor, threshold: torch.Tensor):
    """This function computes the gaussian probability P(X<threshold) where X~N(mean, variance)"""
    std = variance.clamp_min(1e-12).sqrt()
    z = (threshold - mean) / std
    return normal_cdf(z)


@torch.no_grad()
def summarize_prob_less_than(model, likelihood, X: torch.Tensor, y_star: torch.Tensor):
    """This function summarizes the gaussian probability P(f(X)<y*) under a gaussian predictive distribution."""
    # Obtains the marginal predictive mean and variance of the model on X
    mean, variance = marginal_mean_variance(model, likelihood, X)
    # Computes the probability P(f(X)<y*) under the gaussian predictive distribution
    p_less = gaussian_prob_less_than(mean, variance, y_star)
    # Returns the mean, min and max of that probability across the points in X
    return p_less.mean().item(), p_less.min().item(), p_less.max().item()


@torch.no_grad()
def summarize_prob_less_than_from_moments(mean: torch.Tensor, variance: torch.Tensor, y_star: torch.Tensor):
    """Summarizes P(X<y*) from provided gaussian moments."""
    # Computes P(X<y*) under the gaussian distribution with the provided mean and variance
    p_less = gaussian_prob_less_than(mean, variance, y_star)
    # Returns the mean, min and max of that probability across the points in X
    return p_less.mean().item(), p_less.min().item(), p_less.max().item()


@torch.no_grad()
def format_diff_stats(name_a: str, mean_a: torch.Tensor, var_a: torch.Tensor,
    name_b: str, mean_b: torch.Tensor, var_b: torch.Tensor):
    """This function prints the mean/variance differences between two predictive
    distributions."""
    diff_mean = (mean_a - mean_b).abs()
    diff_var = (var_a - var_b).abs()
    print(f"\n{name_a} vs {name_b} on the grid:")
    print(f"Mean abs diff: {diff_mean.mean().item():.6f}")
    print(f"Max abs diff: {diff_mean.max().item():.6f}")
    print(f"Var abs diff: {diff_var.mean().item():.6f}")
    print(f"Max var diff: {diff_var.max().item():.6f}")
