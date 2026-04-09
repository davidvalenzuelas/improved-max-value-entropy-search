#!/usr/bin/env python3
# coding: utf-8
"""

"""
from __future__ import annotations

import numpy as np
import torch
import matplotlib.pyplot as plt

from modified_vfe_sparse_gp import (
    fit_vfe_sparse_gp,
    predictive_distribution,
    normal_cdf,
    VFESparseGP,
    build_init_dist_from_base_gp,
)

from botorch.models.gp_regression import SingleTaskGP
from botorch.fit import fit_gpytorch_mll
from botorch.acquisition.utils import get_optimal_samples
from gpytorch.mlls.exact_marginal_log_likelihood import ExactMarginalLogLikelihood

# Number of training points (observations) we keep from the sampled function
NUM_TRAIN = 4
# Multiplier for the standard deviation when plotting confidence bands
PLOT_STD_MULT = 1.0


def kernel(x: torch.Tensor, y: torch.Tensor, lengthscale: float = 2.0,
    variance: float = 1.0) -> torch.Tensor:
    """This function defines the RBF kernel used in this test"""
    # Scales inputs by lengthscale
    x_scaled = x / lengthscale
    y_scaled = y / lengthscale
    
    # Computes squared distance matrix
    sqdist = (x_scaled[:,None,:] - y_scaled[None,:,:]).pow(2).sum(dim=-1)
    # Returns the RBF kernel matrix
    return variance * torch.exp(-0.5 * sqdist)


@torch.no_grad()
def generate_4obs_problem(num_grid: int = 1000, jitter: float=1e-7,
    seed_latent: int = 123, seed_train: int = 2):
    """ This function generates the synthetic 1D problem used in the test.
    It samples a latent function from a GP with a RBF kernel on [-5,5]
    and selects 4 grid points uniformly at random as training points"""
    dtype = torch.float64
    # Creates a grid of points in [-5,5]
    x_grid = torch.linspace(-5.0, 5.0, num_grid, dtype=dtype).unsqueeze(-1)
    
    # Builds the GP covariance matrix on the grid
    sigma = kernel(x_grid, x_grid) + jitter * torch.eye(num_grid, dtype=dtype)
    # Cholesky decomposition for sampling from the GP prior
    L = torch.linalg.cholesky(sigma)
    
    # Draws one latent function sample
    g_latent = torch.Generator(device="cpu")
    g_latent.manual_seed(seed_latent)
    f_true = L @ torch.randn(num_grid, dtype=dtype, generator=g_latent)
    
    # Randomly selects training points from the grid
    rng = np.random.default_rng(seed_train)
    p_sel = np.sort(rng.choice(np.arange(num_grid),size=NUM_TRAIN,replace=False))
    p_sel = torch.tensor(p_sel, dtype=torch.long)
    
    # Extracts the training inputs and their outputs
    x_train = x_grid[p_sel].contiguous()
    y_train = f_true[p_sel].contiguous()
    
    return x_grid, f_true, x_train, y_train


@torch.no_grad()
def prob_f_below_y_star(model, Xc, y_star): 
    """ This function calculates the mean/min/max P(f(Xc)<y*) under the
    variational distribution of the model, where Xc are constraint points
    and y* is the threshold for the constraint"""
    model.eval()
    # Evaluates posterior at constraint points
    qf = model(Xc)
    # Obtains mean and std
    m = qf.mean
    s = qf.variance.clamp_min(1e-12).sqrt()
    # Standardizes and computes probabilities under gaussian posterior
    z = (y_star - m) / s
    p_less = normal_cdf(z)
    return p_less.mean().item(), p_less.min().item(), p_less.max().item()


@torch.no_grad()
def build_sparse_model_just_initialized(base_gp, inducing_points):
    """This function builds a sparse GP model with the same mean and covariance
    modules as the base GP, and with a variational distribution initialized
    from the posterior of the base GP at the inducing points"""
    
    # Builds variational distribution at inducing points from base GP posterior
    init_dist = build_init_dist_from_base_gp(base_gp, inducing_points)
    
    # Builds sparse GP model
    model = VFESparseGP(inducing_points=inducing_points, init_dist=init_dist,
        mean_module=base_gp.mean_module, covar_module=base_gp.covar_module)
    model = model.to(dtype=inducing_points.dtype, device=inducing_points.device)
    
    # Inducing points are fixed for this test
    model.variational_strategy.inducing_points.requires_grad_(False)
    return model


@torch.no_grad()
def obtain_mean_difference(res_std, res_con, x_grid):
    """This function obtains the mean difference between the constrained
    and the standard model on the grid"""
    # Computes predictive mean on the grid for both models
    mean_std = predictive_distribution(res_std.model, res_std.likelihood,
        x_grid).mean
    mean_con = predictive_distribution(res_con.model, res_con.likelihood,
        x_grid).mean
    
    # Computes and prints mean and max absolute difference
    diff = mean_con - mean_std
    print("Mean absolute difference:", diff.abs().mean().item())
    print("Max absolute difference:", diff.abs().max().item())


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
    base_gp = SingleTaskGP(train_X, train_Y)
    base_gp = base_gp.to(dtype=train_X.dtype, device=train_X.device)
    
    # Starts from a small noise level
    base_gp.likelihood.noise = torch.as_tensor(
        init_noise, dtype=train_X.dtype, device=train_X.device
    )
    
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
    seed_star_selection: int = 2):
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


# Plotter helpers
@torch.no_grad()
def plot_two_predictive_distributions(ax, model_a, likelihood_a, model_b,
    likelihood_b, x_grid, f_true, x_train, y_train, inducing_points, title,
    label_a="Base GP", label_b=" VFE sparse GP after init", y_star=None,
    x_star=None):
    
    """This function plots two predictive distributions for comparison"""
    # Computes predictive distributions on the grid for both models
    pred_a = predictive_distribution(model_a, likelihood_a, x_grid)
    pred_b = predictive_distribution(model_b, likelihood_b, x_grid)
    
    # Obtains mean and std deviation for both models
    mean_a = pred_a.mean.cpu()
    std_a = pred_a.variance.sqrt().cpu()
    mean_b = pred_b.mean.cpu()
    std_b = pred_b.variance.sqrt().cpu()
    
    x_np = x_grid.squeeze(-1).cpu().numpy()
    ax.plot(x_np, f_true.cpu().numpy(), color="0.65", linewidth=0.5,
        label="True latent f")
    ax.plot(x_train.squeeze(-1).cpu().numpy(), y_train.cpu().numpy(), "k*",
        markersize=8, label="Training data")
    
    Z = inducing_points.detach().cpu()
    ax.plot(Z.squeeze(-1).numpy(), np.zeros(Z.shape[0]), "rx", markersize=6,
        mew=2, label="Inducing points")
    
    ax.plot(x_np, mean_a.numpy(), label=label_a)
    ax.fill_between(x_np, (mean_a - PLOT_STD_MULT * std_a).numpy(),
        (mean_a + PLOT_STD_MULT * std_a).numpy(), alpha=0.20)
    
    ax.plot(x_np, mean_b.numpy(), "--", label=label_b)
    ax.fill_between(x_np, (mean_b - PLOT_STD_MULT * std_b).numpy(),
        (mean_b + PLOT_STD_MULT * std_b).numpy(), alpha=0.20)
    
    if y_star is not None:
        ax.axhline(float(y_star), color="lightgreen", linestyle="--", label="y*")
    if x_star is not None:
        ax.axvline(float(x_star), color="lightgreen", linestyle=":", label="x*") 
    ax.set_title(title)
    ax.legend(fontsize=6)


@torch.no_grad()
def plot_mean_and_band(ax, model, likelihood, x_grid, f_true, x_train, y_train,
    inducing_points, title, y_star=None, x_star=None):
    """This function plots the predictive mean and confidence band of a model"""
    # Computes predictive distribution on the grid
    pred = predictive_distribution(model, likelihood, x_grid)
    # Obtains mean and std deviation
    mean = pred.mean.cpu()
    std = pred.variance.sqrt().cpu()
    
    x_np = x_grid.squeeze(-1).cpu().numpy()
    ax.plot(x_np, f_true.cpu().numpy(), color="0.65", linewidth=0.5,
        label="True latent f")
    ax.plot(x_train.squeeze(-1).cpu().numpy(), y_train.cpu().numpy(), "k*",
        markersize=8, label="Training data")
    
    Z = inducing_points.detach().cpu()
    ax.plot(Z.squeeze(-1).numpy(), np.zeros(Z.shape[0]), "rx", markersize=6,
        mew=2, label="Inducing points")
    
    ax.plot(x_np, mean.numpy(), label="Mean")
    ax.fill_between(x_np, (mean - PLOT_STD_MULT * std).numpy(),
        (mean + PLOT_STD_MULT * std).numpy(), alpha=0.30,
        label=f"Confidence (±{PLOT_STD_MULT:.0f}std)")
    
    if y_star is not None:
        ax.axhline(float(y_star), color="lightgreen", linestyle="--", label="y*")
    if x_star is not None:
        ax.axvline(float(x_star), color="lightgreen", linestyle=":", label="x*")
        
    ax.set_title(title)
    ax.legend(fontsize=6, loc="lower left")


@torch.no_grad()
def get_common_plot_limits(x_grid, f_true, y_train, res_std, res_con,
    init_model, y_star=None):
    """This function computes common x/y limits for the three subplots"""
    x_np = x_grid.squeeze(-1).cpu().numpy()
    
    pred_init = predictive_distribution(init_model, res_std.likelihood, x_grid)
    pred_std = predictive_distribution(res_std.model, res_std.likelihood, x_grid)
    pred_con = predictive_distribution(res_con.model, res_con.likelihood, x_grid)
    
    curves = [
        f_true.cpu().numpy(),
        y_train.cpu().numpy(),
        pred_std.mean.cpu().numpy(),
        pred_con.mean.cpu().numpy(),
        pred_init.mean.cpu().numpy(),
        (pred_std.mean - PLOT_STD_MULT * pred_std.variance.sqrt()).cpu().numpy(),
        (pred_std.mean + PLOT_STD_MULT * pred_std.variance.sqrt()).cpu().numpy(),
        (pred_con.mean - PLOT_STD_MULT * pred_con.variance.sqrt()).cpu().numpy(),
        (pred_con.mean + PLOT_STD_MULT * pred_con.variance.sqrt()).cpu().numpy(),
        (pred_init.mean - PLOT_STD_MULT * pred_init.variance.sqrt()).cpu().numpy(),
        (pred_init.mean + PLOT_STD_MULT * pred_init.variance.sqrt()).cpu().numpy(),
    ]
    
    if y_star is not None:
        curves.append(np.array([float(y_star)]))
    
    y_min = min(float(np.min(c)) for c in curves)
    y_max = max(float(np.max(c)) for c in curves)
    y_pad = 0.08 * max(1e-6, y_max - y_min)
    
    x_min = float(x_np.min())
    x_max = float(x_np.max())
    
    return (x_min, x_max), (y_min - y_pad, y_max + y_pad)


def main():
    # Generates synthetic 1D problem with 4 observations
    x_grid, f_true, x_train, y_train = generate_4obs_problem()
    
    # Number of training points
    N = x_train.shape[0]
    # Number of inducing points (same as training points here)
    M = N
    
    # Uses training points as inducing points, fixing them
    fixed_inducing = x_train.contiguous()
    # Number of points for evaluating the constraint term
    num_constraint_points = 100
    # Noise level for the base GP model
    init_noise = 1e-4
    # Epsilon for step constrain term
    epsilon = 1e-1
    
    x_min, x_max = x_grid.min(), x_grid.max()
    # Samples constraint points uniformly from the grid range, used to
    # evaluate the step constraint term
    Xc_eval = x_min + (x_max - x_min) * torch.rand(
        num_constraint_points, 1, dtype=x_grid.dtype, device=x_grid.device
    )

    # Fits the base GP used for y* sampling and sparse initialization
    base_gp = fit_singletask_gp(x_train, y_train, init_noise=init_noise)
    
    # Defines bounds for optimization as the min and max of the grid
    bounds = torch.stack(
        [x_grid.min(dim=0).values, x_grid.max(dim=0).values], dim=0)
    
    # Samples candidate optimal pairs (x*,y*) from the posterior of the base GP
    sampled_x_stars, sampled_y_stars = sample_solution_outputs_from_model(
        base_gp=base_gp,bounds=bounds)
    
    # Selects one pair (x*,y*) from the sampled candidates for the constraint term
    y_star_info = choose_y_star(sampled_x_stars, sampled_y_stars)
    y_star = y_star_info["y_star"]
    x_star = y_star_info["x_star"]
    
    # Builds a sparse GP model just after initialization from the base GP posterior
    init_model = build_sparse_model_just_initialized(base_gp, fixed_inducing)
    
    # Fits standard sparse GP initialized from the SingleTaskGP posterior.
    res_std = fit_vfe_sparse_gp(train_X=x_train, train_Y=y_train,
        noise=float(base_gp.likelihood.noise.detach().cpu().item()),
        train_noise=False,
        M=M,
        y_star=None,
        fixed_inducing_points=fixed_inducing,
        seed_for_init=2024,
        base_gp=base_gp,
        verbose=False,
    )

    noise_star = float(res_std.likelihood.noise.detach().cpu().item())

    # Fit constrained sparse GP initialized from the same SingleTaskGP.
    res_con = fit_vfe_sparse_gp(
        train_X=x_train,
        train_Y=y_train,
        noise=noise_star,
        train_noise=False,
        M=M,
        y_star=y_star,
        epsilon=epsilon,
        lower_bound=x_grid.min(dim=0).values,
        upper_bound=x_grid.max(dim=0).values,
        fixed_inducing_points=fixed_inducing,
        seed_for_init=2024,
        base_gp=base_gp,
        verbose=False,
    )

    print("\nPrinting some results...")
    print("Training x:", x_train.squeeze(-1).cpu().numpy())
    print("Training y:", y_train.cpu().numpy())
    print(f"Selected sampled x*: {x_star:.6f}")
    print(f"Selected sampled y*: {y_star:.6f}")

    print("Mean difference between constrained and standard model on the grid:")
    obtain_mean_difference(res_std, res_con, x_grid)

    y_star_t = torch.tensor(y_star, dtype=x_grid.dtype, device=x_grid.device)
    p_std = prob_f_below_y_star(res_std.model, Xc_eval, y_star_t)
    p_con = prob_f_below_y_star(res_con.model, Xc_eval, y_star_t)

    print("\nP(f(Xc)<y*) under q(f):")
    print(f"Standard   : mean={p_std[0]:.3f}, min={p_std[1]:.3f}, max={p_std[2]:.3f}")
    print(f"Constraint : mean={p_con[0]:.3f}, min={p_con[1]:.3f}, max={p_con[2]:.3f}")

    print("\nNoise:")
    print("  Base GP learned:", float(base_gp.likelihood.noise.detach().cpu().item()))
    print("  Standard sparse:", noise_star)
    print("  Constraint fixed:", res_con.likelihood.noise.item())

    fig, axes = plt.subplots(1, 3, figsize=(19, 5.2), sharex=True, sharey=True)

    plot_two_predictive_distributions(
        axes[0],
        base_gp,
        base_gp.likelihood,
        init_model,
        base_gp.likelihood,
        x_grid,
        f_true,
        x_train,
        y_train,
        fixed_inducing,
        title="SingleTaskGP vs sparse GP just after initialization",
        label_a="SingleTaskGP",
        label_b="Sparse GP after init",
        y_star=y_star,
        x_star=x_star,
    )

    plot_mean_and_band(
        axes[1],
        res_std.model,
        res_std.likelihood,
        x_grid,
        f_true,
        x_train,
        y_train,
        res_std.inducing_points,
        title="Standard ELBO",
        y_star=y_star,
        x_star=x_star,
    )

    plot_mean_and_band(
        axes[2],
        res_con.model,
        res_con.likelihood,
        x_grid,
        f_true,
        x_train,
        y_train,
        res_con.inducing_points,
        title="Standard ELBO + step constraint term",
        y_star=y_star,
        x_star=x_star,
    )

    x_limits, y_limits = get_common_plot_limits(
        x_grid=x_grid,
        f_true=f_true,
        y_train=y_train,
        res_std=res_std,
        res_con=res_con,
        init_model=init_model,
        y_star=y_star,
    )
    for ax in axes:
        ax.set_xlim(*x_limits)
        ax.set_ylim(*y_limits)

    fig.suptitle("1D test with 3 observations (BoTorch base GP)", fontsize=16)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()


if __name__ == "__main__":
    main()
