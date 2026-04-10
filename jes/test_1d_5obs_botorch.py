#!/usr/bin/env python3
# coding: utf-8
from __future__ import annotations

import numpy as np
import torch
import matplotlib.pyplot as plt
import math

from modified_vfe_sparse_gp import (
    fit_vfe_sparse_gp,
    predictive_distribution as sparse_predictive_distribution,
    normal_cdf,
    VFESparseGP,
    build_init_dist_from_base_gp
)

from botorch.models.gp_regression import SingleTaskGP
from botorch.fit import fit_gpytorch_mll
from botorch.acquisition.utils import get_optimal_samples
from gpytorch.mlls.exact_marginal_log_likelihood import ExactMarginalLogLikelihood

# Number of training points (observations) we keep from the sampled function
NUM_TRAIN = 5
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
def generate_5obs_problem(num_grid: int = 1000, jitter: float=1e-7,
    seed_latent: int = 123, seed_train: int = 5):
    """ This function generates the synthetic 1D problem used in the test.
    It samples a latent function from a GP with a RBF kernel on [-5,5]
    and selects 5 grid points uniformly at random as training points"""
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
def predictive_distribution(model, likelihood, grid: torch.Tensor,
    observation_noise: bool=False):
    """This function obtains the predictive distribution of a model
    on the given grid"""
    if hasattr(model, "posterior"):
        return model.posterior(grid, observation_noise=observation_noise)
    return sparse_predictive_distribution(model, likelihood, grid,
        observation_noise=observation_noise)
    
    
@torch.no_grad()
def build_sparse_model_just_initialized(base_gp, inducing_points) -> VFESparseGP:
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


@torch.no_grad()
def marginal_mean_variance(model, likelihood, grid: torch.Tensor,
    observation_noise: bool = False):
    """This function returns the pointwise predictive mean and variance
    of a model on the given grid"""
    # Computes the predictive distribution on the grid
    post = predictive_distribution(model, likelihood, grid, observation_noise=observation_noise)
    
    # Extracts the mean and the variance, and returns them
    mean = post.mean
    variance = post.variance.clamp_min(1e-12)
    return mean, variance


@torch.no_grad()
def extract_noise_variance(model, likelihood, grid: torch.Tensor) -> torch.Tensor:
    """This functions obtains the noise variance of a model on the given grid"""
    # Computes the predictive variance with and without observation noise
    _, var_f = marginal_mean_variance(model, likelihood, grid, observation_noise=False)
    _, var_y = marginal_mean_variance(model, likelihood, grid, observation_noise=True)
    # Returns the difference between those two variances
    return (var_y - var_f).clamp_min(1e-12)


def normal_pdf(z: torch.Tensor) -> torch.Tensor:
    """This function computes the standard normal pdf"""
    return torch.exp(-0.5 * z.pow(2)) / math.sqrt(2.0 * math.pi)


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
def jes_truncated_predictive_moments(model, likelihood, grid: torch.Tensor,
    y_star: float | torch.Tensor, observation_noise: bool = False):
    """ This function computes the JEs style predictive mean and variance after
    upper truncation at y*"""
    # Obtains the marginal predictive mean and variance of the model on the grid
    mean_f, var_f = marginal_mean_variance(model, likelihood, grid,
        observation_noise=False)
    # Converts y_star to a tensor with correct dtype/device
    y_star_t = torch.as_tensor(y_star, dtype=mean_f.dtype, device=mean_f.device)
    # Computes the truncated mean and variance using the JES formulas for upper truncation
    mean_trunc, var_trunc = truncated_upper_normal_moments(mean_f, var_f, y_star_t)
    
    # If there is not observation noise, returns the truncated mean and variance directly
    if not observation_noise:
        return mean_trunc, var_trunc
    
    # If not, adds the noise variance to the truncated variance before returning
    noise_var = extract_noise_variance(model, likelihood, grid)
    return mean_trunc, var_trunc + noise_var


# Useful helpers for comparing probabilities
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


# Plotter helpers
@torch.no_grad()
def plot_mean_and_band(ax, x_grid: torch.Tensor, mean: torch.Tensor, std: torch.Tensor,
    f_true: torch.Tensor, x_obs: torch.Tensor, y_obs: torch.Tensor,
    inducing_points: torch.Tensor, title: str, y_star: float | None = None,
    x_star: float | None = None, x_pseudo: torch.Tensor | None = None,
    y_pseudo: torch.Tensor | None = None, band_label: str | None = None,
    mean_label: str = "Mean"):
    """This function plots predictive moments already computed on the grid."""
    x_np = x_grid.squeeze(-1).cpu().numpy()
    mean_np = mean.reshape(-1).cpu().numpy()
    std_np = std.reshape(-1).cpu().numpy()
    
    ax.plot(x_np,f_true.reshape(-1).cpu().numpy(),color="0.65",linewidth=0.7,
        label="True latent f")
    
    ax.plot(x_obs.squeeze(-1).cpu().numpy(), y_obs.reshape(-1).cpu().numpy(),
        "k*", markersize=8, label="Observed data")
    
    if x_pseudo is not None and y_pseudo is not None:
        ax.plot(x_pseudo.squeeze(-1).cpu().numpy(), y_pseudo.reshape(-1).cpu().numpy(),
            marker="o", linestyle="None", color="tab:green", markersize=7,
            label="Sampled optimum (x*, y*)")
        
    Z = inducing_points.detach().cpu()
    ax.plot(Z.squeeze(-1).numpy(), np.zeros(Z.shape[0]), "rx", markersize=6, mew=2,
        label="Inducing points")
    
    ax.plot(x_np, mean_np, label=mean_label)
    
    if band_label is None:
        band_label = f"Confidence band (±{PLOT_STD_MULT:.0f} std)"
        
    ax.fill_between(x_np, mean_np - PLOT_STD_MULT * std_np, mean_np + PLOT_STD_MULT * std_np,
        alpha=0.25, label=band_label)
    
    if y_star is not None:
        ax.axhline(float(y_star), color="lightgreen", linestyle="--", label="y*")
    if x_star is not None:
        ax.axvline(float(x_star), color="lightgreen", linestyle=":", label="x*")
        
    ax.set_title(title)
    ax.legend(fontsize=6, loc="lower left")


@torch.no_grad()
def get_common_plot_limits(curves: list[np.ndarray], y_star: float | None = None):
    """This function computes common y-limits for a set of curves."""
    if y_star is not None:
        curves = curves + [np.array([float(y_star)])]
        
    y_min = min(float(np.min(c)) for c in curves)
    y_max = max(float(np.max(c)) for c in curves)
    y_pad = 0.08 * max(1e-6, y_max - y_min)
    
    return y_min - y_pad, y_max + y_pad


def main():
    # Generates the synthetic 1D problem with 5 observations
    x_grid, f_true, x_train, y_train = generate_5obs_problem()
    
    # Number of points for evaluating the constraint term
    num_constraint_points = 100
    # Noise level for the base GP model
    init_noise = 1e-6
    # Epsilon for step constrain term
    epsilon = 1e-4
    
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
    
    # Conditions the base GP on the selected optimum pair, obtaining a new GP posterior
    # that incorporates this information
    conditioned_base_gp, x_star_t, y_star_t_col = condition_base_gp_on_optimum(base_gp,
        x_star=x_star, y_star=y_star)
    y_star_t = y_star_t_col.reshape(())

    # The sparse models are now trained on the augmented data set that includes the
    # sampled optimum pair as an additional observation
    train_X_aug = torch.cat([x_train, x_star_t.to(dtype=x_train.dtype, device=x_train.device)], dim=0)
    train_Y_aug = torch.cat([y_train, y_star_t_col.reshape(-1).to(dtype=y_train.dtype, device=y_train.device)], dim=0)
    # Uses training points + sampled optimum as inducing points, fixing them here
    fixed_inducing = train_X_aug.contiguous()
    # Number of inducing points
    M = fixed_inducing.shape[0]
    
    # Builds a sparse GP model just after initializing the base GP posterior
    init_model = build_sparse_model_just_initialized(conditioned_base_gp, fixed_inducing)
    base_gp_noise = float(conditioned_base_gp.likelihood.noise.detach().cpu().item())
    
    # Sparse models after conditioning on (x*,y*)
    # Fits std sparse GP initialized from the SingleTaskGP posterior
    res_std = fit_vfe_sparse_gp(train_X=train_X_aug, train_Y=train_Y_aug,
        noise=base_gp_noise, train_noise=False, M=M, y_star=None,
        fixed_inducing_points=fixed_inducing, seed_for_init=2024,
        base_gp=conditioned_base_gp)
    
    # Fits constrained sparse GP initialized from the same SingleTaskGP
    res_con = fit_vfe_sparse_gp(train_X=train_X_aug, train_Y=train_Y_aug, noise=base_gp_noise,
        train_noise=False, M=M, y_star=y_star, epsilon=epsilon,
        lower_bound=x_grid.min(dim=0).values, upper_bound=x_grid.max(dim=0).values,
        fixed_inducing_points=fixed_inducing, seed_for_init=2024,base_gp=conditioned_base_gp)
    
    # JES style truncation of the std sparse GP predictive
    # Predictive moments of the exact GP conditioned on (x*,y*)
    mean_base_cond, var_base_cond = marginal_mean_variance(conditioned_base_gp,
        conditioned_base_gp.likelihood, x_grid)
    
    # Predictive moments of the sparse GP just after initialization
    mean_init, var_init = marginal_mean_variance(init_model, conditioned_base_gp.likelihood, x_grid)
    # Predictive moments of the standard sparse GP after training with std ELBO
    mean_std, var_std = marginal_mean_variance(res_std.model, res_std.likelihood, x_grid)
    # Predictive moments of the constrained sparse GP after training with constraint ELBO
    mean_con, var_con = marginal_mean_variance(res_con.model, res_con.likelihood, x_grid)
    # JES truncation of the std sparse GP predictive moments after training with std ELBO
    mean_jes, var_jes = jes_truncated_predictive_moments(res_std.model, res_std.likelihood,
        x_grid, y_star=y_star, observation_noise=False)
    # JES truncation of the std sparse GP predictive moments after training with std ELBO,
    # including observation noise
    mean_jes_y, var_jes_y = jes_truncated_predictive_moments(res_std.model, res_std.likelihood,
        x_grid, y_star=y_star, observation_noise=True)
    
    print("\nPrinting some results...")
    print("Original observed x:", x_train.squeeze(-1).cpu().numpy())
    print("Original observed y:", y_train.cpu().numpy())
    print(f"Selected sampled x*: {x_star:.6f}")
    print(f"Selected sampled y*: {y_star:.6f}")
    print("Augmented training x (including x*):", train_X_aug.squeeze(-1).cpu().numpy())
    print("Augmented training y (including y*):", train_Y_aug.cpu().numpy())
    
    # Calculates P(f(Xc)<y*) under the predictive distribution of each model
    p_std = summarize_prob_less_than(res_std.model, res_std.likelihood, Xc_eval, y_star_t)
    p_con = summarize_prob_less_than(res_con.model, res_con.likelihood, Xc_eval, y_star_t)
    
    # JES truncated predictive moments on the evaluation points
    mean_jes_Xc, var_jes_Xc = jes_truncated_predictive_moments(res_std.model, res_std.likelihood,
        Xc_eval, y_star=y_star, observation_noise=False)
    # Calculates P(f(Xc)<y*) under the JES truncated predictive distribution
    p_jes_gaussian_moment_match = summarize_prob_less_than_from_moments(mean_jes_Xc, var_jes_Xc,
        y_star_t)
    
    print("\nP(f(Xc)<y*) summaries:")
    print(
        f"  Standard sparse GP: "
        f"mean={p_std[0]:.3f}, min={p_std[1]:.3f}, max={p_std[2]:.3f}"
    )
    print(
        "  JES truncation of the standard sparse GP predictive: "
        f"mean={p_jes_gaussian_moment_match[0]:.3f}, "
        f"min={p_jes_gaussian_moment_match[1]:.3f}, "
        f"max={p_jes_gaussian_moment_match[2]:.3f}"
    )
    print(
        f"  Modified sparse GP: "
        f"mean={p_con[0]:.3f}, min={p_con[1]:.3f}, max={p_con[2]:.3f}"
    )
    
    format_diff_stats("Standard sparse", mean_std, var_std, "JES truncation", mean_jes, var_jes)
    format_diff_stats("Standard sparse", mean_std, var_std, "Modified sparse", mean_con, var_con)
    format_diff_stats("JES truncation", mean_jes, var_jes, "Modified sparse", mean_con, var_con)
    
    print("\nNoise levels:")
    print("  Base GP fixed noise:", base_gp_noise)
    print("  Standard sparse GP fixed noise:", res_std.likelihood.noise.item())
    print("  Modified sparse GP fixed noise:", res_con.likelihood.noise.item())

    # Figures
    # Obtains standard deviations from variances for plotting
    std_init = var_init.sqrt()
    std_std = var_std.sqrt()
    std_jes = var_jes.sqrt()
    std_con = var_con.sqrt()
    
    x_np = x_grid.squeeze(-1).cpu().numpy()
    # Collects all curves to be plotted for determining common y-limits across panels
    common_curves = [
        f_true.reshape(-1).cpu().numpy(),
        train_Y_aug.reshape(-1).cpu().numpy(),
        mean_base_cond.reshape(-1).cpu().numpy(),
        mean_init.reshape(-1).cpu().numpy(),
        mean_std.reshape(-1).cpu().numpy(),
        mean_jes.reshape(-1).cpu().numpy(),
        mean_con.reshape(-1).cpu().numpy(),
        (mean_init - PLOT_STD_MULT * std_init).reshape(-1).cpu().numpy(),
        (mean_init + PLOT_STD_MULT * std_init).reshape(-1).cpu().numpy(),
        (mean_std - PLOT_STD_MULT * std_std).reshape(-1).cpu().numpy(),
        (mean_std + PLOT_STD_MULT * std_std).reshape(-1).cpu().numpy(),
        (mean_jes - PLOT_STD_MULT * std_jes).reshape(-1).cpu().numpy(),
        (mean_jes + PLOT_STD_MULT * std_jes).reshape(-1).cpu().numpy(),
        (mean_con - PLOT_STD_MULT * std_con).reshape(-1).cpu().numpy(),
        (mean_con + PLOT_STD_MULT * std_con).reshape(-1).cpu().numpy(),
    ]
    y_lim_low, y_lim_high = get_common_plot_limits(common_curves, y_star=y_star)
    
    # 1x4 subplots for the different models and comparisons
    fig, axes = plt.subplots(1, 4, figsize=(24, 5.4), sharex=True, sharey=True)
    
    # First plot: conditioned exact GP vs sparse GP just after initialization
    plot_mean_and_band(axes[0], x_grid=x_grid, mean=mean_init, std=std_init,
        f_true=f_true, x_obs=x_train, y_obs=y_train, inducing_points=fixed_inducing,
        title="Conditioned SingleTask GP vs Sparse GP after init",
        x_star=x_star, y_star=y_star, x_pseudo=x_star_t, y_pseudo=y_star_t_col.reshape(-1),
        mean_label="Sparse GP (after init) mean", band_label="Sparse GP (after init) band")
    
    axes[0].plot(x_np, mean_base_cond.reshape(-1).cpu().numpy(), linestyle="--",
        label="Conditioned SingleTask GP mean")
    axes[0].fill_between(
        x_np,
        (mean_base_cond - PLOT_STD_MULT * var_base_cond.sqrt()).reshape(-1).cpu().numpy(),
        (mean_base_cond + PLOT_STD_MULT * var_base_cond.sqrt()).reshape(-1).cpu().numpy(),
        alpha=0.12,
        label="Conditioned SingleTask GP band",
    )
    axes[0].legend(fontsize=6, loc="lower left")

    # Second plot: standard sparse GP
    plot_mean_and_band(axes[1], x_grid=x_grid, mean=mean_std, std=std_std, f_true=f_true,
        x_obs=x_train, y_obs=y_train, inducing_points=res_std.inducing_points,
        title="Standard sparse GP (after training with std ELBO)",
        x_star=x_star, y_star=y_star, x_pseudo=x_star_t,y_pseudo=y_star_t_col.reshape(-1))
    
    # Third plot: JES-style truncation of the standard sparse GP predictive
    plot_mean_and_band(axes[2], x_grid=x_grid, mean=mean_jes, std=std_jes, f_true=f_true,
        x_obs=x_train, y_obs=y_train, inducing_points=res_std.inducing_points,
        title="JES truncation of the std sparse GP predictive",
        x_star=x_star, y_star=y_star, x_pseudo=x_star_t, y_pseudo=y_star_t_col.reshape(-1))
    
    # Figure 4: modified sparse GP with the step constraint term
    plot_mean_and_band(axes[3], x_grid=x_grid, mean=mean_con, std=std_con, f_true=f_true,
        x_obs=x_train, y_obs=y_train, inducing_points=res_con.inducing_points,
        title="Modified sparse GP (after training with constraint ELBO)",
        x_star=x_star, y_star=y_star, x_pseudo=x_star_t, y_pseudo=y_star_t_col.reshape(-1))
    
    for ax in axes:
        ax.set_xlim(float(x_np.min()), float(x_np.max()))
        ax.set_ylim(y_lim_low, y_lim_high)
        
    fig.suptitle(
        "1D test with 5 observations",
        fontsize=15,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()


if __name__ == "__main__":
    main()
