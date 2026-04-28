#!/usr/bin/env python3
# coding: utf-8
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch
import matplotlib.pyplot as plt

from modified_vfe_sparse_gp import fit_vfe_sparse_gp
from my_utils import (
    fit_singletask_gp,
    sample_solution_outputs_from_model,
    choose_y_star,
    condition_base_gp_on_optimum,
    marginal_mean_variance,
    upper_truncated_predictive_moments,
    gaussian_entropy_reduction_acq,
    summarize_acquisition_curve,
    format_diff_stats,
)

# Number of training points (observations) we keep from the sampled function
NUM_TRAIN = 5
# Multiplier for the standard deviation when plotting confidence bands
PLOT_STD_MULT = 1.0
# Noise level for the base GP model
INIT_NOISE = 1e-6

# Parameters for the rejection approximation to the exact conditional p(y|D,y*)
EXACT_MAX_TOL = 0.1
EXACT_NUM_FUNCTION_SAMPLES = 1000


def kernel(x: torch.Tensor, y: torch.Tensor, lengthscale: float = 2.0,
    variance: float = 1.0) -> torch.Tensor:
    """This function defines the RBF kernel used in this test"""
    # Scales inputs by lengthscale
    x_scaled = x / lengthscale
    y_scaled = y / lengthscale

    # Computes squared distance matrix
    sqdist = (x_scaled[:, None, :] - y_scaled[None, :, :]).pow(2).sum(dim=-1)
    # Returns the RBF kernel matrix
    return variance * torch.exp(-0.5 * sqdist)


@torch.no_grad()
def generate_5obs_problem(num_grid: int = 1000, jitter: float = 1e-7,
    seed_latent: int = 42, seed_train: int = 42):
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
    p_sel = np.sort(rng.choice(np.arange(num_grid), size=NUM_TRAIN, replace=False))
    p_sel = torch.tensor(p_sel, dtype=torch.long)

    # Extracts the training inputs and their outputs
    x_train = x_grid[p_sel].contiguous()
    y_train = f_true[p_sel].contiguous()

    return x_grid, f_true, x_train, y_train


@dataclass
class ExactConditionalApproximation:
    """This class stores the approximation to the exact conditional obtained
    by rejection sampling full function draws from the GP posterior"""
    mean_f: torch.Tensor
    var_f: torch.Tensor
    mean_y: torch.Tensor
    var_y: torch.Tensor
    accepted_maxima: torch.Tensor
    total_draws: int


@torch.no_grad()
def approximate_exact_conditional_from_function_samples(base_gp, grid: torch.Tensor,
    y_star: float | torch.Tensor, noise_variance: float | torch.Tensor,
    max_tol: float = EXACT_MAX_TOL,
    num_function_samples: int = EXACT_NUM_FUNCTION_SAMPLES
    ) -> Optional[ExactConditionalApproximation]:
    """ This function approximates p(y|D,y*) by rejection sampling full
    function draws from the GP posterior on a dense 1D grid
    """
    # Sets model and likelihood to eval mode
    base_gp.eval()
    base_gp.likelihood.eval()
    
    # Posterior of the latent function on the dense grid
    posterior = base_gp.posterior(grid, observation_noise=False)
    # Converts y* and noise variance to tensors with the correct dtype/device
    y_star_t = torch.as_tensor(y_star, dtype=grid.dtype, device=grid.device)
    noise_var_t = torch.as_tensor(noise_variance, dtype=grid.dtype, device=grid.device)
    
    # Samples complete latent functions on the grid from the multivariate Gaussian
    samples = posterior.rsample(sample_shape=torch.Size([num_function_samples]))
    if samples.ndim > 2 and samples.shape[-1] == 1:
        samples = samples.squeeze(-1)
        
    # Keeps only functions whose maximum lies in a neighbourhood of y*
    max = samples.max(dim=-1).values
    keep = (max - y_star_t).abs() <= float(max_tol)
    
    # If no function sample was accepted, returns None
    if not torch.any(keep):
        return None
    
    accepted_functions = samples[keep]
    accepted_maxima = max[keep]
    
    # Mean and variance of the accepted latent functions on the grid
    mean_f = accepted_functions.mean(dim=0)
    var_f = accepted_functions.var(dim=0, unbiased=False).clamp_min(1e-12)
    
    # To move from latent f to predictive y, we add the observation noise variance
    mean_y = mean_f.clone()
    var_y = var_f + noise_var_t
    
    return ExactConditionalApproximation(
        mean_f=mean_f,
        var_f=var_f,
        mean_y=mean_y,
        var_y=var_y,
        accepted_maxima=accepted_maxima,
        total_draws=int(num_function_samples),
    )


@torch.no_grad()
def plot_mean_and_band(ax, x_grid: torch.Tensor, mean: torch.Tensor, std: torch.Tensor,
    f_true: torch.Tensor, x_obs: torch.Tensor, y_obs: torch.Tensor,
    title: str, y_star: float | None = None, band_label: str | None = None,
    mean_label: str = "Mean"):
    """This function plots predictive moments already computed on the grid."""
    x_np = x_grid.squeeze(-1).cpu().numpy()
    mean_np = mean.reshape(-1).cpu().numpy()
    std_np = std.reshape(-1).cpu().numpy()

    ax.plot(x_np, f_true.reshape(-1).cpu().numpy(), color="0.65", linewidth=0.7,
        label="True latent f")

    ax.plot(x_obs.squeeze(-1).cpu().numpy(), y_obs.reshape(-1).cpu().numpy(),
        "k*", markersize=8, label="Observed data")

    ax.plot(x_np, mean_np, label=mean_label)

    if band_label is None:
        band_label = f"Confidence band (±{PLOT_STD_MULT:.0f} std)"

    ax.fill_between(x_np, mean_np - PLOT_STD_MULT * std_np,
        mean_np + PLOT_STD_MULT * std_np, alpha=0.20, label=band_label)

    if y_star is not None:
        ax.axhline(float(y_star), color="lightgreen", linestyle="--", label="y*")

    ax.set_title(title)
    ax.legend(fontsize=7, loc="best")


@torch.no_grad()
def plot_acquisition_comparison(ax, x_grid: torch.Tensor, x_star: float,
    acq_mes: torch.Tensor, acq_model_conditioned: torch.Tensor):
    """This function plots the two acquisition functions requested by the tutor."""
    x_np = x_grid.squeeze(-1).cpu().numpy()

    ax.plot(x_np, acq_mes.reshape(-1).cpu().numpy(), linewidth=2.2,
        label="MES (sin condicionar a (x*, y*))")
    ax.plot(x_np, acq_model_conditioned.reshape(-1).cpu().numpy(), linewidth=2.2,
        label="Modified sparse GP condicionado")

    ax.axvline(float(x_star), color="lightgreen", linestyle=":", linewidth=1.8,
        label="Sampled x*")
    ax.set_xlim(float(x_np.min()), float(x_np.max()))
    ax.set_title("Acquisition comparison: MES vs conditioned model")
    ax.set_xlabel("x")
    ax.set_ylabel("Approximate acquisition")
    ax.legend(fontsize=8, loc="best")


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
    
    # Fits the base GP used for y* sampling and as exact reference model
    base_gp = fit_singletask_gp(x_train, y_train, init_noise=INIT_NOISE)
    base_gp_noise = float(base_gp.likelihood.noise.detach().cpu().item())
    
    # Defines bounds for optimization as the min and max of the grid
    bounds = torch.stack([x_grid.min(dim=0).values, x_grid.max(dim=0).values], dim=0)
    # Samples candidate optimal pairs (x*, y*) from the posterior of the base GP
    sampled_x_stars, sampled_y_stars = sample_solution_outputs_from_model(base_gp=base_gp, bounds=bounds)
    
    # Selects one pair (x*, y*) from the sampled candidates
    y_star_info = choose_y_star(sampled_x_stars, sampled_y_stars)
    y_star = y_star_info["y_star"]
    x_star = y_star_info["x_star"]
    
    print("\nPrinting some results...")
    print("Original observed x:", x_train.squeeze(-1).cpu().numpy())
    print("Original observed y:", y_train.cpu().numpy())
    print(f"Selected sampled x*: {x_star:.6f}")
    print(f"Selected sampled y*: {y_star:.6f}")
    
    # Predictive comparison for p(y|D,y*)
    # Fits the modified sparse GP trained only with the y* constraint, without conditioning
    # on x*
    fixed_inducing_yonly = x_train.contiguous()
    M_yonly = fixed_inducing_yonly.shape[0]
    res_con_yonly = fit_vfe_sparse_gp(train_X=x_train, train_Y=y_train, noise=base_gp_noise,
        train_noise=False, M=M_yonly, y_star=y_star, lower_bound=x_grid.min(dim=0).values,
        upper_bound=x_grid.max(dim=0).values, fixed_inducing_points=fixed_inducing_yonly,
        base_gp=base_gp, old_model=base_gp, verbose=False)
        
    # Gaussian truncation applied directly to the exact base GP predictive
    mean_trunc_y, var_trunc_y = upper_truncated_predictive_moments(base_gp, base_gp.likelihood,
        x_grid, y_star=y_star,observation_noise=True)
    
    # Predictive moments of the modified sparse GP trained only with y*
    mean_con_y, var_con_y = marginal_mean_variance(res_con_yonly.model, res_con_yonly.likelihood,
        x_grid,observation_noise=True)
    
    # Approximation to the exact conditional from accepted function draws
    exact_cond = approximate_exact_conditional_from_function_samples(base_gp=base_gp, grid=x_grid,
        y_star=y_star, noise_variance=base_gp_noise, max_tol=EXACT_MAX_TOL, num_function_samples=EXACT_NUM_FUNCTION_SAMPLES)
    
    if exact_cond is None:
        print(" No function samples were accepted with the current criterion ")
    else:
        format_diff_stats(
            "Exact conditional approx", exact_cond.mean_y, exact_cond.var_y,
            "Gaussian truncation", mean_trunc_y, var_trunc_y)
        format_diff_stats(
            "Exact conditional approx", exact_cond.mean_y, exact_cond.var_y,
            "Modified sparse (y* only)", mean_con_y, var_con_y)
        
    # Conditions the base GP on the selected optimum pair, obtaining a new GP posterior
    # that incorporates this information
    conditioned_base_gp, x_star_t, y_star_t_col = condition_base_gp_on_optimum(
        base_gp, x_star=x_star, y_star=y_star)
    
    # The conditioned sparse model is trained on the augmented data set that includes
    # the sampled optimum pair as an additional observation
    train_X_aug = torch.cat([x_train,
        x_star_t.to(dtype=x_train.dtype, device=x_train.device)], dim=0)
    train_Y_aug = torch.cat([y_train,
        y_star_t_col.reshape(-1).to(dtype=y_train.dtype, device=y_train.device)], dim=0)
    
    fixed_inducing_aug = train_X_aug.contiguous()
    M_aug = fixed_inducing_aug.shape[0]
    
    # Fits the conditioned modified sparse GP
    res_con_xy = fit_vfe_sparse_gp(train_X=train_X_aug, train_Y=train_Y_aug, noise=base_gp_noise,
        train_noise=False, M=M_aug, y_star=y_star, lower_bound=x_grid.min(dim=0).values,
        upper_bound=x_grid.max(dim=0).values, fixed_inducing_points=fixed_inducing_aug,
        base_gp=conditioned_base_gp, old_model=conditioned_base_gp, verbose=False)
    
    # MES: no conditioning on (x*, y*)
    _, var_base = marginal_mean_variance(base_gp, base_gp.likelihood, x_grid,
        observation_noise=False)
    _, var_mes = upper_truncated_predictive_moments(base_gp, base_gp.likelihood,
        x_grid, y_star=y_star, observation_noise=False)
    
    # Our model: conditioned sparse approximation
    _, var_con_xy = marginal_mean_variance(res_con_xy.model, res_con_xy.likelihood,
        x_grid, observation_noise=False)
    
    acq_mes = gaussian_entropy_reduction_acq(var_base, var_mes)
    acq_model_conditioned = gaussian_entropy_reduction_acq(var_base, var_con_xy)
    print("\nApproximate acquisition summaries:")
    summarize_acquisition_curve("   MES (sin condicionar a (x*,y*))", x_grid, acq_mes)
    summarize_acquisition_curve("   Modified sparse GP condicionado", x_grid, acq_model_conditioned)
    
    # Obtains standard deviations from variances for plotting
    std_trunc_y = var_trunc_y.sqrt()
    std_con_y = var_con_y.sqrt()
    
    if exact_cond is not None:
        std_exact_y = exact_cond.var_y.sqrt()
        common_curves = [
            f_true.reshape(-1).cpu().numpy(),
            y_train.reshape(-1).cpu().numpy(),
            exact_cond.mean_y.reshape(-1).cpu().numpy(),
            mean_trunc_y.reshape(-1).cpu().numpy(),
            mean_con_y.reshape(-1).cpu().numpy(),
            (exact_cond.mean_y - PLOT_STD_MULT * std_exact_y).reshape(-1).cpu().numpy(),
            (exact_cond.mean_y + PLOT_STD_MULT * std_exact_y).reshape(-1).cpu().numpy(),
            (mean_trunc_y - PLOT_STD_MULT * std_trunc_y).reshape(-1).cpu().numpy(),
            (mean_trunc_y + PLOT_STD_MULT * std_trunc_y).reshape(-1).cpu().numpy(),
            (mean_con_y - PLOT_STD_MULT * std_con_y).reshape(-1).cpu().numpy(),
            (mean_con_y + PLOT_STD_MULT * std_con_y).reshape(-1).cpu().numpy(),
        ]
    else:
        common_curves = [
            f_true.reshape(-1).cpu().numpy(),
            y_train.reshape(-1).cpu().numpy(),
            mean_trunc_y.reshape(-1).cpu().numpy(),
            mean_con_y.reshape(-1).cpu().numpy(),
            (mean_trunc_y - PLOT_STD_MULT * std_trunc_y).reshape(-1).cpu().numpy(),
            (mean_trunc_y + PLOT_STD_MULT * std_trunc_y).reshape(-1).cpu().numpy(),
            (mean_con_y - PLOT_STD_MULT * std_con_y).reshape(-1).cpu().numpy(),
            (mean_con_y + PLOT_STD_MULT * std_con_y).reshape(-1).cpu().numpy(),
        ]
        
    y_lim_low, y_lim_high = get_common_plot_limits(common_curves, y_star=y_star)
    
    # Figure: predictive comparison and acquisition comparison
    fig, axes = plt.subplots(1, 2, figsize=(22.0, 5.3))
    
    ax_pred = axes[0]
    plot_mean_and_band(ax_pred, x_grid=x_grid, mean=mean_trunc_y, std=std_trunc_y,
        f_true=f_true, x_obs=x_train, y_obs=y_train,
        title="Predictive comparison for p(y|D,y*)", y_star=y_star,
        mean_label="Gaussian truncation mean", band_label="Gaussian truncation band")
    ax_pred.plot(x_grid.squeeze(-1).cpu().numpy(), mean_con_y.reshape(-1).cpu().numpy(),
        linewidth=2.0, label="Modified sparse GP mean")
    ax_pred.fill_between(
        x_grid.squeeze(-1).cpu().numpy(),
        (mean_con_y - PLOT_STD_MULT * std_con_y).reshape(-1).cpu().numpy(),
        (mean_con_y + PLOT_STD_MULT * std_con_y).reshape(-1).cpu().numpy(),
        alpha=0.20,
        label="Modified sparse GP band",
    )
    if exact_cond is not None:
        ax_pred.plot(x_grid.squeeze(-1).cpu().numpy(),
            exact_cond.mean_y.reshape(-1).cpu().numpy(), linestyle="--", linewidth=2.4,
            label="Approx. exact conditional mean")
        ax_pred.fill_between(
            x_grid.squeeze(-1).cpu().numpy(),
            (exact_cond.mean_y - PLOT_STD_MULT * exact_cond.var_y.sqrt()).reshape(-1).cpu().numpy(),
            (exact_cond.mean_y + PLOT_STD_MULT * exact_cond.var_y.sqrt()).reshape(-1).cpu().numpy(),
            alpha=0.12,
            label="Approx. exact conditional band",
        )
    ax_pred.set_xlim(float(x_grid.min().item()), float(x_grid.max().item()))
    ax_pred.set_ylim(y_lim_low, y_lim_high)
    ax_pred.legend(fontsize=7, loc="best")
    
    ax_acq = axes[1]
    plot_acquisition_comparison(ax_acq, x_grid=x_grid, x_star=x_star,
        acq_mes=acq_mes, acq_model_conditioned=acq_model_conditioned)
    
    fig.suptitle("1D test with 5 observations", fontsize=15)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    
    plt.show()


if __name__ == "__main__":
    main()
