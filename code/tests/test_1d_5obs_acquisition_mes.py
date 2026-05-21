#!/usr/bin/env python3
# coding: utf-8
"""Test MES acquisition object on a 1D problem."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import torch

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from my_acquisition_mes import MyAcquisition
from my_utils import (
    choose_y_star,
    fit_singletask_gp,
    format_diff_stats,
    marginal_mean_variance,
    sample_solution_outputs_from_model,
    summarize_acquisition_curve,
    upper_truncated_predictive_moments,
)


NUM_TRAIN = 5
PLOT_STD_MULT = 1.0
INIT_NOISE = 1e-6
M_EXTRA = 10
EXACT_MAX_TOL = 0.01
EXACT_NUM_FUNCTION_SAMPLES = 5000


def kernel(x: torch.Tensor, y: torch.Tensor, lengthscale: float = 2.0,
    variance: float = 1.0) -> torch.Tensor:
    x_scaled = x / lengthscale
    y_scaled = y / lengthscale
    sqdist = (x_scaled[:, None, :] - y_scaled[None, :, :]).pow(2).sum(dim=-1)
    return variance * torch.exp(-0.5 * sqdist)


@torch.no_grad()
def generate_5obs_problem(num_grid: int = 1000, jitter: float = 1e-7,
    seed_latent: int = 42, seed_train: int = 42):
    dtype = torch.float64
    x_grid = torch.linspace(-5.0, 5.0, num_grid, dtype=dtype).unsqueeze(-1)
    sigma = kernel(x_grid, x_grid) + jitter * torch.eye(num_grid, dtype=dtype)
    L = torch.linalg.cholesky(sigma)

    g_latent = torch.Generator(device="cpu")
    g_latent.manual_seed(seed_latent)
    f_true = L @ torch.randn(num_grid, dtype=dtype, generator=g_latent)

    rng = np.random.default_rng(seed_train)
    p_sel = np.sort(rng.choice(np.arange(num_grid), size=NUM_TRAIN, replace=False))
    p_sel = torch.tensor(p_sel, dtype=torch.long)

    x_train = x_grid[p_sel].contiguous()
    y_train = f_true[p_sel].contiguous()
    return x_grid, f_true, x_train, y_train


@dataclass
class ExactConditionalApproximation:
    mean_f: torch.Tensor
    var_f: torch.Tensor
    mean_y: torch.Tensor
    var_y: torch.Tensor
    accepted_maxima: torch.Tensor
    total_draws: int


@torch.no_grad()
def approximate_exact_conditional_from_function_samples(base_gp, grid: torch.Tensor,
    y_star: float | torch.Tensor, noise_variance: float | torch.Tensor,
    num_function_samples: int = EXACT_NUM_FUNCTION_SAMPLES
    ) -> Optional[ExactConditionalApproximation]:
    base_gp.eval()
    base_gp.likelihood.eval()

    posterior = base_gp.posterior(grid, observation_noise=False)
    y_star_t = torch.as_tensor(y_star, dtype=grid.dtype, device=grid.device)
    noise_var_t = torch.as_tensor(noise_variance, dtype=grid.dtype, device=grid.device)

    samples = posterior.rsample(sample_shape=torch.Size([num_function_samples]))
    if samples.ndim > 2 and samples.shape[-1] == 1:
        samples = samples.squeeze(-1)

    maxima = samples.max(dim=-1).values
    keep = (maxima - y_star_t).abs() <= float(np.abs(y_star_t) * EXACT_MAX_TOL)
    if not torch.any(keep):
        return None

    accepted_functions = samples[keep]
    accepted_maxima = maxima[keep]
    mean_f = accepted_functions.mean(dim=0)
    var_f = accepted_functions.var(dim=0, unbiased=False).clamp_min(1e-12)
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
def entropy_reduction(var_initial: torch.Tensor, var_conditional: torch.Tensor) -> torch.Tensor:
    return 0.5 * (
        torch.log(var_initial.reshape(-1).clamp_min(1e-12))
        - torch.log(var_conditional.reshape(-1).clamp_min(1e-12))
    )


@torch.no_grad()
def normalize_acquisition(acq: torch.Tensor) -> torch.Tensor:
    return acq / acq.max().clamp_min(1e-12)


@torch.no_grad()
def get_common_plot_limits(curves: list[np.ndarray], y_star: float | None = None):
    if y_star is not None:
        curves = curves + [np.array([float(y_star)])]
    y_min = min(float(np.min(c)) for c in curves)
    y_max = max(float(np.max(c)) for c in curves)
    y_pad = 0.08 * max(1e-6, y_max - y_min)
    return y_min - y_pad, y_max + y_pad


@torch.no_grad()
def plot_predictive(ax, x_grid: torch.Tensor, x_train: torch.Tensor, y_train: torch.Tensor,
    mean: torch.Tensor, var: torch.Tensor, title: str, y_star: float,
    exact_cond: Optional[ExactConditionalApproximation] = None):
    x_np = x_grid.squeeze(-1).cpu().numpy()
    mean_np = mean.reshape(-1).cpu().numpy()
    std_np = var.reshape(-1).sqrt().cpu().numpy()

    ax.plot(x_train.squeeze(-1).cpu().numpy(), y_train.reshape(-1).cpu().numpy(),
        "k*", markersize=8, label="Observed data")
    ax.plot(x_np, mean_np, linewidth=2.0, label="Mean")
    ax.fill_between(x_np, mean_np - PLOT_STD_MULT * std_np,
        mean_np + PLOT_STD_MULT * std_np, alpha=0.20, label="Band")
    ax.axhline(float(y_star), color="lightgreen", linestyle="--", label="y*")

    if exact_cond is not None:
        exact_mean = exact_cond.mean_y.reshape(-1).cpu().numpy()
        exact_std = exact_cond.var_y.reshape(-1).sqrt().cpu().numpy()
        ax.plot(x_np, exact_mean, linestyle="--", linewidth=2.4,
            label="Approx. exact mean")
        ax.fill_between(x_np, exact_mean - PLOT_STD_MULT * exact_std,
            exact_mean + PLOT_STD_MULT * exact_std, alpha=0.12,
            label="Approx. exact band")

    ax.set_title(title)
    ax.legend(fontsize=7, loc="best")


def main():
    x_grid, f_true, x_train, y_train = generate_5obs_problem()

    base_gp = fit_singletask_gp(x_train, y_train, init_noise=INIT_NOISE)
    base_gp_noise = float(base_gp.likelihood.noise.detach().cpu().item())

    bounds = torch.stack([x_grid.min(dim=0).values, x_grid.max(dim=0).values], dim=0)
    sampled_x_stars, sampled_y_stars = sample_solution_outputs_from_model(
        base_gp=base_gp, bounds=bounds)
    y_star_info = choose_y_star(sampled_x_stars, sampled_y_stars)
    x_star = y_star_info["x_star"]
    y_star = y_star_info["y_star"]

    print("\nMES acquisition object test")
    print("Original observed x:", x_train.squeeze(-1).cpu().numpy())
    print("Original observed y:", y_train.cpu().numpy())
    print(f"Selected sampled x*: {x_star:.6f}")
    print(f"Selected sampled y*: {y_star:.6f}")

    acq_object = MyAcquisition(
        model=base_gp,
        x_star=x_star,
        y_star=y_star,
        M=M_EXTRA,
        lower_bound=x_grid.min(dim=0).values,
        upper_bound=x_grid.max(dim=0).values,
    )

    mean_obj_y, var_obj_y = marginal_mean_variance(
        acq_object.conditional_model, acq_object.conditional_likelihood,
        x_grid, observation_noise=True)
    mean_trunc_y, var_trunc_y = upper_truncated_predictive_moments(
        base_gp, base_gp.likelihood, x_grid, y_star=y_star, observation_noise=True)

    torch.manual_seed(2026)
    exact_cond = approximate_exact_conditional_from_function_samples(
        base_gp=base_gp, grid=x_grid, y_star=y_star,
        noise_variance=base_gp_noise,
        num_function_samples=EXACT_NUM_FUNCTION_SAMPLES)

    if exact_cond is None:
        print("No function samples accepted")
    else:
        format_diff_stats(
            "Exact MES conditional approx", exact_cond.mean_y, exact_cond.var_y,
            "MES Gaussian truncation", mean_trunc_y, var_trunc_y)
        format_diff_stats(
            "Exact MES conditional approx", exact_cond.mean_y, exact_cond.var_y,
            "MES acquisition object model", mean_obj_y, var_obj_y)

    X_acq = x_grid.unsqueeze(-2)
    acq_from_object = acq_object(X_acq).detach()

    initial_var_y = base_gp.posterior(x_grid, observation_noise=True).variance.detach()
    acq_trunc = entropy_reduction(initial_var_y, var_trunc_y)
    acq_object_manual = entropy_reduction(initial_var_y, var_obj_y)

    print("\nAcquisition summaries:")
    if exact_cond is not None:
        acq_exact = entropy_reduction(initial_var_y, exact_cond.var_y)
        summarize_acquisition_curve("   Exact MES conditional", x_grid, acq_exact)
    else:
        acq_exact = None
    summarize_acquisition_curve("   MES Gaussian truncation", x_grid, acq_trunc)
    summarize_acquisition_curve("   MES acquisition object", x_grid, acq_from_object)

    max_forward_diff = (acq_from_object - acq_object_manual).abs().max().item()
    # print(f"Max forward/manual diff: {max_forward_diff:.6e}")

    common_curves = [
        f_true.reshape(-1).cpu().numpy(),
        y_train.reshape(-1).cpu().numpy(),
        mean_trunc_y.reshape(-1).cpu().numpy(),
        mean_obj_y.reshape(-1).cpu().numpy(),
        (mean_trunc_y - PLOT_STD_MULT * var_trunc_y.sqrt()).reshape(-1).cpu().numpy(),
        (mean_trunc_y + PLOT_STD_MULT * var_trunc_y.sqrt()).reshape(-1).cpu().numpy(),
        (mean_obj_y - PLOT_STD_MULT * var_obj_y.sqrt()).reshape(-1).cpu().numpy(),
        (mean_obj_y + PLOT_STD_MULT * var_obj_y.sqrt()).reshape(-1).cpu().numpy(),
    ]
    if exact_cond is not None:
        common_curves += [
            exact_cond.mean_y.reshape(-1).cpu().numpy(),
            (exact_cond.mean_y - PLOT_STD_MULT * exact_cond.var_y.sqrt()).reshape(-1).cpu().numpy(),
            (exact_cond.mean_y + PLOT_STD_MULT * exact_cond.var_y.sqrt()).reshape(-1).cpu().numpy(),
        ]
    y_lim_low, y_lim_high = get_common_plot_limits(common_curves, y_star=y_star)

    fig, axes = plt.subplots(2, 2, figsize=(22.0, 15.3))

    plot_predictive(axes[0, 0], x_grid, x_train, y_train, mean_obj_y, var_obj_y,
        "p(y|D,y*) from MES acquisition object", y_star, exact_cond)
    axes[0, 0].set_xlim(float(x_grid.min().item()), float(x_grid.max().item()))
    axes[0, 0].set_ylim(y_lim_low, y_lim_high)

    plot_predictive(axes[0, 1], x_grid, x_train, y_train, mean_trunc_y, var_trunc_y,
        "p(y|D,y*) from MES Gaussian truncation", y_star, exact_cond)
    axes[0, 1].set_xlim(float(x_grid.min().item()), float(x_grid.max().item()))
    axes[0, 1].set_ylim(y_lim_low, y_lim_high)

    x_np = x_grid.squeeze(-1).cpu().numpy()
    ax = axes[1, 0]
    ax.set_title("Acquisition curves")
    if acq_exact is not None:
        ax.plot(x_np, acq_exact.reshape(-1).cpu().numpy(), linewidth=2.4,
            label="Exact_Acq")
    ax.plot(x_np, acq_trunc.reshape(-1).cpu().numpy(), linewidth=2.4,
        label="MES_Trunc_Acq")
    ax.plot(x_np, acq_from_object.reshape(-1).cpu().numpy(), linewidth=2.4,
        label="MES_Object_Acq")
    ax.axvline(float(x_star), color="lightgreen", linestyle=":", label="x*")
    ax.legend(fontsize=7, loc="best")

    ax = axes[1, 1]
    ax.set_title("Normalized acquisition curves")
    if acq_exact is not None:
        ax.plot(x_np, normalize_acquisition(acq_exact).reshape(-1).cpu().numpy(),
            linewidth=2.4, label="Exact_Acq_Norm")
    ax.plot(x_np, normalize_acquisition(acq_trunc).reshape(-1).cpu().numpy(),
        linewidth=2.4, label="MES_Trunc_Acq_Norm")
    ax.plot(x_np, normalize_acquisition(acq_from_object).reshape(-1).cpu().numpy(),
        linewidth=2.4, label="MES_Object_Acq_Norm")
    ax.axvline(float(x_star), color="lightgreen", linestyle=":", label="x*")
    ax.legend(fontsize=7, loc="best")

    fig.suptitle("1D MES acquisition object test with 5 observations", fontsize=15, y=0.98)
    fig.tight_layout(rect=[0, 0, 1, 0.94], h_pad=3.0)
    plt.show()


if __name__ == "__main__":
    main()
