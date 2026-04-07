#!/usr/bin/env python3
# coding: utf-8
"""
Improved 2D Branin comparison between:
1) standard sparse GP,
2) step-term constrained sparse GP.

This version focuses on presentation quality:
- larger panels,
- no overlapping title/subtitle,
- one dedicated colorbar per row,
- square panels on [0,1]^2,
- printed guide explaining each subplot.
"""

from __future__ import annotations

import time
import math
import torch
import matplotlib.pyplot as plt

from modified_vfe_sparse_gp import fit_vfe_sparse_gp, predictive_distribution, normal_cdf


def branin_unit_box(X: torch.Tensor) -> torch.Tensor:
    """Branin function with inputs in [0, 1]^2, output in the classical scale."""
    x1 = -5.0 + 15.0 * X[..., 0]
    x2 = 15.0 * X[..., 1]

    a = 1.0
    b = 5.1 / (4.0 * torch.pi**2)
    c = 5.0 / torch.pi
    r = 6.0
    s = 10.0
    t = 1.0 / (8.0 * torch.pi)

    y = a * (x2 - b * x1.square() + c * x1 - r).square() + s * (1.0 - t) * torch.cos(x1) + s
    return y


@torch.no_grad()
def branin_true_minima_unit_box(dtype: torch.dtype = torch.float64) -> torch.Tensor:
    mins_classical = torch.tensor(
        [
            [-math.pi, 12.275],
            [math.pi, 2.275],
            [3.0 * math.pi, 2.475],
        ],
        dtype=dtype,
    )
    mins_unit = mins_classical.clone()
    mins_unit[:, 0] = (mins_unit[:, 0] + 5.0) / 15.0
    mins_unit[:, 1] = mins_unit[:, 1] / 15.0
    return mins_unit


@torch.no_grad()
def make_branin_dataset(n_train: int = 80, noise_std: float = 0.05, seed: int = 7):
    torch.manual_seed(seed)
    sobol = torch.quasirandom.SobolEngine(dimension=2, scramble=True, seed=seed)
    X = sobol.draw(n_train).double()
    y_clean = branin_unit_box(X)
    y_noisy = y_clean + noise_std * torch.randn_like(y_clean)

    y_mean = y_noisy.mean()
    y_std = y_noisy.std().clamp_min(1e-12)
    y_scaled = (y_noisy - y_mean) / y_std
    return X, y_scaled, y_mean, y_std


@torch.no_grad()
def posterior_prob_below(pred_mean: torch.Tensor, pred_var: torch.Tensor, y_star: float | torch.Tensor):
    s = pred_var.clamp_min(1e-12).sqrt()
    z = (torch.as_tensor(y_star, dtype=pred_mean.dtype, device=pred_mean.device) - pred_mean) / s
    return normal_cdf(z)


@torch.no_grad()
def make_grid(n_per_dim: int = 200, dtype: torch.dtype = torch.float64):
    grid_1d = torch.linspace(0.0, 1.0, n_per_dim, dtype=dtype)
    xx, yy = torch.meshgrid(grid_1d, grid_1d, indexing="ij")
    X = torch.stack([xx.reshape(-1), yy.reshape(-1)], dim=-1)
    return X, xx, yy


@torch.no_grad()
def add_common_overlays(ax, x_train: torch.Tensor, minima: torch.Tensor):
    ax.scatter(
        x_train[:, 0].cpu(),
        x_train[:, 1].cpu(),
        s=15,
        facecolors="white",
        edgecolors="black",
        linewidths=0.45,
        alpha=0.95,
        zorder=4,
    )
    ax.scatter(
        minima[:, 0].cpu(),
        minima[:, 1].cpu(),
        s=110,
        marker="*",
        facecolors="#d62728",
        edgecolors="white",
        linewidths=0.8,
        zorder=6,
    )
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.set_aspect("equal", adjustable="box")
    ax.set_box_aspect(1)
    ax.set_xlabel("x1")
    ax.set_ylabel("x2")
    ticks = [0.0, 0.25, 0.50, 0.75, 1.0]
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)


@torch.no_grad()
def draw_panel(
    ax,
    xx: torch.Tensor,
    yy: torch.Tensor,
    values: torch.Tensor,
    *,
    title: str,
    cmap: str,
    x_train: torch.Tensor,
    minima: torch.Tensor,
    truth_boundary_values: torch.Tensor | None = None,
    truth_boundary_level: float | None = None,
    pred_boundary_level: float | None = None,
    levels: int = 70,
    vmin: float | None = None,
    vmax: float | None = None,
):
    arr = values.reshape(xx.shape)
    contour = ax.contourf(
        xx.cpu().numpy(),
        yy.cpu().numpy(),
        arr.cpu().numpy(),
        levels=levels,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
    )

    if pred_boundary_level is not None:
        ax.contour(
            xx.cpu().numpy(),
            yy.cpu().numpy(),
            arr.cpu().numpy(),
            levels=[pred_boundary_level],
            colors="black",
            linewidths=1.15,
            linestyles="solid",
            zorder=5,
        )

    if truth_boundary_values is not None and truth_boundary_level is not None:
        ax.contour(
            xx.cpu().numpy(),
            yy.cpu().numpy(),
            truth_boundary_values.reshape(xx.shape).cpu().numpy(),
            levels=[truth_boundary_level],
            colors="white",
            linewidths=1.6,
            linestyles="dashed",
            zorder=5,
        )

    add_common_overlays(ax, x_train, minima)
    ax.set_title(title, pad=8)
    return contour


def main():
    plt.rcParams.update(
        {
            "font.size": 11,
            "axes.titlesize": 13,
            "axes.labelsize": 12,
            "figure.titlesize": 15,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
        }
    )

    torch.manual_seed(0)

    x_train, y_train, y_mean_raw, y_std_raw = make_branin_dataset(n_train=80, noise_std=0.05, seed=7)

    y_star = 0.5
    y_star_raw = (y_mean_raw + y_star * y_std_raw).item()
    M = 40
    num_constraint_points = 300

    print("Training 2D standard sparse GP...")
    t0 = time.perf_counter()
    res_std = fit_vfe_sparse_gp(
        train_X=x_train,
        train_Y=y_train,
        noise=1e-2,
        train_noise=True,
        M=M,
        verbose=False,
        y_star=None,
        seed_for_init=2024,
        inducing_seed=2025,
    )
    t_std = time.perf_counter() - t0

    noise_star = float(res_std.likelihood.noise.detach().cpu().item())

    print("Training 2D step-term constrained GP...")
    t0 = time.perf_counter()
    res_step = fit_vfe_sparse_gp(
        train_X=x_train,
        train_Y=y_train,
        noise=noise_star,
        train_noise=False,
        M=M,
        verbose=False,
        y_star=y_star,
        num_constraint_points=num_constraint_points,
        constraint_sampling="sobol",
        seed_for_init=2024,
        inducing_seed=2025,
        base_gp=res_std.model,
        resample_Xc_each_eval=True,
    )
    t_step = time.perf_counter() - t0

    n_grid = 200
    X_grid, xx, yy = make_grid(n_per_dim=n_grid, dtype=x_train.dtype)
    minima = branin_true_minima_unit_box(dtype=x_train.dtype)
    true_branin_grid = branin_unit_box(X_grid)

    pred_std = predictive_distribution(res_std.model, res_std.likelihood, X_grid, observation_noise=False)
    pred_step = predictive_distribution(res_step.model, res_step.likelihood, X_grid, observation_noise=False)

    std_mean = pred_std.mean
    step_mean = pred_step.mean
    std_std = pred_std.variance.clamp_min(1e-12).sqrt()
    step_std = pred_step.variance.clamp_min(1e-12).sqrt()

    p_std = posterior_prob_below(std_mean, pred_std.variance, y_star)
    p_step = posterior_prob_below(step_mean, pred_step.variance, y_star)

    frac_std_above = (std_mean > y_star).double().mean().item()
    frac_step_above = (step_mean > y_star).double().mean().item()

    mean_abs_diff = (step_mean - std_mean).abs().mean().item()
    max_abs_diff = (step_mean - std_mean).abs().max().item()

    std_abs_diff = (step_std - std_std).abs().mean().item()
    max_std_diff = (step_std - std_std).abs().max().item()

    var_reduction = 1.0 - pred_step.variance / pred_std.variance.clamp_min(1e-12)

    print("\nSummary (2D Branin: standard vs step-term)")
    print(f"Standard sparse GP time                 : {t_std:.3f} s")
    print(f"Step-term constrained GP time          : {t_step:.3f} s")

    print("\nFinal losses")
    print(f"  Standard sparse GP                   : {res_std.losses[-1].item():.6f}")
    print(f"  Step-term constrained GP             : {res_step.losses[-1].item():.6f}")

    print("\nProbability P(f(x) < y*) on 2D grid")
    print(
        f"  Standard sparse GP                   : mean={p_std.mean().item():.3f}, "
        f"min={p_std.min().item():.3f}, max={p_std.max().item():.3f}"
    )
    print(
        f"  Step-term constrained GP             : mean={p_step.mean().item():.3f}, "
        f"min={p_step.min().item():.3f}, max={p_step.max().item():.3f}"
    )

    print("\nFraction of the 2D grid with posterior mean above y*")
    print(f"  Standard sparse GP                   : {frac_std_above:.3f}")
    print(f"  Step-term constrained GP             : {frac_step_above:.3f}")

    print("\nPointwise differences on the 2D grid")
    print(f"  Mean abs diff                        : {mean_abs_diff:.6f}")
    print(f"  Mean max diff                        : {max_abs_diff:.6f}")
    print(f"  Std abs diff                         : {std_abs_diff:.6f}")
    print(f"  Std max diff                         : {max_std_diff:.6f}")

    print("\nVariance reduction of step-term GP w.r.t. standard GP")
    print(f"  Mean reduction                       : {var_reduction.mean().item():.6f}")
    print(f"  Max reduction                        : {var_reduction.max().item():.6f}")

    mean_maps = [std_mean.cpu(), step_mean.cpu()]
    std_maps = [std_std.cpu(), step_std.cpu()]
    prob_maps = [p_std.cpu(), p_step.cpu()]

    mean_vmin = min(v.min().item() for v in mean_maps)
    mean_vmax = max(v.max().item() for v in mean_maps)
    std_vmin = min(v.min().item() for v in std_maps)
    std_vmax = max(v.max().item() for v in std_maps)

    fig = plt.figure(figsize=(15.8, 16.8), constrained_layout=False)
    gs = fig.add_gridspec(
        3,
        3,
        width_ratios=[1.0, 1.0, 0.045],
        height_ratios=[1.0, 1.0, 1.0],
        left=0.06,
        right=0.92,
        bottom=0.06,
        top=0.89,
        wspace=0.14,
        hspace=0.24,
    )

    axes = [[fig.add_subplot(gs[i, j]) for j in range(2)] for i in range(3)]
    cax_mean = fig.add_subplot(gs[0, 2])
    cax_std = fig.add_subplot(gs[1, 2])
    cax_prob = fig.add_subplot(gs[2, 2])

    fig.suptitle(
        "Branin on [0,1]^2: standard sparse GP vs step-term conditioning",
        y=0.965,
        fontweight="semibold",
    )
    fig.text(
        0.5,
        0.936,
        f"White dashed = true boundary f(x)=y*  |  black solid = model boundary  |  raw y*={y_star_raw:.2f}  |  red stars = true minima",
        ha="center",
        va="center",
        fontsize=11,
    )

    img00 = draw_panel(
        axes[0][0], xx, yy, mean_maps[0],
        title="Mean · standard sparse GP",
        cmap="viridis",
        x_train=x_train,
        minima=minima,
        truth_boundary_values=true_branin_grid,
        truth_boundary_level=y_star_raw,
        pred_boundary_level=y_star,
        vmin=mean_vmin,
        vmax=mean_vmax,
    )
    img01 = draw_panel(
        axes[0][1], xx, yy, mean_maps[1],
        title="Mean · step-term GP",
        cmap="viridis",
        x_train=x_train,
        minima=minima,
        truth_boundary_values=true_branin_grid,
        truth_boundary_level=y_star_raw,
        pred_boundary_level=y_star,
        vmin=mean_vmin,
        vmax=mean_vmax,
    )

    img10 = draw_panel(
        axes[1][0], xx, yy, std_maps[0],
        title="Std · standard sparse GP",
        cmap="magma",
        x_train=x_train,
        minima=minima,
        truth_boundary_values=true_branin_grid,
        truth_boundary_level=y_star_raw,
        vmin=std_vmin,
        vmax=std_vmax,
    )
    img11 = draw_panel(
        axes[1][1], xx, yy, std_maps[1],
        title="Std · step-term GP",
        cmap="magma",
        x_train=x_train,
        minima=minima,
        truth_boundary_values=true_branin_grid,
        truth_boundary_level=y_star_raw,
        vmin=std_vmin,
        vmax=std_vmax,
    )

    img20 = draw_panel(
        axes[2][0], xx, yy, prob_maps[0],
        title="P(f(x) < y*) · standard GP",
        cmap="viridis",
        x_train=x_train,
        minima=minima,
        truth_boundary_values=true_branin_grid,
        truth_boundary_level=y_star_raw,
        pred_boundary_level=0.5,
        vmin=0.0,
        vmax=1.0,
    )
    img21 = draw_panel(
        axes[2][1], xx, yy, prob_maps[1],
        title="P(f(x) < y*) · step-term GP",
        cmap="viridis",
        x_train=x_train,
        minima=minima,
        truth_boundary_values=true_branin_grid,
        truth_boundary_level=y_star_raw,
        pred_boundary_level=0.5,
        vmin=0.0,
        vmax=1.0,
    )

    cbar_mean = fig.colorbar(img01, cax=cax_mean)
    cbar_mean.set_label("Posterior mean")
    cbar_std = fig.colorbar(img11, cax=cax_std)
    cbar_std.set_label("Posterior std")
    cbar_prob = fig.colorbar(img21, cax=cax_prob)
    cbar_prob.set_label("Posterior probability")

    plt.show()


if __name__ == "__main__":
    main()
