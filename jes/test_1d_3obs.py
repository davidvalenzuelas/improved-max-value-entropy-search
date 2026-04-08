#!/usr/bin/env python3
# coding: utf-8
"""
1D test with 3 observations for the constrained VFE sparse GP

It compares standard ELBO vs standard ELBO with added step constraint term
using only 3 observations. The synthetic problem is generated in the same
spirit as script_BO.R: we sample a latent 1D function from a GP on [-5, 5]
and then keep only 3 randomly selected observations.

The y* value is sampled from posterior optimum samples of the standard sparse
model, without using BoTorch. For visualization we select a lower quantile of
those sampled optimum outputs, so the step constraint is not trivial.

The structure of this file follows the Snelson 1D test as closely as possible.

Authors: Daniel Hernández-Lobato, David Valenzuela Sánchez
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

# Number of training points (observations) we keep from the sampled function
NUM_TRAIN = 3

INIT_NOISE = 1e-4
NUM_CONSTRAINT_POINTS = 100
EPSILON = 1e-4
TRAINING_ITER = 500
LR = 1e-2
NUM_SOLUTION_SAMPLES = 512
YSTAR_POSTERIOR_QUANTILE = 0.2
PLOT_STD_MULT = 1.0


def kernel(x: torch.Tensor, y: torch.Tensor, lengthscale: float = 2.0,
    variance: float = 1.0) -> torch.Tensor:
    """This function defines a RBF kernel"""
    # Scales inputs by lengthscale
    x_scaled = x / lengthscale
    y_scaled = y / lengthscale
    
    # Computes squared distance matrix
    sqdist = (x_scaled[:,None,:] - y_scaled[None,:,:]).pow(2).sum(dim=-1)
    # Returns the RBF kernel matrix
    return variance * torch.exp(-0.5 * sqdist)


@torch.no_grad()
def generate_3obs_problem(num_grid: int = 1000, jitter: float=1e-7):
    """ This function generates the synthetic 1D problem used in this test.
    It samples a latent function from a GP with an RBF kernel on [-5, 5]
    and selects some training points uniformly at random from the sampled
    function """
    dtype = torch.float64
    # Creates a grid of points in [-5, 5]
    x_grid = torch.linspace(-5.0, 5.0, num_grid, dtype=dtype).unsqueeze(-1)
    
    # Builds the GP covariance matrix on the grid
    sigma = kernel(x_grid, x_grid) + jitter * torch.eye(num_grid, dtype=dtype)
    # Cholesky decomposition for sampling from the GP prior
    L = torch.linalg.cholesky(sigma)
    # Draws one latent function sample
    f_true = L @ torch.randn(num_grid, dtype=dtype)
    
    # Randomly selects training points from the grid
    p_sel = np.sort(np.random.choice(np.arange(num_grid), size=NUM_TRAIN, replace=False))
    p_sel = torch.tensor(p_sel, dtype=torch.long)
    
    # Extracts the training inputs and their outputs
    x_train = x_grid[p_sel].contiguous()
    y_train = f_true[p_sel].contiguous()
    
    return x_grid, f_true, x_train, y_train, p_sel


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


@torch.no_grad()
def sample_solution_outputs_from_model(model, likelihood, x_grid,
    num_samples: int = NUM_SOLUTION_SAMPLES):
    """This function samples posterior solutions (x*, y*) on a dense 1D grid.

    The spirit is the same as in loop_BO.py: sample possible optima from the
    posterior after fitting the model. Here we do it directly from the latent
    posterior on a dense 1D grid, without using BoTorch.
    """
    pred = predictive_distribution(model, likelihood, x_grid, observation_noise=False)
    samples = pred.rsample(torch.Size([num_samples]))

    sampled_y_stars, idxs = samples.max(dim=-1)
    sampled_x_stars = x_grid[idxs].squeeze(-1)

    return sampled_x_stars.clone(), sampled_y_stars.clone()


@torch.no_grad()
def choose_visual_y_star(sampled_x_stars: torch.Tensor,
    sampled_y_stars: torch.Tensor,
    quantile: float = YSTAR_POSTERIOR_QUANTILE):
    """This function selects y* from the sampled optimum outputs.

    To keep the test visually informative, it takes a lower quantile of the
    sampled optimum outputs instead of taking an arbitrary sample that could be
    too optimistic. The selected y* still comes from the posterior optimum
    samples of the standard sparse model.
    """
    sorted_idx = torch.argsort(sampled_y_stars)
    rank = int(quantile * (sorted_idx.numel() - 1))
    chosen_idx = sorted_idx[rank]

    return {
        "chosen_idx": int(chosen_idx.item()),
        "x_star": float(sampled_x_stars[chosen_idx].item()),
        "y_star": float(sampled_y_stars[chosen_idx].item()),
        "rank": int(rank),
        "num_samples": int(sorted_idx.numel()),
    }





# -------------------------
# Plotting helpers
# -------------------------
@torch.no_grad()
def plot_two_predictive_distributions(ax, model_a, likelihood_a, model_b,
    likelihood_b, x_grid, f_true, x_train, y_train, inducing_points, title,
    label_a="Base GP", label_b="Sparse GP after init", y_star=None,
    x_star=None):
    """This function plots two predictive distributions on the same axis"""
    pred_a = predictive_distribution(model_a, likelihood_a, x_grid,
        observation_noise=False)
    pred_b = predictive_distribution(model_b, likelihood_b, x_grid,
        observation_noise=False)

    mean_a = pred_a.mean.cpu()
    std_a = pred_a.variance.sqrt().cpu()
    mean_b = pred_b.mean.cpu()
    std_b = pred_b.variance.sqrt().cpu()

    x_np = x_grid.squeeze(-1).cpu().numpy()
    ax.plot(x_np, f_true.cpu().numpy(), color="0.65", linewidth=1.0,
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
        ax.axvline(float(x_star), color="0.4", linestyle=":", linewidth=1.0,
            label="sampled x*")

    ax.set_title(title)
    ax.legend(fontsize=9)


@torch.no_grad()
def plot_mean_and_band(ax, model, likelihood, x_grid, f_true, x_train, y_train,
    inducing_points, title, y_star=None, x_star=None):
    """This function plots the predictive mean and confidence band of a model"""
    pred = predictive_distribution(model, likelihood, x_grid,
        observation_noise=False)
    mean = pred.mean.cpu()
    std = pred.variance.sqrt().cpu()

    x_np = x_grid.squeeze(-1).cpu().numpy()
    ax.plot(x_np, f_true.cpu().numpy(), color="0.65", linewidth=1.0,
        label="True latent f")
    ax.plot(x_train.squeeze(-1).cpu().numpy(), y_train.cpu().numpy(), "k*",
        markersize=8, label="Training data")

    Z = inducing_points.detach().cpu()
    ax.plot(Z.squeeze(-1).numpy(), np.zeros(Z.shape[0]), "rx", markersize=6,
        mew=2, label="Inducing points")

    ax.plot(x_np, mean.numpy(), label="Mean")
    ax.fill_between(x_np, (mean - PLOT_STD_MULT * std).numpy(),
        (mean + PLOT_STD_MULT * std).numpy(), alpha=0.30,
        label=f"Confidence (±{PLOT_STD_MULT:.0f}σ)")

    if y_star is not None:
        ax.axhline(float(y_star), color="lightgreen", linestyle="--", label="y*")
    if x_star is not None:
        ax.axvline(float(x_star), color="0.4", linestyle=":", linewidth=1.0,
            label="sampled x*")

    ax.set_title(title)
    ax.legend(fontsize=9)




@torch.no_grad()
def get_common_plot_limits(x_grid, f_true, x_train, y_train, res_std, res_con,
    init_model, y_star=None):
    """This function computes common x/y limits for the three subplots.

    We use the same vertical scale in all panels so the visual comparison
    between initialization, standard ELBO and constrained ELBO is fair.
    """
    x_np = x_grid.squeeze(-1).cpu().numpy()

    pred_std = predictive_distribution(res_std.model, res_std.likelihood, x_grid,
        observation_noise=False)
    pred_con = predictive_distribution(res_con.model, res_con.likelihood, x_grid,
        observation_noise=False)
    pred_init = predictive_distribution(init_model, res_std.likelihood, x_grid,
        observation_noise=False)

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


# -------------------------
# Main
# -------------------------
def main():
    # Generates synthetic 1D problem with only 3 observations
    x_grid, f_true, x_train, y_train, p_sel = generate_3obs_problem()

    # Number of data points and inducing points
    N = x_train.shape[0]
    M = N

    # Sets inducing points to training data locations, as in Snelson
    fixed_inducing = x_train.contiguous()

    # Samples fixed constraint points uniformly on the full domain [-5, 5]
    x_min, x_max = x_grid.min(), x_grid.max()
    Xc_eval = x_min + (x_max - x_min) * torch.rand(
        NUM_CONSTRAINT_POINTS, 1, dtype=x_grid.dtype, device=x_grid.device
    )

    # Fits standard VFE sparse GP model using the same training routine as in
    # the Snelson test
    res_std = fit_vfe_sparse_gp(
        train_X=x_train,
        train_Y=y_train,
        noise=INIT_NOISE,
        train_noise=True,
        M=M,
        y_star=None,
        fixed_inducing_points=fixed_inducing,
        seed_for_init=2024,
        training_iter=TRAINING_ITER,
        lr=LR,
        verbose=False,
    )

    # Fixes noise for constrained model
    noise_star = float(res_std.likelihood.noise.detach().cpu().item())

    # Samples posterior optima from the standard sparse model and selects a
    # convenient y* for visualization
    sampled_x_stars, sampled_y_stars = sample_solution_outputs_from_model(
        res_std.model,
        res_std.likelihood,
        x_grid,
        num_samples=NUM_SOLUTION_SAMPLES,
    )
    y_star_info = choose_visual_y_star(
        sampled_x_stars,
        sampled_y_stars,
        quantile=YSTAR_POSTERIOR_QUANTILE,
    )
    y_star = y_star_info["y_star"]
    x_star = y_star_info["x_star"]

    # Creates base GP model and obtains its posterior to use it as initialization
    # for the constrained model
    base_gp = res_std.model
    base_gp.likelihood = res_std.likelihood

    # Builds sparse GP model just after initialization from the base GP posterior
    # at the inducing points
    init_model = build_sparse_model_just_initialized(base_gp, res_std.inducing_points)
    
    # Fits constrained VFE sparse GP model
    res_con = fit_vfe_sparse_gp(
        train_X=x_train,
        train_Y=y_train,
        noise=noise_star,
        train_noise=False,
        M=M,
        y_star=y_star,
        epsilon=EPSILON,
        Xc=Xc_eval,
        num_constraint_points=NUM_CONSTRAINT_POINTS,
        fixed_inducing_points=fixed_inducing,
        seed_for_init=2024,
        base_gp=base_gp,
        training_iter=TRAINING_ITER,
        lr=LR,
        verbose=False,
        resample_Xc_each_eval=False,
    )

    # Obtains mean difference between models
    print("\nPrinting some results...")
    print("Selected training grid indices:", p_sel.cpu().numpy())
    print("Training x:", x_train.squeeze(-1).cpu().numpy())
    print("Training y:", y_train.cpu().numpy())
    #print("Latent sample flipped for visualization:", flipped_for_demo)
    print(f"Sampled optimum x*: {x_star:.6f}")
    print(f"Selected y*: {y_star:.6f}")
    print(
        f"Selected y* comes from posterior optimum quantile "
        f"q={YSTAR_POSTERIOR_QUANTILE:.2f} (rank {y_star_info['rank'] + 1}/"
        f"{y_star_info['num_samples']})"
    )
    print(
        f"Sampled optimum outputs: mean={sampled_y_stars.mean().item():.6f}, "
        f"std={sampled_y_stars.std().item():.6f}, "
        f"min={sampled_y_stars.min().item():.6f}, "
        f"max={sampled_y_stars.max().item():.6f}"
    )

    obtain_mean_difference(res_std, res_con, x_grid)

    # Computes P(f(Xc) < y*) under both models
    y_star_t = torch.tensor(y_star, dtype=x_grid.dtype, device=x_grid.device)
    p_std = prob_f_below_y_star(res_std.model, Xc_eval, y_star_t)
    p_con = prob_f_below_y_star(res_con.model, Xc_eval, y_star_t)

    print("\nP(f(Xc) < y*) under q(f):")
    print(f"  Standard   : mean={p_std[0]:.3f}, min={p_std[1]:.3f}, max={p_std[2]:.3f}")
    print(f"  Constraint : mean={p_con[0]:.3f}, min={p_con[1]:.3f}, max={p_con[2]:.3f}")
    print("\nNoise:")
    print("  Standard learned:", noise_star)
    print("  Constraint fixed :", res_con.likelihood.noise.item())

    # Three Snelson-style plots in one row
    fig, axes = plt.subplots(1, 3, figsize=(19, 5.2), sharex=True, sharey=True)

    plot_two_predictive_distributions(
        axes[0],
        base_gp,
        res_std.likelihood,
        init_model,
        res_std.likelihood,
        x_grid,
        f_true,
        x_train,
        y_train,
        res_std.inducing_points,
        title="Base GP vs sparse GP just after initialization",
        label_a="Base GP",
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
        title="Standard ELBO + Step constraint term",
        y_star=y_star,
        x_star=x_star,
    )

    x_limits, y_limits = get_common_plot_limits(
        x_grid=x_grid,
        f_true=f_true,
        x_train=x_train,
        y_train=y_train,
        res_std=res_std,
        res_con=res_con,
        init_model=init_model,
        y_star=y_star,
    )
    for ax in axes:
        ax.set_xlim(*x_limits)
        ax.set_ylim(*y_limits)

    fig.suptitle("1D test with 3 observations", fontsize=16)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()


if __name__ == "__main__":
    main()
