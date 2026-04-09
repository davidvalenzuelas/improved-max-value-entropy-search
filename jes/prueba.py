#!/usr/bin/env python3
# coding: utf-8
"""1D comparison between:
1) a standard sparse GP conditioned on (x*, y*),
2) the JES-style truncated predictive distribution, and
3) the modified VFE sparse GP with the step constraint term.
"""
from __future__ import annotations

import math
import numpy as np
import torch
import matplotlib.pyplot as plt

from modified_vfe_sparse_gp import (
    fit_vfe_sparse_gp,
    predictive_distribution as sparse_predictive_distribution,
    normal_cdf,
    VFESparseGP,
    build_init_dist_from_base_gp,
)

from botorch.models.gp_regression import SingleTaskGP
from botorch.fit import fit_gpytorch_mll
from botorch.acquisition.utils import get_optimal_samples
from gpytorch.mlls.exact_marginal_log_likelihood import ExactMarginalLogLikelihood

# Number of training points (observations) we keep from the sampled function
NUM_TRAIN = 5
# Multiplier for the standard deviation when plotting confidence bands
PLOT_STD_MULT = 1.0


def kernel(
    x: torch.Tensor,
    y: torch.Tensor,
    lengthscale: float = 2.0,
    variance: float = 1.0,
) -> torch.Tensor:
    """RBF kernel used in the synthetic 1D test."""
    x_scaled = x / lengthscale
    y_scaled = y / lengthscale
    sqdist = (x_scaled[:, None, :] - y_scaled[None, :, :]).pow(2).sum(dim=-1)
    return variance * torch.exp(-0.5 * sqdist)


@torch.no_grad()
def generate_5obs_problem(
    num_grid: int = 1000,
    jitter: float = 1e-7,
    seed_latent: int = 123,
    seed_train: int = 5,
):
    """Generate the 1D synthetic problem used in the comparison."""
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


@torch.no_grad()
def predictive_distribution(model, likelihood, grid: torch.Tensor, observation_noise: bool = False):
    """Return the predictive distribution of either a BoTorch GP or a sparse GP."""
    if hasattr(model, "posterior"):
        return model.posterior(grid, observation_noise=observation_noise)
    return sparse_predictive_distribution(model, likelihood, grid, observation_noise=observation_noise)


@torch.no_grad()
def build_sparse_model_just_initialized(base_gp, inducing_points: torch.Tensor) -> VFESparseGP:
    """Build a sparse GP just after initialization from a base GP posterior."""
    init_dist = build_init_dist_from_base_gp(base_gp, inducing_points)
    model = VFESparseGP(
        inducing_points=inducing_points,
        init_dist=init_dist,
        mean_module=base_gp.mean_module,
        covar_module=base_gp.covar_module,
    )
    model = model.to(dtype=inducing_points.dtype, device=inducing_points.device)
    model.variational_strategy.inducing_points.requires_grad_(False)
    return model

def fit_singletask_gp(train_X: torch.Tensor, train_Y: torch.Tensor, init_noise: float) -> SingleTaskGP:
    """Fit the exact GP used for BO and for sparse-GP initialization."""
    train_X = train_X.double()
    train_Y = train_Y.double()
    if train_X.ndim == 1:
        train_X = train_X.unsqueeze(-1)
    if train_Y.ndim == 1:
        train_Y = train_Y.unsqueeze(-1)

    base_gp = SingleTaskGP(train_X, train_Y, outcome_transform=None)
    base_gp = base_gp.to(dtype=train_X.dtype, device=train_X.device)

    base_gp.likelihood.noise = torch.as_tensor(init_noise, dtype=train_X.dtype, device=train_X.device)
    base_gp.likelihood.noise_covar.raw_noise.requires_grad_(False)

    mll = ExactMarginalLogLikelihood(base_gp.likelihood, base_gp)
    fit_gpytorch_mll(mll)

    base_gp.eval()
    base_gp.likelihood.eval()
    return base_gp


def sample_solution_outputs_from_model(
    base_gp,
    bounds,
    num_samples: int = 512,
    seed_posterior_samples: int = 1,
):
    """Sample posterior optimal pairs (x*, y*) from the exact GP."""
    torch.manual_seed(seed_posterior_samples)
    optimal_inputs, optimal_outputs = get_optimal_samples(
        model=base_gp,
        bounds=bounds,
        num_optima=num_samples,
    )
    sampled_x_stars = optimal_inputs.reshape(num_samples, -1).squeeze(-1).detach()
    sampled_y_stars = optimal_outputs.reshape(num_samples).detach()
    return sampled_x_stars, sampled_y_stars


@torch.no_grad()
def choose_y_star(
    sampled_x_stars: torch.Tensor,
    sampled_y_stars: torch.Tensor,
    seed_star_selection: int = 4,
):
    """Choose one sampled pair (x*, y*) reproducibly."""
    n = sampled_y_stars.numel()
    g = torch.Generator(device=sampled_y_stars.device)
    g.manual_seed(seed_star_selection)
    chosen_idx = torch.randint(low=0, high=n, size=(1,), generator=g).item()
    return {
        "chosen_idx": int(chosen_idx),
        "x_star": float(sampled_x_stars[chosen_idx].item()),
        "y_star": float(sampled_y_stars[chosen_idx].item()),
        "num_samples": int(n),
    }


@torch.no_grad()
def condition_base_gp_on_optimum(base_gp: SingleTaskGP, x_star: float, y_star: float):
    dtype = next(base_gp.parameters()).dtype
    device = next(base_gp.parameters()).device

    x_star_t = torch.tensor([[x_star]], dtype=dtype, device=device)
    y_star_t = torch.tensor([[y_star]], dtype=dtype, device=device)

    base_gp.eval()
    base_gp.likelihood.eval()

    # Inicializa las cachés necesarias para ExactGP.get_fantasy_model
    _ = base_gp.posterior(x_star_t, observation_noise=False)
    # alternativa equivalente:
    # _ = base_gp(x_star_t)

    conditioned_gp = base_gp.condition_on_observations(X=x_star_t, Y=y_star_t)
    conditioned_gp = conditioned_gp.to(dtype=dtype, device=device)
    conditioned_gp.eval()
    conditioned_gp.likelihood.eval()

    return conditioned_gp, x_star_t, y_star_t


@torch.no_grad()
def marginal_mean_variance(model, likelihood, X: torch.Tensor, observation_noise: bool = False):
    """Return pointwise predictive mean and variance."""
    post = predictive_distribution(model, likelihood, X, observation_noise=observation_noise)
    mean = post.mean
    variance = post.variance.clamp_min(1e-12)
    return mean, variance


@torch.no_grad()
def extract_noise_variance(model, likelihood, X: torch.Tensor) -> torch.Tensor:
    """Extract the marginal observation-noise contribution at X."""
    _, var_f = marginal_mean_variance(model, likelihood, X, observation_noise=False)
    _, var_y = marginal_mean_variance(model, likelihood, X, observation_noise=True)
    return (var_y - var_f).clamp_min(1e-12)


def normal_pdf(z: torch.Tensor) -> torch.Tensor:
    """Standard normal pdf."""
    return torch.exp(-0.5 * z.pow(2)) / math.sqrt(2.0 * math.pi)


@torch.no_grad()
def truncated_upper_normal_moments(
    mean: torch.Tensor,
    variance: torch.Tensor,
    upper: torch.Tensor,
    eps: float = 1e-12,
):
    """Moments of X | X < upper, with X ~ N(mean, variance)."""
    variance = variance.clamp_min(eps)
    std = variance.sqrt()
    beta = (upper - mean) / std

    Phi = normal_cdf(beta).clamp_min(eps)
    phi = normal_pdf(beta)
    lam = phi / Phi

    mean_trunc = mean - std * lam
    var_trunc = variance * (1.0 - beta * lam - lam.pow(2)).clamp_min(eps)
    return mean_trunc, var_trunc


@torch.no_grad()
def jes_truncated_predictive_moments(
    model,
    likelihood,
    X: torch.Tensor,
    y_star: float | torch.Tensor,
    observation_noise: bool = False,
):
    """JES-style predictive moments after upper truncation at y*.

    For the latent function f(x), the predictive distribution is the exact
    truncated normal obtained from the Gaussian predictive marginal.

    For noisy observations y(x), we follow the JES approximation mentioned by
    the tutor: moment-match the truncated latent distribution by a Gaussian and
    then add the observation-noise variance.
    """
    mean_f, var_f = marginal_mean_variance(model, likelihood, X, observation_noise=False)
    y_star_t = torch.as_tensor(y_star, dtype=mean_f.dtype, device=mean_f.device)
    mean_trunc, var_trunc = truncated_upper_normal_moments(mean_f, var_f, y_star_t)

    if not observation_noise:
        return mean_trunc, var_trunc

    noise_var = extract_noise_variance(model, likelihood, X)
    return mean_trunc, var_trunc + noise_var


@torch.no_grad()
def gaussian_prob_less_than(mean: torch.Tensor, variance: torch.Tensor, threshold: torch.Tensor):
    """Gaussian probability P(X < threshold)."""
    std = variance.clamp_min(1e-12).sqrt()
    z = (threshold - mean) / std
    return normal_cdf(z)


@torch.no_grad()
def summarize_prob_less_than(model, likelihood, X: torch.Tensor, y_star: torch.Tensor):
    """Summarize P(f(X) < y*) under a Gaussian predictive distribution."""
    mean, variance = marginal_mean_variance(model, likelihood, X, observation_noise=False)
    p_less = gaussian_prob_less_than(mean, variance, y_star)
    return p_less.mean().item(), p_less.min().item(), p_less.max().item()


@torch.no_grad()
def summarize_prob_less_than_from_moments(mean: torch.Tensor, variance: torch.Tensor, y_star: torch.Tensor):
    """Summarize P(X < y*) from provided Gaussian moments."""
    p_less = gaussian_prob_less_than(mean, variance, y_star)
    return p_less.mean().item(), p_less.min().item(), p_less.max().item()


@torch.no_grad()
def format_diff_stats(name_a: str, mean_a: torch.Tensor, var_a: torch.Tensor,
    name_b: str, mean_b: torch.Tensor, var_b: torch.Tensor):
    """Print mean/variance differences between two predictive summaries."""
    diff_mean = (mean_a - mean_b).abs()
    diff_var = (var_a - var_b).abs()
    print(f"\n{name_a} vs {name_b} on the grid:")
    print(f"  Mean abs diff: {diff_mean.mean().item():.6f}")
    print(f"  Max  abs diff: {diff_mean.max().item():.6f}")
    print(f"  Var  abs diff: {diff_var.mean().item():.6f}")
    print(f"  Max  var diff: {diff_var.max().item():.6f}")


@torch.no_grad()
def pointwise_normal_pdf_grid(mean: torch.Tensor, variance: torch.Tensor,
    y_values: torch.Tensor) -> torch.Tensor:
    """Evaluate the pdf of N(mean, variance) on a 1D grid of y-values."""
    std = variance.clamp_min(1e-12).sqrt()
    z = (y_values - mean) / std
    return normal_pdf(z) / std


@torch.no_grad()
def pointwise_upper_truncated_pdf_grid(mean: torch.Tensor, variance: torch.Tensor,
    upper: torch.Tensor, y_values: torch.Tensor) -> torch.Tensor:
    """Evaluate the exact upper-truncated normal pdf on a 1D y-grid."""
    std = variance.clamp_min(1e-12).sqrt()
    beta = (upper - mean) / std
    norm = normal_cdf(beta).clamp_min(1e-12)
    z = (y_values - mean) / std
    base_pdf = normal_pdf(z) / std
    mask = (y_values <= upper).to(base_pdf.dtype)
    return mask * base_pdf / norm


@torch.no_grad()
def get_common_plot_limits(curves: list[np.ndarray], y_star: float | None = None):
    """Return common y-axis limits for a set of curves."""
    if y_star is not None:
        curves = curves + [np.array([float(y_star)])]
    y_min = min(float(np.min(c)) for c in curves)
    y_max = max(float(np.max(c)) for c in curves)
    y_pad = 0.08 * max(1e-6, y_max - y_min)
    return y_min - y_pad, y_max + y_pad


@torch.no_grad()
def plot_mean_and_band(
    ax,
    x_grid: torch.Tensor,
    mean: torch.Tensor,
    std: torch.Tensor,
    f_true: torch.Tensor,
    x_obs: torch.Tensor,
    y_obs: torch.Tensor,
    inducing_points: torch.Tensor,
    title: str,
    x_star: float,
    y_star: float,
    x_pseudo: torch.Tensor | None = None,
    y_pseudo: torch.Tensor | None = None,
    band_label: str | None = None,
):
    """Plot predictive mean and a moment band."""
    x_np = x_grid.squeeze(-1).cpu().numpy()
    mean_np = mean.reshape(-1).cpu().numpy()
    std_np = std.reshape(-1).cpu().numpy()

    ax.plot(x_np, f_true.reshape(-1).cpu().numpy(), color="0.65", linewidth=0.7, label="True latent f")
    ax.plot(x_obs.squeeze(-1).cpu().numpy(), y_obs.reshape(-1).cpu().numpy(), "k*", markersize=8, label="Observed data")

    if x_pseudo is not None and y_pseudo is not None:
        ax.plot(
            x_pseudo.squeeze(-1).cpu().numpy(),
            y_pseudo.reshape(-1).cpu().numpy(),
            marker="o",
            linestyle="None",
            color="tab:green",
            markersize=7,
            label="Pseudo-observation (x*, y*)",
        )

    Z = inducing_points.detach().cpu()
    ax.plot(Z.squeeze(-1).numpy(), np.zeros(Z.shape[0]), "rx", markersize=6, mew=2, label="Inducing points")
    ax.plot(x_np, mean_np, label="Mean")

    if band_label is None:
        band_label = f"Moment band (±{PLOT_STD_MULT:.0f} std)"
    ax.fill_between(
        x_np,
        mean_np - PLOT_STD_MULT * std_np,
        mean_np + PLOT_STD_MULT * std_np,
        alpha=0.25,
        label=band_label,
    )

    ax.axhline(float(y_star), color="lightgreen", linestyle="--", label="y*")
    ax.axvline(float(x_star), color="lightgreen", linestyle=":", label="x*")
    ax.set_title(title)
    ax.legend(fontsize=6, loc="lower left")


def main():
    # ------------------------------------------------------------------
    # 1) Synthetic problem and original exact GP used for BO.
    # ------------------------------------------------------------------
    x_grid, f_true, x_train, y_train = generate_5obs_problem()

    num_constraint_points = 100
    init_noise = 1e-6
    epsilon = 1e-4

    x_min, x_max = x_grid.min(), x_grid.max()
    Xc_eval = x_min + (x_max - x_min) * torch.rand(
        num_constraint_points, 1, dtype=x_grid.dtype, device=x_grid.device
    )

    base_gp = fit_singletask_gp(x_train, y_train, init_noise=init_noise)
    bounds = torch.stack([x_grid.min(dim=0).values, x_grid.max(dim=0).values], dim=0)

    sampled_x_stars, sampled_y_stars = sample_solution_outputs_from_model(base_gp=base_gp, bounds=bounds)
    y_star_info = choose_y_star(sampled_x_stars, sampled_y_stars)
    y_star = y_star_info["y_star"]
    x_star = y_star_info["x_star"]

    # ------------------------------------------------------------------
    # 2) Condition on the sampled optimum pair before any further step.
    # ------------------------------------------------------------------
    conditioned_base_gp, x_star_t, y_star_t_col = condition_base_gp_on_optimum(base_gp, x_star=x_star, y_star=y_star)
    y_star_t = y_star_t_col.reshape(())

    # The sparse models are now trained on the augmented data set that includes
    # the sampled optimum pair as an additional observation.
    train_X_aug = torch.cat([x_train, x_star_t.to(dtype=x_train.dtype, device=x_train.device)], dim=0)
    train_Y_aug = torch.cat([y_train, y_star_t_col.reshape(-1).to(dtype=y_train.dtype, device=y_train.device)], dim=0)
    fixed_inducing = train_X_aug.contiguous()
    M = fixed_inducing.shape[0]

    init_model = build_sparse_model_just_initialized(conditioned_base_gp, fixed_inducing)
    base_gp_noise = float(base_gp.likelihood.noise.detach().cpu().item())

    # ------------------------------------------------------------------
    # 3) Sparse models after conditioning on (x*, y*).
    # ------------------------------------------------------------------
    res_std = fit_vfe_sparse_gp(
        train_X=train_X_aug,
        train_Y=train_Y_aug,
        noise=base_gp_noise,
        train_noise=False,
        M=M,
        y_star=None,
        fixed_inducing_points=fixed_inducing,
        seed_for_init=2024,
        base_gp=conditioned_base_gp,
        verbose=False,
    )

    res_con = fit_vfe_sparse_gp(
        train_X=train_X_aug,
        train_Y=train_Y_aug,
        noise=base_gp_noise,
        train_noise=False,
        M=M,
        y_star=y_star,
        epsilon=epsilon,
        lower_bound=x_grid.min(dim=0).values,
        upper_bound=x_grid.max(dim=0).values,
        fixed_inducing_points=fixed_inducing,
        seed_for_init=2024,
        base_gp=conditioned_base_gp,
        verbose=False,
    )

    # ------------------------------------------------------------------
    # 4) JES-style truncation of the standard sparse predictive Gaussian.
    # ------------------------------------------------------------------
    mean_base_cond, var_base_cond = marginal_mean_variance(
        conditioned_base_gp,
        conditioned_base_gp.likelihood,
        x_grid,
        observation_noise=False,
    )
    mean_init, var_init = marginal_mean_variance(init_model, base_gp.likelihood, x_grid, observation_noise=False)
    mean_std, var_std = marginal_mean_variance(res_std.model, res_std.likelihood, x_grid, observation_noise=False)
    mean_con, var_con = marginal_mean_variance(res_con.model, res_con.likelihood, x_grid, observation_noise=False)
    mean_jes, var_jes = jes_truncated_predictive_moments(
        res_std.model,
        res_std.likelihood,
        x_grid,
        y_star=y_star,
        observation_noise=False,
    )
    mean_jes_y, var_jes_y = jes_truncated_predictive_moments(
        res_std.model,
        res_std.likelihood,
        x_grid,
        y_star=y_star,
        observation_noise=True,
    )

    # ------------------------------------------------------------------
    # 5) Numerical summaries.
    # ------------------------------------------------------------------
    print("\n============================================================")
    print("1D comparison: standard sparse GP vs JES truncation vs modified VFE")
    print("============================================================")
    print("Original observed x:", x_train.squeeze(-1).cpu().numpy())
    print("Original observed y:", y_train.cpu().numpy())
    print(f"Selected sampled x*: {x_star:.6f}")
    print(f"Selected sampled y*: {y_star:.6f}")
    print("Augmented training x (including x*):", train_X_aug.squeeze(-1).cpu().numpy())
    print("Augmented training y (including y*):", train_Y_aug.cpu().numpy())

    p_std = summarize_prob_less_than(res_std.model, res_std.likelihood, Xc_eval, y_star_t)
    p_con = summarize_prob_less_than(res_con.model, res_con.likelihood, Xc_eval, y_star_t)

    mean_std_Xc, var_std_Xc = marginal_mean_variance(res_std.model, res_std.likelihood, Xc_eval, observation_noise=False)
    mean_jes_Xc, var_jes_Xc = jes_truncated_predictive_moments(
        res_std.model,
        res_std.likelihood,
        Xc_eval,
        y_star=y_star,
        observation_noise=False,
    )
    p_jes_gaussian_moment_match = summarize_prob_less_than_from_moments(mean_jes_Xc, var_jes_Xc, y_star_t)

    print("\nP(f(Xc) < y*) summaries:")
    print(f"  Standard sparse GP (Gaussian posterior): mean={p_std[0]:.3f}, min={p_std[1]:.3f}, max={p_std[2]:.3f}")
    print("  JES truncated latent posterior: exactly 1.000 at every x by construction")
    print(
        "  JES Gaussian moment-match of truncated latent posterior: "
        f"mean={p_jes_gaussian_moment_match[0]:.3f}, "
        f"min={p_jes_gaussian_moment_match[1]:.3f}, "
        f"max={p_jes_gaussian_moment_match[2]:.3f}"
    )
    print(f"  Modified sparse GP (Gaussian posterior): mean={p_con[0]:.3f}, min={p_con[1]:.3f}, max={p_con[2]:.3f}")

    format_diff_stats("Standard sparse", mean_std, var_std, "JES truncation", mean_jes, var_jes)
    format_diff_stats("Standard sparse", mean_std, var_std, "Modified sparse", mean_con, var_con)
    format_diff_stats("JES truncation", mean_jes, var_jes, "Modified sparse", mean_con, var_con)

    print("\nNoise levels:")
    print("  Base GP fixed noise:", float(base_gp.likelihood.noise.detach().cpu().item()))
    print("  Standard sparse GP fixed noise:", res_std.likelihood.noise.item())
    print("  Modified sparse GP fixed noise:", res_con.likelihood.noise.item())
    print(
        "\nFor y(x) with observation noise, the JES-style approximation uses the "
        "same truncated latent mean and adds the observation-noise variance to the "
        "truncated latent variance."
    )
    print(
        f"  Mean of latent truncated variance on grid: {var_jes.mean().item():.6f}\n"
        f"  Mean of noisy JES-style variance on grid: {var_jes_y.mean().item():.6f}"
    )

    # ------------------------------------------------------------------
    # 6) Figures: predictive moments across x.
    # ------------------------------------------------------------------
    std_init = var_init.sqrt()
    std_std = var_std.sqrt()
    std_jes = var_jes.sqrt()
    std_con = var_con.sqrt()

    x_np = x_grid.squeeze(-1).cpu().numpy()
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

    fig, axes = plt.subplots(1, 4, figsize=(24, 5.4), sharex=True, sharey=True)

    plot_mean_and_band(
        axes[0],
        x_grid=x_grid,
        mean=mean_init,
        std=std_init,
        f_true=f_true,
        x_obs=x_train,
        y_obs=y_train,
        inducing_points=fixed_inducing,
        title="Conditioned exact GP vs sparse GP after init\n(same moments at inducing points)",
        x_star=x_star,
        y_star=y_star,
        x_pseudo=x_star_t,
        y_pseudo=y_star_t_col.reshape(-1),
    )
    axes[0].plot(x_np, mean_init.reshape(-1).cpu().numpy(), label="Init sparse mean")
    axes[0].plot(
        x_np,
        mean_base_cond.reshape(-1).cpu().numpy(),
        linestyle="--",
        label="Conditioned exact GP mean",
    )
    axes[0].fill_between(
        x_np,
        (mean_base_cond - PLOT_STD_MULT * var_base_cond.sqrt()).reshape(-1).cpu().numpy(),
        (mean_base_cond + PLOT_STD_MULT * var_base_cond.sqrt()).reshape(-1).cpu().numpy(),
        alpha=0.12,
        label="Conditioned exact GP band",
    )
    axes[0].legend(fontsize=6, loc="lower left")

    plot_mean_and_band(
        axes[1],
        x_grid=x_grid,
        mean=mean_std,
        std=std_std,
        f_true=f_true,
        x_obs=x_train,
        y_obs=y_train,
        inducing_points=res_std.inducing_points,
        title="Standard sparse GP\nafter conditioning on (x*, y*)",
        x_star=x_star,
        y_star=y_star,
        x_pseudo=x_star_t,
        y_pseudo=y_star_t_col.reshape(-1),
    )

    plot_mean_and_band(
        axes[2],
        x_grid=x_grid,
        mean=mean_jes,
        std=std_jes,
        f_true=f_true,
        x_obs=x_train,
        y_obs=y_train,
        inducing_points=res_std.inducing_points,
        title="JES-style truncation of the latent predictive Gaussian",
        x_star=x_star,
        y_star=y_star,
        x_pseudo=x_star_t,
        y_pseudo=y_star_t_col.reshape(-1),
        band_label="Truncated moments (±1 std)",
    )

    plot_mean_and_band(
        axes[3],
        x_grid=x_grid,
        mean=mean_con,
        std=std_con,
        f_true=f_true,
        x_obs=x_train,
        y_obs=y_train,
        inducing_points=res_con.inducing_points,
        title="Modified sparse GP\n(Standard ELBO + step term)",
        x_star=x_star,
        y_star=y_star,
        x_pseudo=x_star_t,
        y_pseudo=y_star_t_col.reshape(-1),
    )

    for ax in axes:
        ax.set_xlim(float(x_np.min()), float(x_np.max()))
        ax.set_ylim(y_lim_low, y_lim_high)

    fig.suptitle(
        "1D test: conditioning on (x*, y*) first, then comparing JES truncation with the modified VFE sparse GP",
        fontsize=15,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.95])

    # ------------------------------------------------------------------
    # 7) Figure: pointwise predictive densities at the location where the
    #    truncated predictive mean differs most from the modified sparse GP.
    # ------------------------------------------------------------------
    idx_ref = torch.argmax((mean_jes - mean_con).abs()).item()
    x_ref = x_grid[idx_ref : idx_ref + 1]

    mean_std_ref, var_std_ref = marginal_mean_variance(res_std.model, res_std.likelihood, x_ref, observation_noise=False)
    mean_con_ref, var_con_ref = marginal_mean_variance(res_con.model, res_con.likelihood, x_ref, observation_noise=False)
    mean_jes_ref, var_jes_ref = jes_truncated_predictive_moments(
        res_std.model,
        res_std.likelihood,
        x_ref,
        y_star=y_star,
        observation_noise=False,
    )

    y_low = min(
        float((mean_std_ref - 4.0 * var_std_ref.sqrt()).item()),
        float((mean_con_ref - 4.0 * var_con_ref.sqrt()).item()),
        float((mean_jes_ref - 4.0 * var_jes_ref.sqrt()).item()),
        float(y_star),
    )
    y_high = max(
        float((mean_std_ref + 4.0 * var_std_ref.sqrt()).item()),
        float((mean_con_ref + 4.0 * var_con_ref.sqrt()).item()),
        float((mean_jes_ref + 4.0 * var_jes_ref.sqrt()).item()),
        float(y_star),
    )
    y_pad = 0.08 * max(1e-6, y_high - y_low)
    y_values = torch.linspace(y_low - y_pad, y_high + y_pad, 800, dtype=x_grid.dtype, device=x_grid.device)

    pdf_std = pointwise_normal_pdf_grid(mean_std_ref.reshape(()), var_std_ref.reshape(()), y_values)
    pdf_trunc = pointwise_upper_truncated_pdf_grid(mean_std_ref.reshape(()), var_std_ref.reshape(()), y_star_t, y_values)
    pdf_mod = pointwise_normal_pdf_grid(mean_con_ref.reshape(()), var_con_ref.reshape(()), y_values)

    fig2, ax2 = plt.subplots(1, 1, figsize=(7.2, 4.8))
    ax2.plot(y_values.cpu().numpy(), pdf_std.cpu().numpy(), label="Standard sparse predictive Gaussian")
    ax2.plot(y_values.cpu().numpy(), pdf_trunc.cpu().numpy(), label="Exact upper-truncated Gaussian (JES on f)")
    ax2.plot(y_values.cpu().numpy(), pdf_mod.cpu().numpy(), label="Modified sparse GP Gaussian")
    ax2.axvline(float(y_star), color="lightgreen", linestyle="--", label="y*")
    ax2.set_title(
        "Pointwise predictive density at x_ref = "
        f"{float(x_ref.item()):.3f}\n(max |mean_JES - mean_modified| on the grid)"
    )
    ax2.set_xlabel("Function value")
    ax2.set_ylabel("Density")
    ax2.legend(fontsize=8)
    fig2.tight_layout()

    print("\nReference point for the density comparison:")
    print(f"  x_ref = {float(x_ref.item()):.6f}")
    print(f"  Standard sparse moments     : mean={float(mean_std_ref.item()):.6f}, var={float(var_std_ref.item()):.6f}")
    print(f"  JES truncated latent moments: mean={float(mean_jes_ref.item()):.6f}, var={float(var_jes_ref.item()):.6f}")
    print(f"  Modified sparse moments     : mean={float(mean_con_ref.item()):.6f}, var={float(var_con_ref.item()):.6f}")

    plt.show()


if __name__ == "__main__":
    main()
