#!/usr/bin/env python3
# coding: utf-8
"""
Snelson 1D test for comparing:
1) the standard sparse GP (before adding the constraint)
2) the old constrained setting
3) the new constrained setting with:
   - variational init from base_gp=res_std.model
   - dynamic resampling of Xc in the training domain

Authors: Daniel Hernández-Lobato, David Valenzuela Sánchez
"""

import io
import time
import numpy as np
import requests
import torch
import matplotlib.pyplot as plt

from modified_vfe_sparse_gp import fit_vfe_sparse_gp, predictive_distribution, normal_cdf


def load_snelson():
    url = "http://arantxa.ii.uam.es/~dhernan/MLAS2023/EdSnelson.npy"
    r = requests.get(url, timeout=30)
    r.raise_for_status()
    data = np.load(io.BytesIO(r.content), allow_pickle=False)

    x_train = torch.from_numpy(data[0]).double()
    y_train = torch.from_numpy(data[1]).double()

    if x_train.ndim == 1:
        x_train = x_train.unsqueeze(-1)

    return x_train, y_train


@torch.no_grad()
def prob_f_below_y_star(model, Xc, y_star):
    model.eval()
    qf = model(Xc)
    m = qf.mean
    s = qf.variance.clamp_min(1e-12).sqrt()
    z = (y_star - m) / s
    p_less = normal_cdf(z)
    return p_less.mean().item(), p_less.min().item(), p_less.max().item()


@torch.no_grad()
def evaluate_mean_difference(res_a, res_b, x_train):
    test_x = torch.linspace(
        x_train.min().item(),
        x_train.max().item(),
        400,
        dtype=x_train.dtype,
        device=x_train.device,
    ).unsqueeze(-1)

    mean_a = predictive_distribution(
        res_a.model, res_a.likelihood, test_x, observation_noise=False
    ).mean
    mean_b = predictive_distribution(
        res_b.model, res_b.likelihood, test_x, observation_noise=False
    ).mean

    diff = mean_b - mean_a
    return diff.abs().mean().item(), diff.abs().max().item()


def plot_model(ax, model, likelihood, x_train, y_train, inducing_points, title, y_star=None):
    test_x = torch.linspace(
        x_train.min().item(),
        x_train.max().item(),
        400,
        dtype=x_train.dtype,
        device=x_train.device,
    ).unsqueeze(-1)

    with torch.no_grad():
        pred = predictive_distribution(model, likelihood, test_x, observation_noise=False)
        mean = pred.mean.cpu()
        std = pred.variance.sqrt().cpu()

    ax.plot(
        x_train.squeeze(-1).cpu().numpy(),
        y_train.cpu().numpy(),
        "k*",
        markersize=4,
        label="Training data",
    )

    Z = inducing_points.detach().cpu()
    ax.plot(
        Z.squeeze(-1).numpy(),
        np.zeros(Z.shape[0]),
        "rx",
        markersize=5,
        mew=1.5,
        label="Inducing points",
    )

    ax.plot(test_x.squeeze(-1).cpu().numpy(), mean.numpy(), label="Mean")
    ax.fill_between(
        test_x.squeeze(-1).cpu().numpy(),
        (mean - 2 * std).numpy(),
        (mean + 2 * std).numpy(),
        alpha=0.25,
        label="Confidence",
    )

    if y_star is not None:
        ax.axhline(float(y_star), color="lightgreen", linestyle="--", label="y*")

    ax.set_title(title)


def main():
    torch.manual_seed(0)

    # Data
    x_train, y_train = load_snelson()
    N = x_train.shape[0]
    M = N
    fixed_inducing = x_train.contiguous()

    # Constraint configuration
    y_star = 0.3
    num_constraint_points = 100
    epsilon = 1e-6

    # Common fixed evaluation set in the training domain
    d = x_train.shape[1]
    x_lower = x_train.min(dim=0).values
    x_upper = x_train.max(dim=0).values

    torch.manual_seed(123)
    Xc_fixed = x_lower + (x_upper - x_lower) * torch.rand(
        num_constraint_points,
        d,
        dtype=x_train.dtype,
        device=x_train.device,
    )

    torch.manual_seed(999)
    Xc_eval = x_lower + (x_upper - x_lower) * torch.rand(
        num_constraint_points,
        d,
        dtype=x_train.dtype,
        device=x_train.device,
    )

    # Standard sparse GP: used to learn the noise and as base_gp for the new run
    print("Training standard sparse GP...")
    t0 = time.perf_counter()
    res_std = fit_vfe_sparse_gp(
        train_X=x_train,
        train_Y=y_train,
        noise=1e-2,
        train_noise=True,
        M=M,
        y_star=None,
        fixed_inducing_points=fixed_inducing,
        seed_for_init=2024,
        verbose=False,
    )
    t_std = time.perf_counter() - t0

    noise_star = float(res_std.likelihood.noise.detach().cpu().item())

    # OLD constrained behaviour:
    # - no base_gp
    # - fixed Xc
    # - no dynamic resampling
    print("Training OLD constrained model...")
    t0 = time.perf_counter()
    res_con_old = fit_vfe_sparse_gp(
        train_X=x_train,
        train_Y=y_train,
        noise=noise_star,
        train_noise=False,
        M=M,
        y_star=y_star,
        Xc=Xc_fixed,
        num_constraint_points=num_constraint_points,
        fixed_inducing_points=fixed_inducing,
        seed_for_init=2024,
        epsilon=epsilon,
        verbose=False,
        base_gp=None,
        resample_Xc_each_eval=False,
    )
    t_old = time.perf_counter() - t0

    # NEW constrained behaviour:
    # - base_gp from res_std.model
    # - Xc resampled at each evaluation in the training domain
    print("Training NEW constrained model...")
    t0 = time.perf_counter()
    res_con_new = fit_vfe_sparse_gp(
        train_X=x_train,
        train_Y=y_train,
        noise=noise_star,
        train_noise=False,
        M=M,
        y_star=y_star,
        Xc=None,
        num_constraint_points=num_constraint_points,
        fixed_inducing_points=fixed_inducing,
        seed_for_init=2024,
        epsilon=epsilon,
        verbose=False,
        base_gp=res_std.model,
        resample_Xc_each_eval=True,
    )
    t_new = time.perf_counter() - t0

    # Quantitative comparison on a common fixed evaluation set
    y_star_t = torch.tensor(y_star, dtype=x_train.dtype, device=x_train.device)
    p_std = prob_f_below_y_star(res_std.model, Xc_eval, y_star_t)
    p_old = prob_f_below_y_star(res_con_old.model, Xc_eval, y_star_t)
    p_new = prob_f_below_y_star(res_con_new.model, Xc_eval, y_star_t)
    mad_old_new, maxad_old_new = evaluate_mean_difference(res_con_old, res_con_new, x_train)
    mad_std_old, maxad_std_old = evaluate_mean_difference(res_std, res_con_old, x_train)
    mad_std_new, maxad_std_new = evaluate_mean_difference(res_std, res_con_new, x_train)

    print("\n=== Summary ===")
    print(f"Standard sparse GP time          : {t_std:.3f} s")
    print(f"Old constrained model time      : {t_old:.3f} s")
    print(f"New constrained model time      : {t_new:.3f} s")
    if t_old > 0:
        print(f"Speed ratio (old/new)           : {t_old / t_new:.3f}")

    print("\nFinal losses")
    print(f"  Standard sparse GP            : {res_std.losses[-1].item():.6f}")
    print(f"  Old constrained               : {res_con_old.losses[-1].item():.6f}")
    print(f"  New constrained               : {res_con_new.losses[-1].item():.6f}")

    print("\nP(f(Xc_eval) < y*) on common fixed evaluation set")
    print(f"  Standard sparse GP            : mean={p_std[0]:.3f}, min={p_std[1]:.3f}, max={p_std[2]:.3f}")
    print(f"  Old constrained               : mean={p_old[0]:.3f}, min={p_old[1]:.3f}, max={p_old[2]:.3f}")
    print(f"  New constrained               : mean={p_new[0]:.3f}, min={p_new[1]:.3f}, max={p_new[2]:.3f}")

    print("\nDifference between constrained means")
    print(f"  Old vs New mean abs diff      : {mad_old_new:.6f}")
    print(f"  Old vs New max abs diff       : {maxad_old_new:.6f}")

    print("\nDifference with respect to standard model")
    print(f"  Standard vs Old mean abs diff : {mad_std_old:.6f}")
    print(f"  Standard vs Old max abs diff  : {maxad_std_old:.6f}")
    print(f"  Standard vs New mean abs diff : {mad_std_new:.6f}")
    print(f"  Standard vs New max abs diff  : {maxad_std_new:.6f}")

    # Three plots: standard, old constrained, new constrained
    fig, axes = plt.subplots(1, 3, figsize=(20, 4.8), sharex=True, sharey=True)

    plot_model(
        axes[0],
        res_std.model,
        res_std.likelihood,
        x_train,
        y_train,
        res_std.inducing_points,
        title="Standard sparse GP",
        y_star=y_star,
    )

    plot_model(
        axes[1],
        res_con_old.model,
        res_con_old.likelihood,
        x_train,
        y_train,
        res_con_old.inducing_points,
        title="Constrained model - old behaviour",
        y_star=y_star,
    )

    plot_model(
        axes[2],
        res_con_new.model,
        res_con_new.likelihood,
        x_train,
        y_train,
        res_con_new.inducing_points,
        title="Constrained model - new behaviour",
        y_star=y_star,
    )

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=5)
    plt.tight_layout(rect=[0, 0, 1, 0.92])
    plt.show()


if __name__ == "__main__":
    main()
