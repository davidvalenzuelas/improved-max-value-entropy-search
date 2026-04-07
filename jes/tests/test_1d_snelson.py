#!/usr/bin/env python3
# coding: utf-8
"""
Snelson 1D test for VFE sparse GP

It compares standard ELBO vs standard ELBO with added step constraint term 
using as many inducing points as data points and placing them on the observed
data. The noise is learned in the standard model and fixed in the constrained
one.

It is not the best test for the constrained model, due to the large number of
training points we're using here.

Authors: Daniel Hernández-Lobato, David Valenzuela Sánchez
"""

import io
import numpy as np
import requests
import torch
import matplotlib.pyplot as plt

from modified_vfe_sparse_gp import (
    fit_vfe_sparse_gp,
    predictive_distribution,
    normal_cdf,
    VFESparseGP,
    build_init_dist_from_base_gp,
)


def load_snelson():
    """This function downloads the Snelson 1D dataset"""
    # Downloads dataset from URL and loads it into numpy arrays
    url = "http://arantxa.ii.uam.es/~dhernan/MLAS2023/EdSnelson.npy"
    r = requests.get(url, timeout=30)
    r.raise_for_status()
    data = np.load(io.BytesIO(r.content), allow_pickle=False)
    
    # Converts to double precision torch tensors
    x_train = torch.from_numpy(data[0]).double()
    y_train = torch.from_numpy(data[1]).double()
    
    # Ensures x_train is 2D
    if x_train.ndim == 1:
        x_train = x_train.unsqueeze(-1)
        
    return x_train, y_train


@torch.no_grad()
def prob_f_below_y_star(model, Xc, y_star): 
    """ This function calculates the mean/min/max P(f(Xc) < y*) under the
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
    """This function builds a sparse GP model with the same mean and covariance modules
    as the base GP, and with a variational distribution initialized from the posterior
    of the base GP at the inducing points"""
    
    # Builds variational distribution at inducing points from base GP posterior
    init_dist = build_init_dist_from_base_gp(base_gp, inducing_points)
    
    # Builds sparse GP model
    model = VFESparseGP(inducing_points=inducing_points, init_dist=init_dist,
        mean_module=base_gp.mean_module, covar_module=base_gp.covar_module)
    model = model.to(dtype=inducing_points.dtype, device=inducing_points.device)

    # Fixed inducing points here
    model.variational_strategy.inducing_points.requires_grad_(False)
    return model


@torch.no_grad()
def compare_predictive_distributions(model_a, likelihood_a, model_b, likelihood_b, x_train, name_a, name_b):
    """Compares the predictive distributions of two models by computing the mean and variance"""
    # Grid for comparison
    test_x = torch.linspace(x_train.min().item(), x_train.max().item(), 400,
        dtype=x_train.dtype,device=x_train.device).unsqueeze(-1)
    
    # Obtains predictive distributions without observation noise
    pred_a = predictive_distribution(model_a, likelihood_a, test_x, observation_noise=False)
    pred_b = predictive_distribution(model_b, likelihood_b, test_x, observation_noise=False)
    
    # Computes mean and variance differences
    mean_diff = (pred_a.mean - pred_b.mean).abs()
    var_diff = (pred_a.variance - pred_b.variance).abs()
    
    print(f"\nComparing {name_a} vs {name_b} just after initialization:")
    print(f"Mean abs diff (avg): {mean_diff.mean().item():.6e}")
    print(f"Mean abs diff (max): {mean_diff.max().item():.6e}")
    print(f"Var abs diff (avg): {var_diff.mean().item():.6e}")
    print(f"Var abs diff (max): {var_diff.max().item():.6e}")
    

@torch.no_grad()
def plot_two_predictive_distributions(model_a, likelihood_a, model_b, likelihood_b,
    x_train, y_train, inducing_points, title, label_a="Base GP", label_b="Sparse init",
    y_star=None):
    
    test_x = torch.linspace(x_train.min().item(), x_train.max().item(), 400,
        dtype=x_train.dtype, device=x_train.device).unsqueeze(-1)
    
    # Obtains predictive distributions without observation noise
    pred_a = predictive_distribution(model_a, likelihood_a, test_x, observation_noise=False)
    pred_b = predictive_distribution(model_b, likelihood_b, test_x, observation_noise=False)
    
    # Computes mean and std for both models
    mean_a = pred_a.mean.cpu()
    std_a = pred_a.variance.sqrt().cpu()
    mean_b = pred_b.mean.cpu()
    std_b = pred_b.variance.sqrt().cpu()
    
    # Plots training data, inducing points, and predictive means with confidence bands
    plt.figure(figsize=(9, 4))
    plt.plot(x_train.squeeze(-1).cpu().numpy(), y_train.cpu().numpy(), "k*", markersize=4, label="Training data")
    
    Z = inducing_points.detach().cpu()
    plt.plot(Z.squeeze(-1).numpy(), np.zeros(Z.shape[0]), "rx", markersize=5, mew=1.5, label="Inducing points")
    
    plt.plot(test_x.squeeze(-1).cpu().numpy(), mean_a.numpy(), label=label_a)
    plt.fill_between(
        test_x.squeeze(-1).cpu().numpy(),
        (mean_a - 2 * std_a).numpy(),
        (mean_a + 2 * std_a).numpy(),
        alpha=0.20,
    )
    
    plt.plot(test_x.squeeze(-1).cpu().numpy(), mean_b.numpy(), "--", label=label_b)
    plt.fill_between(
        test_x.squeeze(-1).cpu().numpy(),
        (mean_b - 2 * std_b).numpy(),
        (mean_b + 2 * std_b).numpy(),
        alpha=0.20,
    )
    
    # Plots y* if provided
    if y_star is not None:
        plt.axhline(float(y_star), color="lightgreen", linestyle="--", label="y*")
        
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_mean_and_band(model, likelihood, x_train, y_train, 
    inducing_points, title, y_star=None):
    """ This function plots the predictive mean and 2*std band of the model,
    as well as the training data and inducing points. If y* is provided, it
    also plots a horizontal line at it."""
    # Test grid for plotting
    test_x = torch.linspace(
        x_train.min().item(),
        x_train.max().item(),
        400
    ).double().unsqueeze(-1)
    
    # Predicts latent distribution at test points without noice
    with torch.no_grad():
        pred = predictive_distribution(model, likelihood, test_x, observation_noise=False)
        mean = pred.mean
        std = pred.variance.sqrt()
        
    # Plots training data
    plt.figure(figsize=(9, 4))
    plt.plot(x_train.squeeze(-1).numpy(), y_train.numpy(), "k*",
            markersize=4, label="Training data")
    
    # Plots inducing points
    Z = inducing_points.detach().cpu()
    plt.plot(Z.squeeze(-1).numpy(), np.zeros(Z.shape[0]), "rx",
            markersize=6, mew=2, label="Inducing points")
    
    # Plot predictive mean and 2*std confidence band
    plt.plot(test_x.squeeze(-1).numpy(), mean.numpy(), label="Mean")
    plt.fill_between(
        test_x.squeeze(-1).numpy(),
        (mean - 2 * std).numpy(),
        (mean + 2 * std).numpy(),
        alpha=0.3,
        label="Confidence"
    )
    
    # Plots y* if provided
    if y_star is not None:
        plt.axhline(float(y_star), color="lightgreen",
                    linestyle="--", label="y*")
    
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.show()


def obtain_mean_difference(res_std, res_con, x_train):
    """This method obtains the mean difference between constrained
    and standard models """
    # Grid for comparison
    test_x = torch.linspace(
        x_train.min().item(),
        x_train.max().item(),
        400
    ).double().unsqueeze(-1)
    
    # Predicts mean at test points for both models
    with torch.no_grad():
        mean_std = predictive_distribution(res_std.model, res_std.likelihood, test_x).mean
        mean_con = predictive_distribution(res_con.model, res_con.likelihood, test_x).mean
        
    # Computes mean difference
    diff = mean_con - mean_std
    # Prints mean and max absolute difference
    print("Mean absolute difference:", diff.abs().mean().item())
    print("Max absolute difference:", diff.abs().max().item())


def main():
    # Global reproducibility seed
    torch.manual_seed(0)
    # Loads Snelson dataset
    x_train, y_train = load_snelson()
    
    # Obtains number of data points and sets number of inducing points to it
    N = x_train.shape[0]
    M = N
    
    # Sets inducing points to training data locations
    fixed_inducing = x_train.contiguous()
    # Some parameters for the constrained model
    y_star = 0.8
    num_constraint_points = 100
    
    # Samples constraint points uniformly in the input space
    d = x_train.shape[1]
    torch.manual_seed(123)
    
    # If we want to sample uniformly in the box [0,1]^d
    # Xc = torch.rand(num_constraint_points, d, dtype=x_train.dtype, device=x_train.device)
    # If we want to sample uniformly in the box defined by the training data
    x_min, x_max = x_train.min(), x_train.max()
    Xc_eval = x_min + (x_max - x_min) * torch.rand(num_constraint_points, d, dtype=x_train.dtype,
        device=x_train.device)
    # If we want to check the constraint only at the inducing points, we can set Xc_eval to them
    # Xc_eval = fixed_inducing
    
    # Fits standard VFE sparse GP model
    init_noise = 1e-2
    res_std = fit_vfe_sparse_gp(train_X=x_train, train_Y=y_train, noise=init_noise,
        train_noise=True, M=M, y_star=None, fixed_inducing_points=fixed_inducing,
        seed_for_init=2024)
    
    # Fixes noise for constrained model
    noise_star = float(res_std.likelihood.noise.detach().cpu().item())
    
    # Creates base GP model and obtains its posterior to use it as initialization for
    # the constrained model
    base_gp = res_std.model
    base_gp.likelihood = res_std.likelihood
    
    # Builds sparse GP model just after initialization from the base GP posterior at
    # the inducing points
    init_model = build_sparse_model_just_initialized(base_gp, res_std.inducing_points)
    
    plot_two_predictive_distributions(
        base_gp,
        res_std.likelihood,
        init_model,
        res_std.likelihood,
        x_train,
        y_train,
        res_std.inducing_points,
        title="Base GP vs sparse GP just after initialization",
        label_a="Base GP",
        label_b="Sparse GP after init",
        y_star=y_star,
    )
    
    compare_predictive_distributions(
        base_gp,
        res_std.likelihood,
        init_model,
        res_std.likelihood,
        x_train,
        "Base GP",
        "Sparse GP after init",
    )
    
    # Fits constrained VFE sparse GP model
    res_con = fit_vfe_sparse_gp(train_X=x_train, train_Y=y_train, noise=noise_star,
        train_noise=False, M=M, y_star=y_star, fixed_inducing_points=fixed_inducing,
        seed_for_init=2024, base_gp=base_gp)
    
    # Plots
    plot_mean_and_band(res_std.model, res_std.likelihood, x_train, y_train,
        res_std.inducing_points, title="Standard ELBO", y_star=y_star)
    
    plot_mean_and_band(res_con.model, res_con.likelihood, x_train, y_train,
        res_con.inducing_points, title="Standard ELBO + Step constraint term",
        y_star=y_star)
    
    # Obtains mean difference between models
    print("\nPrinting some results...")
    obtain_mean_difference(res_std, res_con, x_train)
    
    # Computes P(f(Xc) < y*) under both models
    y_star_t = torch.tensor(y_star, dtype=x_train.dtype, device=x_train.device)
    p_std = prob_f_below_y_star(res_std.model, Xc_eval, y_star_t)
    p_con = prob_f_below_y_star(res_con.model, Xc_eval, y_star_t)
    # Prints results
    print("\nP(f(Xc) < y*) under q(f):")
    print(f"  Standard   : mean={p_std[0]:.3f}, min={p_std[1]:.3f}, max={p_std[2]:.3f}")
    print(f"  Constraint : mean={p_con[0]:.3f}, min={p_con[1]:.3f}, max={p_con[2]:.3f}")
    print("\nNoise:")
    print("  Standard learned:", noise_star)
    print("  Constraint fixed :", res_con.likelihood.noise.item())


if __name__ == "__main__":
    main()