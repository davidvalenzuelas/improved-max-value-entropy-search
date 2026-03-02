#!/usr/bin/env python3
# coding: utf-8
"""
Snelson 1D test for VFE sparse GP

It compares standard ELBO vs standard ELBO + step constraint term using
fixed inducing points and fixed constraint points for a fair comparison.

Authors: Daniel Hernández-Lobato, David Valenzuela Sánchez
"""

import io
import numpy as np
import requests
import torch
import matplotlib.pyplot as plt

from modified_vfe_sparse_gp import fit_vfe_sparse_gp, predictive_distribution, normal_cdf


def load_snelson():
    """This function loads the Snelson 1D dataset."""
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
    # Evaluates posterior at constraint points
    qf = model(Xc)
    # Obtains meand and std
    m = qf.mean
    s = qf.variance.clamp_min(1e-12).sqrt()
    # Standardizes and computes probabilities under gaussian posterior
    z = (y_star - m) / s
    p_less = normal_cdf(z)
    return p_less.mean().item(), p_less.min().item(), p_less.max().item()


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
        pred = predictive_distribution(
            model, likelihood, test_x, observation_noise=False)
        mean = pred.mean
        std = pred.variance.sqrt()
    
    # Plots training data
    plt.figure(figsize=(9, 4))
    plt.plot(x_train.squeeze(-1).numpy(), y_train.numpy(),"k*",
            markersize=4, label="Training data")

    # Shows inducing points locations
    Z = inducing_points.detach().cpu()
    plt.plot(Z.squeeze(-1).numpy(), np.zeros(Z.shape[0]),
            "rx", markersize=6, mew=2, label="Inducing points")
    # Plots predictive mean and 2*std band
    plt.plot(test_x.squeeze(-1).numpy(), mean.numpy(), label="Mean")
    plt.fill_between(
        test_x.squeeze(-1).numpy(),
        (mean - 2 * std).numpy(),
        (mean + 2 * std).numpy(),
        alpha=0.3, label="Confidence"
    )
    
    if y_star is not None:
        plt.axhline(float(y_star), color="lightgreen", linestyle="--",label="y*")
        
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.show()


def obtain_mean_difference(res_std, res_con, x_train):
    """This method obtains the mean difference between constrained
    and standard models """
    # Test grid
    test_x = torch.linspace(
        x_train.min().item(),
        x_train.max().item(),
        400
    ).double().unsqueeze(-1)
    
    # Predicts mean at test points for both models
    with torch.no_grad():
        mean_std = predictive_distribution(
            res_std.model, res_std.likelihood, test_x
        ).mean
        mean_con = predictive_distribution(
            res_con.model, res_con.likelihood, test_x
        ).mean
    
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
    
    # Sets up hyperparameters
    M = 10
    training_iter = 400
    lr = 5e-3
    noise = 1e-2
    train_noise = True
    y_star = -1.0
    epsilon = 0.05
    constraint_weight = 1000.0
    num_constraint_points = 100
    
    # Fixes inducing points once for both models
    g = torch.Generator(device=x_train.device)
    g.manual_seed(999)
    perm = torch.randperm(x_train.shape[0], generator=g, device=x_train.device)
    fixed_inducing = x_train[perm[:M]].contiguous()
    
    # Fixes inducing points for both models
    d = x_train.shape[1]
    torch.manual_seed(123)
    Xc = torch.rand(num_constraint_points, d, dtype=x_train.dtype, device=x_train.device)
    
    # Fits standard vfe sparse GP model
    res_std = fit_vfe_sparse_gp(train_X=x_train, train_Y=y_train, noise=noise,
        train_noise=train_noise, M=M, training_iter=training_iter, lr=lr,
        y_star=None, fixed_inducing_points=fixed_inducing, seed_for_init=2024)
    
    # Fits constrained vfe sparse GP model
    res_con = fit_vfe_sparse_gp(train_X=x_train, train_Y=y_train, noise=noise,
        train_noise=train_noise, M=M, training_iter=training_iter, lr=lr,
        y_star=y_star, epsilon=epsilon, constraint_weight=constraint_weight,
        Xc=Xc, fixed_inducing_points=fixed_inducing, seed_for_init=2024)
    
    # plots
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
    p_std = prob_f_below_y_star(res_std.model, Xc, y_star_t)
    p_con = prob_f_below_y_star(res_con.model, Xc, y_star_t)
    
    # prints results
    print("\nP(f(Xc) < y*) under q(f):")
    print(f"  Standard   : mean={p_std[0]:.3f}")
    print(f"  Constraint : mean={p_con[0]:.3f}")
    print("\nLearned noise:")
    print("  Standard   :", res_std.likelihood.noise.item())
    print("  Constraint :", res_con.likelihood.noise.item())


if __name__ == "__main__":
    main()