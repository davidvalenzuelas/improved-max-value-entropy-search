#!/usr/bin/env python3
# coding: utf-8

# This file implements a VFE Sparse GP model using Gpytorch, as well as a training loop
# with Adam optimizer. The function 'fit_model_vfe_sparse' will be the main entry point
# for fitting the model.

import torch
import gpytorch

from gpytorch.models import ApproximateGP
from gpytorch.variational import CholeskyVariationalDistribution, UnwhitenedVariationalStrategy
from gpytorch.mlls import VariationalELBO


# Defines our approximate GP method based on the VFE approach
class VFESparseGP(ApproximateGP):
    
    def __init__(self, inducing_points: torch.Tensor):
        """
        inducing_points: (M, d)
        """
        # Zero mean and RBF kernel for covariances
        mean_module = gpytorch.means.ZeroMean()
        covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())
        
        # Variational approximate distribution q
        # We use a cholesky factorization to represent its parameters
        variational_distribution = CholeskyVariationalDistribution(
            inducing_points.size(0),
            mean_init_std=0.0,
        )
        
        # Smooth initialization of the variational distribution with this prior-like gaussian
        init_dist = gpytorch.distributions.MultivariateNormal(
            torch.zeros(inducing_points.size(0), dtype=inducing_points.dtype, device=inducing_points.device),
            covar_module(inducing_points) * 1e-5,
        )
        variational_distribution.initialize_variational_distribution(init_dist)
        
        # Variational strategy, defining how the inducing points ares used to approximate the full GP
        variational_strategy = UnwhitenedVariationalStrategy(
            self,
            inducing_points,
            variational_distribution,
            learn_inducing_locations=True, # this makes inducing points trainable
        )
        
        # Avoids internal random reinitialization of the variational parameters, because we have already
        # initialized them with the prior-like distribution above
        variational_strategy.variational_params_initialized = torch.tensor(1)
        
        # Initializes base ApproximateGP class
        super().__init__(variational_strategy)
        
        # Stores mean and covariances
        self.mean_module = mean_module
        self.covar_module = covar_module
        
    # Defines the GP prior for a given input x
    def forward(self, x: torch.Tensor):
        # Computes mean vector and covariance matrix for the input x
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        
        # Returns a gaussian distribution
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)
    
    
# Training function using ADAM optimizer
def train_model_ADAM(model: torch.nn.Module, mll: torch.nn.Module, train_x: torch.Tensor, train_y: torch.Tensor,
                    training_iter: int = 500, likelihood: torch.nn.Module | None = None, lr: float = 0.01,
                    verbose: bool = True,):
    """ Trains the variational GP model by maximizing the elbo using the ADAM optimizer"""
    model.train()
    if likelihood is not None:
        likelihood.train()

    # Parámetros a optimizar
    if likelihood is None:
        parameters = model.parameters()
    else:
        parameters = list(model.parameters()) + list(likelihood.parameters())

    optimizer = torch.optim.Adam(parameters, lr=lr)

    def closure():
        optimizer.zero_grad()
        output = model(train_x)
        loss = -mll(output, train_y)
        loss.backward()
        return loss

    losses = []
    for i in range(training_iter):
        loss = closure() * train_x.shape[0]
        optimizer.step()

        losses.append(loss.item())
        if verbose and ((i + 1) % 50 == 0 or i == 0):
            print(f"Iter {i+1}/{training_iter} - Loss: {loss.item():.6f}")

    model.eval()
    if likelihood is not None:
        likelihood.eval()

    return losses


# -------------------------
# (C) Fit wrapper: esto es lo que vas a llamar desde loop_BO.py (reemplaza fit_model)
# -------------------------
def fit_model_vfe_sparse(
    train_X: torch.Tensor,
    train_Y: torch.Tensor,
    state_dict: dict | None = None,
    M: int = 64,
    training_iter: int = 500,
    lr: float = 0.01,
    noise: float = 1e-4,
    verbose: bool = True,
):
    """
    Devuelve (model, likelihood) para que puedas:
      - usar el model en BO
      - guardar/cargar state_dict de ambos si quieres warm-start
    """
    train_X = train_X.double()
    train_Y = train_Y.double()

    # Likelihood gaussiana
    likelihood = gpytorch.likelihoods.GaussianLikelihood()
    likelihood.noise = torch.tensor(noise, dtype=train_X.dtype, device=train_X.device)

    # Inducing points: elige M puntos (subconjunto aleatorio de train_X)
    N = train_X.shape[0]
    M_eff = min(M, N)
    perm = torch.randperm(N, device=train_X.device)
    inducing_points = train_X[perm[:M_eff]].contiguous()

    model = VFESparseGP(inducing_points=inducing_points)

    # Warm-start (si quieres mantenerlo igual que tu loop actual)
    # Nota: aquí hay dos state_dict: el del model y el del likelihood
    if state_dict is not None:
        model.load_state_dict(state_dict["model"])
        likelihood.load_state_dict(state_dict["likelihood"])

    # ELBO variacional
    mll = VariationalELBO(likelihood, model, num_data=train_X.size(0))

    # Entrena con ADAM
    train_model_ADAM(
        model=model,
        mll=mll,
        train_x=train_X,
        train_y=train_Y.squeeze(-1) if train_Y.ndim == 2 and train_Y.shape[1] == 1 else train_Y,
        training_iter=training_iter,
        likelihood=None,   # OJO: VariationalELBO ya incluye likelihood
        lr=lr,
        verbose=verbose,
    )

    return model, likelihood


def pack_state_dict(model, likelihood) -> dict:
    return {"model": model.state_dict(), "likelihood": likelihood.state_dict()}