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
    
    # Sets the model and likelihood in training mode
    model.train()
    if likelihood is not None:
        likelihood.train()
        
    # Determines which parameters to optimize
    if likelihood is None:
        parameters = model.parameters()
    else:
        parameters = list(model.parameters()) + list(likelihood.parameters())
        
    # Defines ADAM optimizer
    optimizer = torch.optim.Adam(parameters, lr=lr)
    
    # Closure function to compute loss and gradients
    def closure():
        optimizer.zero_grad()
        output = model(train_x)
        loss = -mll(output, train_y) # we maximize ELBO, so we minimize -ELBO
        loss.backward()
        return loss
    
    losses = []
    # This is the main training loop, we call the closure function here to compute loss and gradients, and
    # then we use the optimizer to update the parameters
    for i in range(training_iter):
        # Scales by number of data points
        loss = closure() * train_x.shape[0]
        # The closure is called explicitly and not passed to optimizer.step(), because Adam does not require it
        # ,unlike LBFGS
        # Updates parameters
        optimizer.step()
        
        losses.append(loss.item())
        if verbose and ((i + 1) % 50 == 0 or i == 0):
            print(f"Iter {i+1}/{training_iter} - Loss: {loss.item():.6f}")
            
    # Sets the model and likelihood in evaluation mode after training        
    model.eval()
    if likelihood is not None:
        likelihood.eval()
        
    return losses

# Fits the VFE sparse GP model to the training data, and returns the trained model and likelihood
def fit_model_vfe_sparse(train_X: torch.Tensor, train_Y: torch.Tensor, state_dict: dict | None = None,
    M: int = 64, training_iter: int = 500, lr: float = 0.01,
    noise_eps: float = 1e-6,  # tiny fixed noise for numerical stability
    verbose: bool = True,):
    """Fits a variational sparse GP model with tiny observation noise """
    
    # Uses double precision for better numeical stability, important for Cholesk decompositions
    train_X = train_X.double()
    train_Y = train_Y.double()
    
    # Target vector should be 1D
    y_vec = train_Y.squeeze(-1) if train_Y.ndim == 2 and train_Y.shape[1] == 1 else train_Y
    
    #FIXME:
    # Noiseless likelihood
    # Fixes the noise to a small eps to approximate noiseless observations
    fixed_noise = torch.full_like(y_vec, noise_eps)
    likelihood = gpytorch.likelihoods.FixedNoiseGaussianLikelihood(noise=fixed_noise)
    
    # Selects inducing points
    N = train_X.shape[0]
    # Chooses up to M inducing points randomly from the training data
    M_eff = min(M, N)
    perm = torch.randperm(N, device=train_X.device)
    inducing_points = train_X[perm[:M_eff]].contiguous()
    
    # Instantiate variational sparse GP model
    model = VFESparseGP(inducing_points=inducing_points)
    
    # Loads state dict if provided, it allows resuming training from checkpoints
    if state_dict is not None:
        model.load_state_dict(state_dict["model"])
        try:
            likelihood.load_state_dict(state_dict["likelihood"])
        except Exception:
            pass  # allows loading checkpoints created with GaussianLikelihood
        
    # Defines the variational ELBO loss, our objective to maximize during training
    mll = VariationalELBO(likelihood, model, num_data=train_X.size(0))
    
    # Training optimizes model parameters using ADAM to minimize -ELBO
    # ELBO already includes likelihood
    train_model_ADAM(model=model, mll=mll, train_x=train_X, train_y=y_vec, training_iter=training_iter,
        likelihood=None, lr=lr, verbose=verbose,)
    
    return model, likelihood

# Packs model and likelihood state dicts into a single dictionary for checkpointing
def pack_state_dict(model, likelihood) -> dict:
    return {"model": model.state_dict(), "likelihood": likelihood.state_dict()}