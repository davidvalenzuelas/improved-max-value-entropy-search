#!/usr/bin/env python3
# coding: utf-8
"""
This file implements a VFE sparse GP with an additional probabilistic
step constraint term in the ELBO, which encourages th model to satisfy
a soft inequality constraint over some constraint points.

Authors: Daniel Hernández-Lobato, David Valenzuela Sánchez
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Literal, Optional

import torch
import gpytorch

from gpytorch.models import ApproximateGP
from gpytorch.variational import CholeskyVariationalDistribution
from gpytorch.variational import VariationalStrategy
from gpytorch.mlls import VariationalELBO
from gpytorch.constraints.constraints import GreaterThan


def sample_unit_box(num_constraint_points: int, d: int,
    method: Literal["rand", "sobol"] = "rand",
    device: Optional[torch.device] = None,
    dtype: Optional[torch.dtype] = None) -> torch.Tensor:
    """ This method samples points uniformly from the unit box [0,1]^d,
    using either standard random or Sobol quasi random sampling, depending
    on the specified method argument."""
    
    device = device or torch.device("cpu")
    dtype = dtype or torch.float64
    
    # Standard iid random sampling from uniform distribution in [0,1]^d
    if method == "rand":
        return torch.rand(num_constraint_points, d, device=device, dtype=dtype)
    
    # Quasi random Sobol sampling for better space filling coverage
    elif method == "sobol":
        engine = torch.quasirandom.SobolEngine(dimension=d, scramble=True)
        X = engine.draw(num_constraint_points).to(device=device, dtype=dtype)
        return X
    
    raise ValueError(f"Unknown sampling method: {method}")


class VFESparseGP(ApproximateGP):
    """ This class defines our approximate GP model using the VFE sparse
    GP approach. It uses inducing points to approximate the full GP and
    it is designed to be trained with the modified ELBO, which includes
    a step constraint term."""
    def __init__(self, inducing_points: torch.Tensor):
        # Zero mean and RBF kernel for covariances
        mean_module = gpytorch.means.ZeroMean()
        covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())
        
        # Variational approximate distribution q
        # We use a Cholesky factorization to represent its parameters
        variational_distribution = CholeskyVariationalDistribution(
            inducing_points.size(0), mean_init_std=0.0)
        
        # Smooth initialization of the variational distribution with a prior
        # like gaussian
        init_dist = gpytorch.distributions.MultivariateNormal(
            torch.zeros(
                inducing_points.size(0),
                dtype=inducing_points.dtype,
                device=inducing_points.device
            ),
            covar_module(inducing_points) * 1e-5)
        variational_distribution.initialize_variational_distribution(init_dist)
        
        # Variational strategy, defining how the inducing points ares used to
        # approximate the full GP
        variational_strategy = VariationalStrategy(self, inducing_points, variational_distribution,
            learn_inducing_locations=True, # this makes inducing points trainable
        )
        
        # Avoids internal random reinitialization of the variational parameters, because we have already
        # initialized them with the prior like distribution above
        variational_strategy.variational_params_initialized = torch.tensor(1, device=inducing_points.device, dtype=inducing_points.dtype)
        
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


def train_model_ADAM(model: torch.nn.Module, mll: gpytorch.mlls.MarginalLogLikelihood,
    train_x: torch.Tensor, train_y: torch.Tensor, training_iter: int = 200,
    likelihood: Optional[torch.nn.Module] = None, lr: float = 1e-2,
    verbose: bool = True) -> torch.Tensor:
    """ This function trains the VFE sparse GP model using the Adam optimizer, by 
    maximizing the ELBO."""
    
    # Sets the model and likelihood in training mode
    model.train()
    if likelihood is not None:
        likelihood.train()
        
    # Stores the loss values during training
    losses = torch.zeros(training_iter, dtype=train_x.dtype, device=train_x.device)
    
    # Determines which hyperparameters to optimize
    if likelihood is None:
        params = list(model.parameters())
    else:
        params = list(model.parameters()) + list(likelihood.parameters())
        
    # Defines ADAM optimizer
    optimizer = torch.optim.Adam(params, lr=lr)
    
    # Closure function to compute loss and gradients for ADAM
    def closure():
        optimizer.zero_grad()
        output = model(train_x)
        # We want to maximize ELBO, so we minimize -ELBO
        loss = -mll(output, train_y)
        loss.backward()
        return loss
    
    # Main training loop, we call the closure function here to compute loss
    # and gradients, and then we use the optimizer to update the parameters
    for i in range(training_iter):
        loss = closure()
        if verbose:
            print(f"Iter {i+1}/{training_iter} - Loss: {loss.item():.6f}")
        losses[i] = loss.detach()
        # Performs an optimization step
        optimizer.step()
        
    # Returns the loss values during training
    return losses


def _normal_cdf(z: torch.Tensor) -> torch.Tensor:
    """ This function computes the CDF of the standard normal distribution at z, using
    the error function."""
    # phi(z) = 0.5 * (1 + erf(z/sqrt(2)))
    return 0.5 * (1.0 + torch.erf(z / torch.sqrt(torch.tensor(2.0, device=z.device, dtype=z.dtype))))


class StepConstraintVariationalELBO(VariationalELBO):
    """This class implements the Variational ELBO with an added step constraint
    term, which encourages the VFE sparse GP to satisfy a constraint P(f(Xc) < y*)
    over some constraint points Xc."""
    def __init__(self, likelihood: gpytorch.likelihoods.Likelihood,
        model: ApproximateGP, num_data: int, Xc: torch.Tensor,
        y_star: torch.Tensor, epsilon: float = 0.05,
        constraint_weight: float = 1.0):
        
        # Initializes the parent VariationalELBO class
        super().__init__(likelihood, model, num_data=num_data)
        
        # Validates epsilon
        if not (0.0 < float(epsilon) < 1.0):
            raise ValueError("Epsilon must be in (0,1).")
        
        # Stores the parameters for the step constraint term
        self.Xc = Xc
        self.y_star = y_star
        # We need smooth step
        self.epsilon = float(epsilon)
        self.constraint_weight = float(constraint_weight)
        
    def _step_term(self) -> torch.Tensor:
        """This method calculates the average soft penalty term for the step
        constraint"""
        # Computes the variational posterior at the constraint points, which is a
        # Gaussian distribution
        qf = self.model(self.Xc)
        
        # Extracts posterior mean and variance, avoiding issues with zero variance
        m = qf.mean
        v = qf.variance.clamp_min(1e-12)
        # Standard deviation
        s = v.sqrt()
        # Standarize distance to y*
        z = (self.y_star - m) / s
        
        # Computes probabilities under gaussian posterior
        p_less = _normal_cdf(z) # P(f(Xc) < y*)
        p_greater = 1.0 - p_less # P(f(Xc) > y*)
        # Log probabilities for the step constraint term
        log_eps = torch.log(torch.as_tensor(self.epsilon, device=m.device, dtype=m.dtype))
        log_1m = torch.log(torch.as_tensor(1.0 - self.epsilon, device=m.device, dtype=m.dtype))
        
        # Computes the step constraint term
        term = log_eps * p_greater + log_1m * p_less
        # Mean over constraint points to get the total step constraint term
        return term.mean()
    
    def _log_likelihood_term(self, variational_dist_f, target, **kwargs):
        """ Overrides the standard log likelihood term in the ELBO to add
        the step constraint term."""
        # Standard expected log likelihood term from VariationalELBO
        base = super()._log_likelihood_term(variational_dist_f, target, **kwargs)
        # Additional (normalized) step constraint term
        step = self._step_term()
        
        # Combines both contributions
        return base + self.constraint_weight * step

    def forward(self, variational_dist_f, target, **kwargs):
        """ This function computes the ELBO = expected log likelihood - KL
        divergence, but with our modified log likelihood term that includes
        the step constraint"""
        return super().forward(variational_dist_f, target, **kwargs)


@torch.no_grad()
def predictive_distribution(model: VFESparseGP,
    likelihood: gpytorch.likelihoods.Likelihood, test_x: torch.Tensor,
    observation_noise: bool = False) -> gpytorch.distributions.MultivariateNormal:
    """ Returns the predictive distribution at test_x."""
    
    # Sets model and likelihood in eval mode
    model.eval()
    likelihood.eval()
    test_x = test_x.to(
        dtype=next(model.parameters()).dtype,
        device=next(model.parameters()).device
    )
    
    # Uses faster variance computations for predictive distribution
    with gpytorch.settings.fast_pred_var():
        # Computes latent predictive distribution q()
        latent = model(test_x)
        return likelihood(latent) if observation_noise else latent


@dataclass
class FitResult:
    """ This class stores the results of fitting the VFE sparse GP model"""
    model: VFESparseGP
    likelihood: gpytorch.likelihoods.Likelihood
    mll: gpytorch.mlls.MarginalLogLikelihood
    losses: torch.Tensor
    inducing_points: torch.Tensor
    Xc: Optional[torch.Tensor] = None


def fit_vfe_sparse_gp(train_X: torch.Tensor, train_Y: torch.Tensor,
    noise: float, train_noise: bool, M: int = 100, verbose: bool = True, 
    training_iter: int = 200, lr: float = 1e-2,
    # We allow to train vfe sparse gp with the modified ELBO (contains
    # the step constraint term) if y* is provided, otherwise we train
    # it with the standard ELBO.
    y_star: Optional[float | torch.Tensor] = None,
    epsilon: float = 0.05, constraint_weight: float = 1.0,
    num_constraint_points: int = 100,
    constraint_sampling: Literal["rand", "sobol"] = "rand",
    Xc: Optional[torch.Tensor] = None) -> FitResult:
    """ This function fits a VFE sparse GP to the given training data, using
    the Adam optimizer to maximize the ELBO. If y* is provided, it trains
    with the modified ELBO that includes the step constraint term """
    # Converts training data to double precision
    train_X = train_X.double()
    train_Y = train_Y.double()
    
    # Ensure inputs are 2D, (N, d). 
    # For 1D inputs, make them (N, 1).
    if train_X.ndim == 1:
        train_X = train_X.unsqueeze(-1)
    if train_Y.ndim == 2 and train_Y.shape[-1] == 1:
        y_vec = train_Y.squeeze(-1)
    else:
        y_vec = train_Y
        
    # Creates gaussian likelihood
    likelihood = gpytorch.likelihoods.GaussianLikelihood(
        noise_constraint=GreaterThan(1e-8)
    )
    likelihood = likelihood.to(dtype=train_X.dtype, device=train_X.device)
    
    # Sets tiny noise level
    likelihood.noise = torch.as_tensor(noise, dtype=train_X.dtype, device=train_X.device)
    likelihood.raw_noise.requires_grad_(train_noise)
        
    # Number of training points
    N = train_X.shape[0]
    # The number of inducing points cannot exceed the number of training points
    M_eff = min(int(M), int(N))
    # Subsamples inducing points from training data
    perm = torch.randperm(N, device=train_X.device)
    inducing_points = train_X[perm[:M_eff]].contiguous()
    
    # Instantiates the VFE sparse GP model with the selected inducing points
    model = VFESparseGP(inducing_points=inducing_points)
    model = model.to(dtype=train_X.dtype, device=train_X.device)
    
    # If no constraint is provided, trains with standar ELBO
    # If y* is provided, trains with the step constraint term ELBO
    if y_star is None:
        mll = VariationalELBO(likelihood, model, num_data=N)
    else:
        # Gets the dimension of the input space from training data
        d = train_X.shape[-1]
        
        # If constraint points are not previded, we sample them uniformly from the
        # unit box [0,1]^d
        if Xc is None:
            Xc = sample_unit_box(num_constraint_points, d, method=constraint_sampling,
                device=train_X.device, dtype=train_X.dtype)
        else:
            Xc = Xc.to(device=train_X.device, dtype=train_X.dtype)
        
        # Converts y* to a tensor if it is a scalar
        y_star_t = torch.as_tensor(y_star, device=train_X.device, dtype=train_X.dtype)
        
        # Uses the modified ELBO with the step constraint term
        mll = StepConstraintVariationalELBO(likelihood=likelihood, model=model,
            num_data=N, Xc=Xc, y_star=y_star_t, epsilon=epsilon,
            constraint_weight=constraint_weight,)
        
    # Trains model by minimizing the -ELBO with ADAM optimizer  
    losses = train_model_ADAM(model=model, mll=mll, train_x=train_X, train_y=y_vec,
        training_iter=training_iter, likelihood=likelihood, lr=lr,
        verbose=verbose)
    
    # Returns the results after fitting the vfe sparse GP
    return FitResult(model=model, likelihood=likelihood, mll=mll, losses=losses,
        inducing_points=inducing_points, Xc=Xc if y_star is not None else None)