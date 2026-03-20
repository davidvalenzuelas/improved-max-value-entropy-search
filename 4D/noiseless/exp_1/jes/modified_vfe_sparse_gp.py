#!/usr/bin/env python3
# coding: utf-8
"""
This file implements a VFE sparse GP with an additional probabilistic
step constraint term in the ELBO, which encourages the model to satisfy
a soft inequality constraint over some constraint points.

Authors: Daniel Hernández-Lobato, David Valenzuela Sánchez
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Literal, Optional

import torch
import gpytorch
import math

from gpytorch.models import ApproximateGP
from gpytorch.variational import CholeskyVariationalDistribution
from gpytorch.variational import VariationalStrategy
from gpytorch.mlls import VariationalELBO
from gpytorch.constraints.constraints import GreaterThan


def sample_Xc(num_constraint_points: int, d: int,
    method: Literal["rand", "sobol"] = "rand",
    device: Optional[torch.device] = None,
    dtype: Optional[torch.dtype] = None,
    lower_bound: Optional[torch.Tensor] = None,
    upper_bound: Optional[torch.Tensor] = None) -> torch.Tensor:
    """ This method samples points uniformly (using rand) or quasiuniformly
    (using sobol) in the input space, either from the unit box [0,1]^d or from
    a box defined by lower_bound and upper_bound."""
        
    device = device or torch.device("cpu")
    dtype = dtype or torch.float64
    
    # Firstly, samples are made in the box [0,1]^d
    if method == "rand":
        X = torch.rand(num_constraint_points, d, device=device, dtype=dtype)
    elif method == "sobol":
        engine = torch.quasirandom.SobolEngine(dimension=d, scramble=True)
        X = engine.draw(num_constraint_points).to(device=device, dtype=dtype)
    else:
        raise ValueError(f"Unknown sampling method: {method}")
    
    # If no bounds are provided, keeps the samples in [0,1]^d
    if lower_bound is None and upper_bound is None:
        return X
    
    if lower_bound is None or upper_bound is None:
        raise ValueError("Both lower_bound and upper_bound must be provided")
    
    lower_bound = lower_bound.to(device=device, dtype=dtype)
    upper_bound = upper_bound.to(device=device, dtype=dtype)
    
    # Rescales the samples to the box defined by both bounds
    return lower_bound + (upper_bound - lower_bound) * X


@torch.no_grad()
def build_init_dist_from_base_gp(base_gp, inducing_points: torch.Tensor,
    jitter: float = 1e-6) -> gpytorch.distributions.MultivariateNormal:
    """ This function builds the initial variational distribution q(u)
    using the latent posterior of a base GP evaluated at the inducing
    points. """
    
    # Puts the base GP and the likelihood in evaluation mode
    base_gp.eval()
    if hasattr(base_gp, "likelihood") and base_gp.likelihood is not None:
        base_gp.likelihood.eval()
    
    # Matches inducing points to the device and dtype of the base GP
    param0 = next(base_gp.parameters())
    Z_base = inducing_points.to(device=param0.device, dtype=param0.dtype)
    
    # Evaluates the latent GP posterior p(f(Z) | D) at the inducing points
    post_u = base_gp(Z_base)
    
    # Extracts posterior mean
    mean_u = post_u.mean
    if mean_u.ndim > 1 and mean_u.shape[-1] == 1:
        mean_u = mean_u.squeeze(-1)
    mean_u = mean_u.to(device=inducing_points.device, dtype=inducing_points.dtype)
    
    # Extracts posterior covariance matrix
    cov_u = post_u.covariance_matrix
    cov_u = cov_u.to(device=inducing_points.device, dtype=inducing_points.dtype)
    # We want to ensure that the covariance matrix is symmetric and positive definite,
    # so we add a small jitter to the diagonal
    cov_u = 0.5 * (cov_u + cov_u.transpose(-1, -2))
    # Identity matrix
    eye = torch.eye(cov_u.size(-1), dtype=cov_u.dtype, device=cov_u.device)
    cov_u = cov_u + jitter * eye
    
    # Returns the gaussian used to initialize q(u)
    return gpytorch.distributions.MultivariateNormal(mean_u, cov_u)


class VFESparseGP(ApproximateGP):
    """ This class defines our approximate GP model using the VFE sparse
    GP approach. It uses inducing points to approximate the full GP and
    it is designed to be trained with the modified ELBO, which includes
    a step constraint term."""
    def __init__(self, inducing_points: torch.Tensor,
        init_dist: Optional[gpytorch.distributions.MultivariateNormal] = None):
        # Zero mean and RBF kernel for covariances
        mean_module = gpytorch.means.ZeroMean()
        covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())
        
        # Variational approximate distribution q
        # We use a Cholesky factorization to represent its parameters
        variational_distribution = CholeskyVariationalDistribution(
            inducing_points.size(0), mean_init_std=0.0)
        
        # If an initial distribution isn't provided
        if init_dist is None:
            init_covar = covar_module(inducing_points).evaluate()
            # Ensures the covariance matrix is symmetric
            init_covar = 0.5 * (init_covar + init_covar.transpose(-1, -2))
            init_covar = init_covar * 1e-5
            # Small jitter added to the diagonal so that the matrix is positive definite
            init_covar = init_covar + 1e-8 * torch.eye(
                inducing_points.size(0),
                dtype=inducing_points.dtype,
                device=inducing_points.device,
            )
            # Builds the initial distribution for q with zero mean and the kernel covariance
            # at the inducing points
            init_dist = gpytorch.distributions.MultivariateNormal(
                torch.zeros(
                    inducing_points.size(0),
                    dtype=inducing_points.dtype,
                    device=inducing_points.device,
                ),
                init_covar,
            )
        variational_distribution.initialize_variational_distribution(init_dist)
        
        # Variational strategy, defining how the inducing points are used to
        # approximate the full GP
        variational_strategy = VariationalStrategy(self, inducing_points, variational_distribution,
            learn_inducing_locations=True, # this makes the inducing points trainable
        )
        
        # Avoids internal random reinitialization of the variational parameters, because we have already
        # initialized them with the prior like distribution above
        variational_strategy.variational_params_initialized = torch.tensor(
            True, device=inducing_points.device, dtype=torch.bool
        )
        
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
        
        # Returns a gaussian distribution used to compute the prior
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


def train_model_ADAM(model: torch.nn.Module, mll: gpytorch.mlls.MarginalLogLikelihood,
    train_x: torch.Tensor, train_y: torch.Tensor, training_iter: int = 400,
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


def normal_cdf(z: torch.Tensor) -> torch.Tensor:
    """ This function computes the CDF of the standard normal distribution at z, using
    the error function."""
    # phi(z) = 0.5 * (1 + erf(z/sqrt(2)))
    return 0.5 * (1.0 + torch.erf(z / math.sqrt(2.0)))

class StepConstraintVariationalELBO(VariationalELBO):
    """This class implements the Variational ELBO with an added step constraint
    term, which encourages the VFE sparse GP to satisfy a constraint P(f(Xc) < y*)
    over some constraint points Xc."""
    def __init__(self, likelihood: gpytorch.likelihoods.Likelihood,
        model: ApproximateGP, num_data: int, Xc: Optional[torch.Tensor],
        y_star: torch.Tensor, epsilon: float = 0.05,
        num_constraint_points: Optional[int] = None,
        d: Optional[int] = None,
        constraint_sampling: Literal["rand", "sobol"] = "rand",
        # Optional bounds for sampling Xc
        lower_bound: Optional[torch.Tensor] = None,
        upper_bound: Optional[torch.Tensor] = None):
        
        # Initializes the parent VariationalELBO class
        super().__init__(likelihood, model, num_data=num_data)
        
        # Validates epsilon
        if not (0.0 < float(epsilon) < 1.0):
            raise ValueError("Epsilon must be in (0,1).")
        
        # Stores the parameters for the step constraint term
        self.y_star = y_star
        self.epsilon = float(epsilon)
        self.num_constraint_points = num_constraint_points
        self.d = d
        # If Xc is provided, we will use it as fixed constraint points
        self.Xc = Xc
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound
        
        # Extra info for optional sampling of Xc
        self.constraint_sampling = constraint_sampling
        
        if self.Xc is None and (self.num_constraint_points is None or self.d is None):
            raise ValueError(
                "If Xc is None, num_constraint_points and d must be provided."
            )
            
    def _get_Xc(self) -> torch.Tensor:
        """This method returns the constraint points Xc"""
        # If Xc is fixed, always use it
        if self.Xc is not None:
            return self.Xc
        
        # If Xc is not fixed, samples them
        return sample_Xc(
            num_constraint_points=self.num_constraint_points,
            d=self.d,
            method=self.constraint_sampling,
            device=self.y_star.device,
            dtype=self.y_star.dtype,
            lower_bound=self.lower_bound,
            upper_bound=self.upper_bound,
        )
        
    def _step_term(self) -> torch.Tensor:
        """This method calculates the summed soft penalty term for the step
        constraint"""
        # Computes the variational posterior at the constraint points, which is a
        # Gaussian distribution
        Xc_eval = self._get_Xc()
        qf = self.model(Xc_eval)
        
        # Extracts posterior mean and variance, avoiding issues with zero variance
        m = qf.mean
        v = qf.variance.clamp_min(1e-12)
        # Standard deviation
        s = v.sqrt()
        # Standardize distance to y*
        z = (self.y_star - m) / s
        
        # Computes probabilities under gaussian posterior
        p_less = normal_cdf(z) # P(f(Xc) < y*)
        p_greater = 1.0 - p_less # P(f(Xc) > y*)
        # Log probabilities for the step constraint term
        log_eps = torch.log(torch.as_tensor(self.epsilon, device=m.device, dtype=m.dtype))
        log_1m = torch.log(torch.as_tensor(1.0 - self.epsilon, device=m.device, dtype=m.dtype))
        
        # Computes the step constraint term
        term = log_eps * p_greater + log_1m * p_less
        # Sums over constraint points to get the added step constraint term
        return term.sum()
    
    def _log_likelihood_term(self, variational_dist_f, target, **kwargs):
        """ Overrides the standard log likelihood term in the ELBO to add
        the step constraint term."""
        # Standard expected log likelihood term from VariationalELBO
        base = super()._log_likelihood_term(variational_dist_f, target, **kwargs)
        # Additional step constraint term
        step = self._step_term()
        
        # Combines both contributions
        return base + step
    
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
    noise: float, train_noise: bool, M: int, verbose: bool = True, 
    epsilon: float = 1e-6, training_iter: int = 200, lr: float = 0.01,
    # We allow to train vfe sparse gp with the modified ELBO (contains
    # the step constraint term) if y* is provided, otherwise we train
    # it with the standard ELBO.
    y_star: Optional[float | torch.Tensor] = None,
    num_constraint_points: int = 100,
    constraint_sampling: Literal["rand", "sobol"] = "rand",
    Xc: Optional[torch.Tensor] = None,
    # For testing
    fixed_inducing_points: Optional[torch.Tensor] = None,
    seed_for_init: Optional[int] = None,
    inducing_seed: Optional[int] = None,
    # Allows to initialize q from the posterior of a provided base GP
    base_gp = None,
    # Allows to resample Xc at each evaluation of the ELBO
    resample_Xc_each_eval: bool = False) -> FitResult:
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
        noise_constraint=GreaterThan(1e-6)
    )
    likelihood = likelihood.to(dtype=train_X.dtype, device=train_X.device)
    
    # Sets tiny noise level
    likelihood.noise = torch.as_tensor(noise, dtype=train_X.dtype, device=train_X.device)
    likelihood.raw_noise.requires_grad_(train_noise)
    
    # Optional reproducible init
    if seed_for_init is not None:
        torch.manual_seed(seed_for_init)
        
    # Number of training points
    N = train_X.shape[0]
    # The number of inducing points cannot exceed the number of training points
    M_eff = min(int(M), int(N))
    
    # Selects inducing points
    if fixed_inducing_points is None:
        # Random (optionally deterministic) subsampling
        if inducing_seed is not None:
            # Uses a separate generator for inducing point selection
            g = torch.Generator(device=train_X.device)
            g.manual_seed(int(inducing_seed))
            # Selects the points using the generator for reproducibility
            perm = torch.randperm(N, generator=g, device=train_X.device)
        else:
            # No seed provided, uses standard random permutation
            perm = torch.randperm(N, device=train_X.device)
        # Selects the first M_eff points from the permuted indices as inducing points
        inducing_points = train_X[perm[:M_eff]].contiguous()
    else:
        # Uses the provided fixed inducing points
        inducing_points = fixed_inducing_points.to(
            device=train_X.device, dtype=train_X.dtype
        ).contiguous()
        inducing_points = inducing_points[:M_eff].contiguous()
    
    # Builds the initial distribution for q from the posterior of a base GP if this
    # GP is provided
    init_dist = None
    if base_gp is not None:
        init_dist = build_init_dist_from_base_gp(
            base_gp=base_gp, inducing_points=inducing_points,
        )
    
    # Instantiates the VFE sparse GP model with the selected inducing points
    model = VFESparseGP(inducing_points=inducing_points, init_dist=init_dist)
    model = model.to(dtype=train_X.dtype, device=train_X.device)
    
    # If inducing points are fixed, we don't want them to be updated during training
    if fixed_inducing_points is not None:
        model.variational_strategy.inducing_points.requires_grad_(False)
    
    # If no constraint is provided, trains with standar ELBO
    # If y* is provided, trains with standard ELBO + step constraint term
    if y_star is None:
        mll = VariationalELBO(likelihood, model, num_data=N)
    else:
        # Gets the dimension of the input space from training data
        d = train_X.shape[-1]
        
        # Determines the bounds for sampling Xc
        x_lower = train_X.min(dim=0).values
        x_upper = train_X.max(dim=0).values
        
        if Xc is not None:
            # If Xc is provided, we will use them as fixed constraint points
            Xc = Xc.to(device=train_X.device, dtype=train_X.dtype)
        elif not resample_Xc_each_eval:
            # If Xc is not provided and we aren't resampling them at each evaluation,
            # we sample them once and keep them fixed during training
            Xc = sample_Xc(
                num_constraint_points, d, method=constraint_sampling,
                device=train_X.device, dtype=train_X.dtype,
                lower_bound=x_lower, upper_bound=x_upper
            )
            
        # Converts y* to a tensor if it is a scalar
        y_star_t = torch.as_tensor(y_star, device=train_X.device, dtype=train_X.dtype)
        
        # Uses the standard ELBO with the added step constraint term
        mll = StepConstraintVariationalELBO(
            likelihood=likelihood,
            model=model,
            num_data=N,
            Xc=Xc,
            y_star=y_star_t,
            epsilon=epsilon,
            num_constraint_points=num_constraint_points,
            d=d,
            constraint_sampling=constraint_sampling,
            lower_bound=x_lower,
            upper_bound=x_upper,
        )
        
    # Trains model by minimizing the -ELBO with ADAM optimizer  
    losses = train_model_ADAM(model=model, mll=mll, train_x=train_X, train_y=y_vec,
        training_iter=training_iter, likelihood=likelihood, lr=lr,
        verbose=verbose)
    
    # Returns the results after fitting the vfe sparse GP
    return FitResult(model=model, likelihood=likelihood, mll=mll, losses=losses,
        inducing_points=inducing_points, Xc=Xc if y_star is not None else None)