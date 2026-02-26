#!/usr/bin/env python3
# coding: utf-8

from __future__ import annotations
from dataclasses import dataclass
from typing import Literal, Optional

import torch
import gpytorch

from gpytorch.models import ApproximateGP
from gpytorch.variational import CholeskyVariationalDistribution
from gpytorch.variational import VariationalStrategy
from gpytorch.mlls import VariationalELBO


def sample_unit_box(
    n: int,
    d: int,
    method: Literal["rand", "sobol"] = "sobol",
    device: Optional[torch.device] = None,
    dtype: Optional[torch.dtype] = None,
) -> torch.Tensor:
    """
    Sample n points uniformly from [0,1]^d.
    - method="rand": torch.rand
    - method="sobol": SobolEngine (scrambled for quasi-random)
    """
    device = device or torch.device("cpu")
    dtype = dtype or torch.float64

    if method == "rand":
        return torch.rand(n, d, device=device, dtype=dtype)

    if method == "sobol":
        engine = torch.quasirandom.SobolEngine(dimension=d, scramble=True)
        X = engine.draw(n).to(device=device, dtype=dtype)
        return X

    raise ValueError(f"Unknown sampling method: {method}")


class VFESparseGP(ApproximateGP):
    """ This class defines our approximate GP model using the VFE sparse
    GP approach. It uses inducing points to approximate the full GP and
    is it designed to be trained with the modified ELBO, which includes
    the step constraint term."""
    
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
        params = model.parameters()
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
    
    # Gets number of training points
    N = train_x.shape[0]
    
    # Main training loop, we call the closure function here to compute loss
    # and gradients, and then we use the optimizer to update the parameters
    for i in range(training_iter):
        loss = closure() * N
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
    # TODO: add comments to this class and its methods
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
        #TODO: functionexplanation
        """  """
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
        # Sums over constraint points to get the total step constraint term
        return term.sum()
    
    
    @torch.no_grad()
    def _step_term_eval(self) -> torch.Tensor:
        """ Computes the step term in eval mode without gradients"""
        self.model.eval()
        return self._step_term()
    
    
    def _log_likelihood_term(self, variational_dist_f, target, **kwargs):
        """ Overrides the standard log likelihood term in the ELBO to add
        the step constraint term."""
        base = super()._log_likelihood_term(variational_dist_f, target, **kwargs)
        step = self._step_term()
        return base + self.constraint_weight * step
    
    
    def forward(self, variational_dist_f, target, **kwargs):
        """
        Use parent forward which combines:
        expected log likelihood term - KL term, with our overridden log-likelihood term.
        """
        return super().forward(variational_dist_f, target, **kwargs)


@torch.no_grad()
def predictive_distribution(
    model: VFESparseGP,
    likelihood: gpytorch.likelihoods._GaussianLikelihoodBase,
    test_x: torch.Tensor,
    observation_noise: bool = False,
) -> gpytorch.distributions.MultivariateNormal:
    """
    Returns predictive distribution at test_x.
    If observation_noise=True, returns p(y* | x*, D) (includes likelihood noise).
    Otherwise returns p(f* | x*, D).
    """
    model.eval()
    likelihood.eval()
    test_x = test_x.to(dtype=next(model.parameters()).dtype, device=next(model.parameters()).device)

    with gpytorch.settings.fast_pred_var():
        latent = model(test_x)
        return likelihood(latent) if observation_noise else latent


@dataclass
class FitResult:
    model: VFESparseGP
    likelihood: gpytorch.likelihoods._GaussianLikelihoodBase
    mll: gpytorch.mlls.MarginalLogLikelihood
    losses: torch.Tensor
    inducing_points: torch.Tensor
    Xc: Optional[torch.Tensor] = None


def fit_vfe_sparse_gp(
    train_X: torch.Tensor,
    train_Y: torch.Tensor,
    M: int = 25,
    training_iter: int = 500,
    lr: float = 1e-2,
    noise: float = 1e-6,
    fix_noise: bool = True,
    verbose: bool = True,
    # Optional step-constraint params:
    y_star: Optional[float | torch.Tensor] = None,
    epsilon: float = 0.05,
    constraint_weight: float = 1.0,
    num_constraint_points: int = 100,
    constraint_sampling: Literal["rand", "sobol"] = "sobol",
    Xc: Optional[torch.Tensor] = None,
) -> FitResult:
    """
    Fits a VFE Sparse GP. If y_star is provided, trains with the step-constraint term.

    Assumptions:
    - train_X is (N, d)
    - train_Y is (N,) or (N,1)
    - Constraint points live in [0,1]^d (unit box).
    """
    train_X = train_X.double()
    train_Y = train_Y.double()
    y_vec = train_Y.squeeze(-1) if train_Y.ndim == 2 and train_Y.shape[-1] == 1 else train_Y

    # Likelihood (Gaussian) in *noiseless* setting: use tiny fixed noise for numerical stability
    likelihood = gpytorch.likelihoods.GaussianLikelihood()
    likelihood.noise = torch.as_tensor(noise, dtype=train_X.dtype, device=train_X.device)
    if fix_noise:
        likelihood.raw_noise.requires_grad_(False)

    # Inducing points: random subset of training inputs (as in notebook)
    N = train_X.shape[0]
    M_eff = min(int(M), int(N))
    perm = torch.randperm(N, device=train_X.device)
    inducing_points = train_X[perm[:M_eff]].contiguous()

    model = VFESparseGP(inducing_points=inducing_points)

    if y_star is None:
        mll = VariationalELBO(likelihood, model, num_data=N)
    else:
        d = train_X.shape[-1]
        if Xc is None:
            Xc = sample_unit_box(
                num_constraint_points,
                d,
                method=constraint_sampling,
                device=train_X.device,
                dtype=train_X.dtype,
            )
        else:
            Xc = Xc.to(device=train_X.device, dtype=train_X.dtype)

        y_star_t = torch.as_tensor(y_star, device=train_X.device, dtype=train_X.dtype)

        mll = StepConstraintVariationalELBO(
            likelihood=likelihood,
            model=model,
            num_data=N,
            Xc=Xc,
            y_star=y_star_t,
            epsilon=epsilon,
            constraint_weight=constraint_weight,
        )

    losses = train_model_ADAM(
        model=model,
        mll=mll,
        train_x=train_X,
        train_y=y_vec,
        training_iter=training_iter,
        likelihood=likelihood,
        lr=lr,
        verbose=verbose,
    )

    return FitResult(
        model=model,
        likelihood=likelihood,
        mll=mll,
        losses=losses,
        inducing_points=inducing_points,
        Xc=Xc if y_star is not None else None,
    )