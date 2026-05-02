#!/usr/bin/env python3
# coding: utf-8
from __future__ import annotations
from typing import Literal

import torch
import gpytorch

from torch import Tensor
from botorch.acquisition.acquisition import AcquisitionFunction
from botorch.utils.transforms import t_batch_mode_transform

from modified_vfe_sparse_gp import fit_vfe_sparse_gp


class MyAcquisition(AcquisitionFunction):
    """
    Custom acquisition function adapted to the current VFE sparse GP code.

    It supports two modes:

        style="mes":
            MES-style approximation.
            The conditioned model approximates p(y | D, y*).

        style="jes":
            JES-style approximation.
            The conditioned model approximates p(y | D, x*, y*).

    In both cases, the acquisition is:

        log Var[y(x) | D] - log Var[y(x) | conditioned information]

    For now, only q=1 is supported.
    """
    
    def __init__(self, model, x_star: float | Tensor, y_star: float | Tensor,
        M: int, style: Literal["mes", "jes"] = "mes",lower_bound: Tensor | None = None,
        upper_bound: Tensor | None = None) -> None:
        # Inits base class
        super().__init__(model=model)
        
        self.style = style
        
        # Original GP model: p(f|D)
        self.initial_model = model
        # Sets the original model and its likelihood to evaluation mode
        self.initial_model.eval()
        self.initial_model.likelihood.eval()
        
        train_X = self.initial_model.train_inputs[0].detach()
        train_Y = self.initial_model.train_targets.detach()
        if train_X.ndim == 1:
            train_X = train_X.unsqueeze(-1)
        if train_Y.ndim == 2 and train_Y.shape[-1] == 1:
            train_Y = train_Y.squeeze(-1)
            
        dtype = train_X.dtype
        device = train_X.device
        d = train_X.shape[-1]
        
        self.x_star = self._format_x_star(x_star=x_star, d=d, dtype=dtype, device=device)
        self.y_star = torch.as_tensor(y_star, dtype=dtype, device=device).reshape(())
        
        # Same observation noise as the original model.
        noise = self.initial_model.likelihood.noise.detach().mean()
        
        if self.style == "mes":
            # Uses the original data.
            train_X_cond = train_X
            train_Y_cond = train_Y.reshape(-1)
            
            # Base GP is still the original GP.
            base_gp_for_sparse_init = self.initial_model
            
            # Fixed inducing points are the original observed inputs
            fixed_inducing_points = train_X.contiguous()
            
        else:
            # Inits internal caches
            with torch.no_grad():
                _ = self.initial_model.posterior(self.x_star, observation_noise=False)
            
            # Condition the exact GP on the sampled optimum pair (x*,y*)
            conditioned_base_gp = self.initial_model.condition_on_observations(
                X=self.x_star, Y=self.y_star.view(1, 1))
            conditioned_base_gp.eval()
            conditioned_base_gp.likelihood.eval()
            
            # Includes (x*,y*) as an observation for training
            train_X_cond = torch.cat([train_X, self.x_star],dim=0).contiguous()
            train_Y_cond = torch.cat([train_Y.reshape(-1), self.y_star.view(1)],dim=0).contiguous()
            
            # Base GP for sparse init is the conditioned exact GP
            base_gp_for_sparse_init = conditioned_base_gp
            
            # Fixed inducing points include the observed inputs and x*
            fixed_inducing_points = train_X_cond
            
        # Trains the modified VFE sparse GP
        # In MES, this approximate p(y|D,y*); in JES, p(y|D,x*,y*)
        fit = fit_vfe_sparse_gp(train_X=train_X_cond, train_Y=train_Y_cond, noise=noise,
            train_noise=False, M=M, y_star=self.y_star, x_star=self.x_star,
            lower_bound=lower_bound, upper_bound=upper_bound, fixed_inducing_points=fixed_inducing_points,
            base_gp=base_gp_for_sparse_init)
        
        # Stores the trained conditioned sparse model and its likelihood
        self.conditional_model = fit.model
        self.conditional_likelihood = fit.likelihood
        self.conditional_model.eval()
        self.conditional_likelihood.eval()
        
    @t_batch_mode_transform()
    def forward(self, X: Tensor) -> Tensor:
        
        # Only q=1
        if X.shape[-2] != 1:
            raise ValueError("Only q=1 is supported")

        # Save batch shape to return the acquisition with the right shape.
        batch_shape = X.shape[:-2]

        # Convert batch_shape x 1 x d into N x d for GP evaluation.
        X_eval = X.reshape(-1, X.shape[-1])

        # Predictive variance before conditioning:
        #
        #     Var[y(x) | D]
        #
        # observation_noise=True means we use the predictive variance of y,
        # not only the latent function f.
        initial_var = self.initial_model.posterior(
            X_eval,
            observation_noise=True,
        ).variance.reshape(-1).clamp_min(1e-12)

        # Predictive variance after conditioning.
        #
        # If style="mes", this is approximately:
        #     Var[y(x) | D, y*]
        #
        # If style="jes", this is approximately:
        #     Var[y(x) | D, x*, y*]
        with gpytorch.settings.fast_pred_var():
            conditional_post = self.conditional_likelihood(
                self.conditional_model(X_eval)
            )

        conditional_var = conditional_post.variance.reshape(-1).clamp_min(1e-12)

        # Acquisition:
        #
        #     log Var before - log Var after
        #
        # Larger values mean larger reduction in predictive uncertainty.
        return (
            torch.log(initial_var) - torch.log(conditional_var)
        ).reshape(batch_shape)
    
    
    @staticmethod
    def _format_x_star(x_star: float | Tensor, d: int, dtype: torch.dtype,
        device: torch.device,) -> Tensor:
        """ Converts x_star to a tensor with shape 1 x d"""
        
        x_star = torch.as_tensor(x_star, dtype=dtype, device=device)
        
        #Scalar x_star, typical in 1D.
        if x_star.ndim == 0:
            x_star = x_star.view(1, 1)
        # Vector x_star, for d dimensional inputs.
        elif x_star.ndim == 1:
            x_star = x_star.view(1, -1)
            
        # We only allow one sampled optimum input x*
        if x_star.shape != (1, d):
            raise ValueError(f"x_star must have shape (1, {d}).")
        
        # x_star is fixed, not optimized inside the acquisition function.
        return x_star.detach()