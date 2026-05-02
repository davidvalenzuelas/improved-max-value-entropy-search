#!/usr/bin/env python3
# coding: utf-8

from __future__ import annotations

import torch
import gpytorch

from torch import Tensor
from botorch.acquisition.acquisition import AcquisitionFunction
from botorch.utils.transforms import t_batch_mode_transform

from modified_vfe_sparse_gp import fit_vfe_sparse_gp


class MyAcquisition(AcquisitionFunction):
    
    def __init__(self, model, x_star: float | Tensor, y_star: float | Tensor,
        M: int, training_iter: int = 1000, lr: float = 0.5e-3,
        lower_bound: Tensor | None = None, upper_bound: Tensor | None = None,
        verbose: bool = True) -> None:
        # Initializes base class
        super().__init__(model=model)
        
        self.initial_model = model
        # Sets the gp and its likelihood to evaluation mode
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
        
        self.x_star = self._format_x_star(x_star, d=d, dtype=dtype, device=device)
        self.y_star = torch.as_tensor(y_star, dtype=dtype, device=device).reshape(())
        
        noise = self.initial_model.likelihood.noise.detach().mean()
        
        # We build the model conditioned on the sampled optimum (x*,y*) here
        _ = self.initial_model.posterior(self.x_star, observation_noise=False)
        conditioned_base_gp = self.initial_model.condition_on_observations(
            X=self.x_star,
            Y=self.y_star.view(1, 1),
        )
        conditioned_base_gp.eval()
        conditioned_base_gp.likelihood.eval()

        # Sparse approximation to the model conditioned on (x*, y*) and constrained by y*.
        train_X_aug = torch.cat([train_X, self.x_star], dim=0).contiguous()
        train_Y_aug = torch.cat(
            [train_Y.reshape(-1), self.y_star.view(1)], dim=0
        ).contiguous()

        fit = fit_vfe_sparse_gp(
            train_X=train_X_aug,
            train_Y=train_Y_aug,
            noise=noise,
            train_noise=False,
            M=M,
            y_star=self.y_star,
            x_star=self.x_star,
            training_iter=training_iter,
            lr=lr,
            lower_bound=lower_bound,
            upper_bound=upper_bound,
            fixed_inducing_points=train_X_aug,
            base_gp=conditioned_base_gp,
            verbose=verbose,
        )

        self.conditional_model = fit.model
        self.conditional_likelihood = fit.likelihood
        self.conditional_model.eval()
        self.conditional_likelihood.eval()

    @t_batch_mode_transform()
    def forward(self, X: Tensor) -> Tensor:
        """Evaluate log Var[y(x)|D] - log Var[y(x)|D, x*, y*]."""
        if X.shape[-2] != 1:
            raise ValueError("Only q=1 is supported.")

        batch_shape = X.shape[:-2]
        X_eval = X.reshape(-1, X.shape[-1])

        initial_var = self.initial_model.posterior(
            X_eval,
            observation_noise=True,
        ).variance.reshape(-1).clamp_min(1e-12)

        with gpytorch.settings.fast_pred_var():
            conditional_post = self.conditional_likelihood(
                self.conditional_model(X_eval)
            )

        conditional_var = conditional_post.variance.reshape(-1).clamp_min(1e-12)

        return (torch.log(initial_var) - torch.log(conditional_var)).reshape(batch_shape)

    @staticmethod
    def _format_x_star(
        x_star: float | Tensor,
        d: int,
        dtype: torch.dtype,
        device: torch.device,
    ) -> Tensor:
        x_star = torch.as_tensor(x_star, dtype=dtype, device=device)

        if x_star.ndim == 0:
            x_star = x_star.view(1, 1)
        elif x_star.ndim == 1:
            x_star = x_star.view(1, -1)

        if x_star.shape != (1, d):
            raise ValueError(f"x_star must have shape (1, {d}).")

        return x_star.detach()
