#!/usr/bin/env python3
# coding: utf-8
"""MES acquisition for VFE sparse GP."""
from __future__ import annotations

import torch
import gpytorch

from torch import Tensor
from botorch.acquisition.acquisition import AcquisitionFunction
from botorch.utils.transforms import t_batch_mode_transform

from modified_vfe_sparse_gp import fit_vfe_sparse_gp


class MyAcquisition(AcquisitionFunction):
    def __init__(self, model, x_star: float | Tensor, y_star: float | Tensor,
        M: int, lower_bound: Tensor | None = None,
        upper_bound: Tensor | None = None) -> None:
        super().__init__(model=model)

        self.initial_model = model
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

        noise = self.initial_model.likelihood.noise.detach().mean()

        train_X_cond = train_X
        train_Y_cond = train_Y.reshape(-1)
        base_gp_for_sparse_init = self.initial_model
        fixed_inducing_points = train_X.contiguous()

        fit = fit_vfe_sparse_gp(train_X=train_X_cond, train_Y=train_Y_cond, noise=noise,
            train_noise=False, M=M, y_star=self.y_star, x_star=self.x_star,
            lower_bound=lower_bound, upper_bound=upper_bound, fixed_inducing_points=fixed_inducing_points,
            base_gp=base_gp_for_sparse_init, verbose=False)

        self.conditional_model = fit.model
        self.conditional_likelihood = fit.likelihood
        self.conditional_model.eval()
        self.conditional_likelihood.eval()

    @t_batch_mode_transform()
    def forward(self, X: Tensor) -> Tensor:
        if X.shape[-2] != 1:
            raise ValueError("Only q=1 is supported")

        batch_shape = X.shape[:-2]
        X_eval = X.reshape(-1, X.shape[-1])

        initial_var = self.initial_model.posterior(X_eval, observation_noise=True
        ).variance.reshape(-1).clamp_min(1e-12)

        with gpytorch.settings.fast_pred_var():
            conditional_post = self.conditional_likelihood(self.conditional_model(X_eval))
        conditional_var = conditional_post.variance.reshape(-1).clamp_min(1e-12)

        acq = 0.5 * (torch.log(initial_var) - torch.log(conditional_var))
        return acq.reshape(batch_shape)

    @staticmethod
    def _format_x_star(x_star: float | Tensor, d: int, dtype: torch.dtype,
        device: torch.device,) -> Tensor:
        x_star = torch.as_tensor(x_star, dtype=dtype, device=device)

        if x_star.ndim == 0:
            x_star = x_star.view(1, 1)
        elif x_star.ndim == 1:
            x_star = x_star.view(1, -1)

        if x_star.shape != (1, d):
            raise ValueError(f"x_star must have shape (1, {d}).")

        return x_star.detach()
