#!/usr/bin/env python3
# coding: utf-8
"""MES acquisition with Gaussian truncation."""
from __future__ import annotations

import torch

from torch import Tensor
from botorch.acquisition.acquisition import AcquisitionFunction
from botorch.utils.transforms import t_batch_mode_transform

from my_utils import normal_cdf, normal_pdf


class MyAcquisition(AcquisitionFunction):
    def __init__(self, model, x_star: float | Tensor, y_star: float | Tensor,
        M: int, lower_bound: Tensor | None = None,
        upper_bound: Tensor | None = None) -> None:
        super().__init__(model=model)

        self.initial_model = model
        self.initial_model.eval()
        self.initial_model.likelihood.eval()

        train_X = self.initial_model.train_inputs[0].detach()
        if train_X.ndim == 1:
            train_X = train_X.unsqueeze(-1)

        dtype = train_X.dtype
        device = train_X.device

        self.y_star = torch.as_tensor(y_star, dtype=dtype, device=device).reshape(())
        self.noise = self.initial_model.likelihood.noise.detach().mean().to(dtype=dtype, device=device)

    @t_batch_mode_transform()
    def forward(self, X: Tensor) -> Tensor:
        if X.shape[-2] != 1:
            raise ValueError("Only q=1 is supported")

        batch_shape = X.shape[:-2]
        X_eval = X.reshape(-1, X.shape[-1])

        initial_var = self.initial_model.posterior(X_eval, observation_noise=True
        ).variance.reshape(-1).clamp_min(1e-12)

        posterior = self.initial_model.posterior(X_eval, observation_noise=False)
        mean = posterior.mean.reshape(-1)
        variance = posterior.variance.reshape(-1).clamp_min(1e-12)
        std = variance.sqrt()

        beta = (self.y_star - mean) / std
        Phi = normal_cdf(beta).clamp_min(1e-12)
        phi = normal_pdf(beta)
        lam = phi / Phi
        trunc_var = variance * (1.0 - beta * lam - lam.pow(2)).clamp_min(1e-12)
        conditional_var = (trunc_var + self.noise).clamp_min(1e-12)

        acq = 0.5 * (torch.log(initial_var) - torch.log(conditional_var))
        return acq.reshape(batch_shape)
