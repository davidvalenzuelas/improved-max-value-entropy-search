#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

r"""
Acquisition function for generalized renyi entropy search (RES).

.. [Hvarfner2022joint]
    C. Hvarfner, F. Hutter, L. Nardi,
    Joint Entropy Search for Maximally-informed Bayesian Optimization.
    In Proceedings of the Annual Conference on Neural Information
    Processing Systems (NeurIPS), 2022.

.. [Tu2022joint]
    B. Tu, A. Gandy, N. Kantas, B. Shafei,
    Joint Entropy Search for Multi-objective Bayesian Optimization.
    In Proceedings of the Annual Conference on Neural Information
    Processing Systems (NeurIPS), 2022.
"""

from __future__ import annotations

import warnings
from math import log, pi

import os
from typing import Optional

import json
import dill
import numpy as np

import torch
from botorch import settings
from botorch.acquisition.acquisition import AcquisitionFunction, MCSamplerMixin
from botorch.acquisition.objective import PosteriorTransform

from botorch.models.fully_bayesian import SaasFullyBayesianSingleTaskGP
try:
    from botorch.models.gp_regression import MIN_INFERRED_NOISE_LEVEL
except ImportError:
    # Temporary fix to avoid error when importing
    MIN_INFERRED_NOISE_LEVEL = 1e-6
from botorch.models.model import Model

from botorch.models.utils import check_no_nans, fantasize as fantasize_flag
from botorch.sampling.normal import SobolQMCNormalSampler
from botorch.utils.transforms import concatenate_pending_points, t_batch_mode_transform
from torch import Tensor

from renyi_entropy_search import qRenyiEntropySearch
from botorch.optim import optimize_acqf

from torch.distributions import Normal

#from botorch.util.util import torch_pdf, torch_cdf

# def torch_pdf(x, mean=torch.tensor(0.0), var=torch.tensor(1.0)):
#     pdf = (1 / torch.sqrt(2 * torch.pi * var)) * torch.exp(-((x - mean) ** 2) / (2 * var))
#     return pdf

# def torch_cdf(X, loc=torch.tensor(0.0), scale=torch.tensor(1.0)):
#     normal = torch.distributions.normal.Normal(
#             loc=torch.zeros(1, device=X.device, dtype=X.dtype),
#             scale=torch.ones(1, device=X.device, dtype=X.dtype),
#         )
#     return normal.cdf(X)

MCMC_DIM = -3  # Only relevant if you do Fully Bayesian GPs.

# The CDF query cannot be strictly zero in the division
# and this clamping helps assure that it is always positive.
CLAMP_LB = torch.finfo(torch.float32).eps
FULLY_BAYESIAN_ERROR_MSG = (
    "HRES is not yet available with Fully Bayesian GPs. Track the issue, "
    "which regards conditioning on a number of optima on a collection "
    "of models, in detail at https://github.com/pytorch/botorch/issues/1680"
)


class qRenyiEntropySearchHedge():
    r"""The acquisition function for the Hedge Renyi Entropy Search, where the batches
    `q > 1` are supported through the lower bound formulation.

    """

    def __init__(
        self,
        model: Model,
        optimal_inputs: Tensor,
        optimal_outputs: Tensor,
        condition_noiseless: bool = True,
        posterior_transform: Optional[PosteriorTransform] = None,
        X_pending: Optional[Tensor] = None,
        # estimation_type: str = "LB",
        maximize: bool = True,
        num_samples: int = 64,
        alphas: list = [ 0.001, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.999 ],
        filename_rewards: list = None,
        filename_candidates: list = None,
        filename_selected_alphas: list = None,
        observation_noise: bool = True,
        eps: float = 1e-6,
        num_iters: int = 100,
    ) -> None:
        r"""Hedge portfolio Renyi entropy search acquisition function.

        Args:
            model: A fitted single-outcome model.
            X* optimal_inputs: A `num_samples x d`-dim tensor containing the sampled
                optimal inputs of dimension `d`. We assume for simplicity that each
                sample only contains one optimal set of inputs.
            y* optimal_outputs: A `num_samples x 1`-dim Tensor containing the optimal
                set of objectives of dimension `1`.
            condition_noiseless: Whether to condition on noiseless optimal observations
                `f*` [Hvarfner2022joint]_ or noisy optimal observations `y*`
                [Tu2022joint]_. These are sampled identically, so this only controls
                the fashion in which the GP is reshaped as a result of conditioning
                on the optimum.
            estimation_type: estimation_type: A string to determine which entropy
                estimate is computed: Lower bound" ("LB") or "Monte Carlo" ("MC").
                Lower Bound is recommended due to the relatively high variance
                of the MC estimator.
            maximize: If true, we consider a maximization problem.
            X_pending: A `m x d`-dim Tensor of `m` design points that have been
                submitted for function evaluation, but have not yet been evaluated.
            num_samples: The number of Monte Carlo samples used for the Monte Carlo
                estimate.
            alpha: Hyper-parameter of the acquisition function that generalizes Shannon
            entropy to Renyi entropy. Limit of alpha=1 gives Shannon entropy.
        """

        self.optimal_inputs = optimal_inputs
        self.optimal_outputs = optimal_outputs
        self.condition_noiseless = condition_noiseless
        self.maximize = maximize
        self.posterior_transform = posterior_transform
        self.X_pending = X_pending
        self.num_samples = num_samples
        self.observation_noise = observation_noise

        self.filename_rewards = filename_rewards
        self.filename_candidates = filename_candidates
        self.filename_selected_alphas = filename_selected_alphas
        
        self.eps = eps
        self.model = model

        self.eta = torch.sqrt( 8 * torch.log(torch.tensor(len(alphas))) / torch.tensor(num_iters) )

        self.alphas = torch.FloatTensor(alphas)
       
        if filename_rewards is None or not os.path.exists(filename_rewards):
            self.v_rewards = (self.alphas * 0.0).reshape((1, len(self.alphas)))
            self.store_v_rewards(self.filename_rewards)
            self.selected_alphas = np.array([], dtype=np.float64).reshape(0, 1)
        else:
            self.load_v_rewards(filename_rewards)
            self.load_d_candidates(filename_candidates)
            self.update_v_rewards(model)
            self.load_selected_alphas(self.filename_selected_alphas)

        self.current_alpha = self.select_alpha(self.alphas, self.v_rewards[ self.v_rewards.shape[ 0 ] - 1, : ]).item()

        print("Hedge porfolio Renyi entropy")

    def get_key(self, alpha):
        return f"{alpha:.3f}"

    def load_v_rewards(self, filename):
        self.v_rewards = torch.from_numpy(np.loadtxt(filename, ndmin = 2))

    def store_v_rewards(self, filename):
        np.savetxt(filename, self.v_rewards.numpy())        
        exp_rewards = torch.exp(self.eta * self.v_rewards)
        probs = exp_rewards / torch.sum(exp_rewards, 1, keepdims = True)
        np.savetxt("probs_" + filename, probs.numpy())

    def load_d_candidates(self, filename):
        with open(filename, "rb") as fr:
            self.d_candidates = dill.load(fr)

    def store_d_candidates(self, filename):
        with open(filename, "wb") as fw:
            dill.dump(self.d_candidates, fw)

    def load_selected_alphas(self, filename):
        with open(filename, "r") as fr:
            self.selected_alphas = np.atleast_2d(np.loadtxt(fr)).T

    def update_selected_alphas(self, filename):
        with open(filename, "w") as fw:
            np.savetxt(fw, self.selected_alphas)

    def select_alpha(self, alphas, v_rewards):
            
        exp_rewards = torch.exp(self.eta * (v_rewards - torch.max(v_rewards)))
        v_probs = exp_rewards / torch.sum(exp_rewards)

        choice_index = torch.multinomial(v_probs, 1).item()

        alpha_chose = alphas[ choice_index ]
        
        self.selected_alphas = np.concatenate((self.selected_alphas, np.array([[alpha_chose]])))
        self.update_selected_alphas(self.filename_selected_alphas)

        return alpha_chose


    def update_v_rewards(self, model_updated):
        
        self.v_rewards = torch.concatenate((self.v_rewards, torch.zeros((1, self.v_rewards.shape[ 1 ]))), 0)
        self.v_rewards[ self.v_rewards.shape[ 0 ] - 1, : ] += self.v_rewards[ self.v_rewards.shape[ 0 ] - 2, : ] 

        for i, alpha in enumerate(self.alphas):
            self.v_rewards[ self.v_rewards.shape[ 0 ] - 1, i ] += model_updated(self.d_candidates[ self.get_key(alpha) ]).mean.item()
        
        self.store_v_rewards(self.filename_rewards)

    def optimize_acqf(self, bounds, num_restarts_opt, raw_samples_opt_acq, q: int = 1):
        
        d_acq_values = {}
        self.d_candidates = {}

        acq = qRenyiEntropySearch(model=self.model, optimal_inputs=self.optimal_inputs.double(), \
                    optimal_outputs=self.optimal_outputs.double(), alpha=0.5, \
                    condition_noiseless=self.condition_noiseless, posterior_transform=self.posterior_transform, \
                    X_pending=self.X_pending, maximize=self.maximize, num_samples=self.num_samples, \
                    observation_noise=self.observation_noise, eps=self.eps)

        for alpha in self.alphas:

            acq.alpha = alpha
 
            candidate, acq_value = optimize_acqf(acq_function=acq, bounds=bounds, q=q, num_restarts=num_restarts_opt, raw_samples=raw_samples_opt_acq)
            self.d_candidates[ self.get_key(alpha) ] = candidate
            d_acq_values[ self.get_key(alpha) ] = acq_value

        self.store_d_candidates(self.filename_candidates)

        return self.d_candidates[ self.get_key(self.current_alpha) ], d_acq_values[ self.get_key(self.current_alpha) ]


