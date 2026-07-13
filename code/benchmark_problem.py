#!/usr/bin/env python3
# coding: utf-8
"""
Benchmark function for MES+ experiments

Adapted from:
https://github.com/zi-w/Max-value-Entropy-Search/tree/master/test_functions

Shekel reference: https://www.sfu.ca/~ssurjano/shekel.html

Original matlab implementation: https://www.sfu.ca/~ssurjano/Code/shekelm.html

Copyright 2013. Derek Bingham, Simon Fraser University.
Authors: Sonja Surjanovic and Derek Bingham.
Original license: GNU General Public License, version 2.0.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch


# Shekel m=10 metadata
SHEKEL_M = 10
SHEKEL_LOWER = (0.0, 0.0, 0.0, 0.0)
SHEKEL_UPPER = (10.0, 10.0, 10.0, 10.0)
SHEKEL_X_MIN = (4.0, 4.0, 4.0, 4.0)
SHEKEL_F_MIN = -10.536283726219605


# Same constants as the matlab implementation.
SHEKEL_BETA = (
    0.1, 0.2, 0.2, 0.4, 0.4,
    0.6, 0.3, 0.7, 0.5, 0.5,
)

SHEKEL_C = (
    (4.0, 1.0, 8.0, 6.0, 3.0, 2.0, 5.0, 8.0, 6.0, 7.0),
    (4.0, 1.0, 8.0, 6.0, 7.0, 9.0, 3.0, 1.0, 2.0, 3.6),
    (4.0, 1.0, 8.0, 6.0, 3.0, 2.0, 5.0, 8.0, 6.0, 7.0),
    (4.0, 1.0, 8.0, 6.0, 7.0, 9.0, 3.0, 1.0, 2.0, 3.6),
)


@dataclass
class BenchmarkInfo:
    name: str
    num_dims: int
    lower: torch.Tensor
    upper: torch.Tensor
    x_opt_original: torch.Tensor
    x_opt_unit: torch.Tensor
    f_opt: float


class BenchmarkProblem:
    def __init__(self, name: str, dtype: torch.dtype = torch.float64):
        self.name = name.lower()
        self.dtype = dtype

        if self.name in {"shekel", "shekel4d", "shekel_m10"}:
            self.name = "shekel"
            self.info = self._shekel_info()
        else:
            raise ValueError(f"Unknown benchmark: {name}")

        self.num_dims = self.info.num_dims

    def _shekel_info(self) -> BenchmarkInfo:
        lower = torch.tensor(SHEKEL_LOWER, dtype=self.dtype)
        upper = torch.tensor(SHEKEL_UPPER, dtype=self.dtype)

        x_min_original = torch.tensor(SHEKEL_X_MIN, dtype=self.dtype)
        x_max_unit = self.to_unit(x_min_original, lower, upper)

        # Adapts the problem to maximization
        return BenchmarkInfo(name="shekel", num_dims=4, lower=lower,
            upper=upper, x_opt_original=x_min_original, x_opt_unit=x_max_unit,
            f_opt=-SHEKEL_F_MIN)

    @staticmethod
    def to_original(x_unit: torch.Tensor, lower: torch.Tensor,upper: torch.Tensor) -> torch.Tensor:
        return lower + x_unit * (upper - lower)

    @staticmethod
    def to_unit(x_original: torch.Tensor, lower: torch.Tensor, upper: torch.Tensor) -> torch.Tensor:
        return (x_original - lower) / (upper - lower)
    

    def f(self, x_unit: torch.Tensor) -> torch.Tensor:
        x_unit = torch.as_tensor(x_unit, dtype=self.dtype)

        if x_unit.ndim == 1:
            x_unit = x_unit.reshape(1, -1)

        if x_unit.shape[-1] != self.num_dims:
            raise ValueError(
                f"Expected dimension {self.num_dims}, got {x_unit.shape[-1]}."
            )

        with torch.no_grad():
            # Definition of the shekel function
            values = -self._shekel_min(x_unit)

        return values.reshape(1, -1)

    def f_noisy(self, x_unit: torch.Tensor) -> torch.Tensor:
        return self.f(x_unit)

    def __call__(self, x_unit: torch.Tensor) -> torch.Tensor:
        return self.f(x_unit)

    def _shekel_min(self, x_unit: torch.Tensor) -> torch.Tensor:
        x = self.to_original(x_unit, self.info.lower, self.info.upper)

        beta = torch.tensor(SHEKEL_BETA, dtype=self.dtype, device=x.device)
        C = torch.tensor(SHEKEL_C, dtype=self.dtype, device=x.device)

        # Shape: n x m
        sq_dist = ((x.unsqueeze(-1) - C.unsqueeze(0)) ** 2).sum(dim=1)

        # f(x) = -sum(1 / (||x - C_i||^2 + beta_i))
        return -(1.0 / (sq_dist + beta)).sum(dim=1)


if __name__ == "__main__":
    problem = BenchmarkProblem("shekel")

    x_opt = problem.info.x_opt_unit.reshape(1, -1)
    y_opt = problem.f(x_opt)

    print("Benchmark:", problem.info.name)
    print("Dimension:", problem.info.num_dims)
    print("x_opt original:", problem.info.x_opt_original)
    print("x_opt unit:", problem.info.x_opt_unit)
    print("f(x_opt):", y_opt)
    print("Known f_opt:", problem.info.f_opt)

    x_test = torch.rand(5, problem.num_dims, dtype=torch.float64)
    y_test = problem.f(x_test)

    print("x_test shape:", x_test.shape)
    print("y_test shape:", y_test.shape)
