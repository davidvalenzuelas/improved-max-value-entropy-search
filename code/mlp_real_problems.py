#!/usr/bin/env python3
# coding: utf-8
""" Implements the MLP real problems for MES+ experiments

Authors: Daniel Fernández Sánchez, Daniel Hernández-Lobato, David Valenzuela Sánchez
"""
import timeit
from enum import Enum

import numpy as np
import pandas as pd
import torch
from sklearn.datasets import fetch_openml
from sklearn.model_selection import RepeatedStratifiedKFold, cross_val_score
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler

from util import rescale_arguments


class Dataset(Enum):
    AUSTRALIAN_CREDIT = 40981
    IONOSPHERE= 59


class MLP_real_problems():

    def __init__(self, dataset_id, maximize: bool = True,
                 n_folds: int = 5, n_repeats: int = 1, cache_evals: bool = True, seed: int = 0):

        self.num_dims = 5
        self.maximize = maximize
        self.n_repeats_remove_noise = n_repeats
        self.n_folds = n_folds

        np_state = np.random.get_state()
        torch_state = torch.random.get_rng_state()

        self.data = fetch_openml(data_id=dataset_id, as_frame=True)
        X, y = self.data.data, self.data.target

        categorical_cols = X.select_dtypes(include=["object", "category"]).columns
        if len(categorical_cols) > 0:
            X_dummies = pd.get_dummies(X[categorical_cols], drop_first=True).astype(int)
            X_non_categorical = X.drop(columns=categorical_cols)
            self.X = pd.concat([X_non_categorical, X_dummies], axis=1)
        else:
            self.X = X.copy()

        self.X = self.X.apply(pd.to_numeric, errors="coerce")
        self.X = self.X.fillna(self.X.median(numeric_only=True))

        self.y = LabelEncoder().fit_transform(y.astype(str))
        self.seed = seed

        np.random.set_state(np_state)
        torch.random.set_rng_state(torch_state)

        self.cache_evals = cache_evals
        self.d_evaluations = dict()

    def _objective_function(self, args, n_repeats: int = 1):
        np_state = np.random.get_state()
        torch_state = torch.random.get_rng_state()

        np.random.seed(self.seed)

        params = rescale_arguments(args.tolist())
        hidden_layer_sizes = (params["width"],) * params["depth"]

        pipeline = Pipeline([
            ("scaler", StandardScaler()),
            ("mlp", MLPClassifier(
                hidden_layer_sizes=hidden_layer_sizes,
                alpha=params["alpha"],
                batch_size=params["batch_size"],
                learning_rate_init=params["learning_rate_init"],
                max_iter=params["max_iters"],
                random_state=self.seed,
            )),
        ])

        cv = RepeatedStratifiedKFold(
            n_splits=self.n_folds,
            n_repeats=n_repeats,
            random_state=self.seed,
        )

        import warnings
        warnings.filterwarnings("ignore")
        scores = cross_val_score(pipeline, self.X, self.y, cv=cv, scoring="accuracy",
                    n_jobs=1, error_score="raise")
        mean_accuracy = np.mean(scores)

        print(f"Mean Accuracy: {mean_accuracy:.2f} ; Scores: {scores.flatten()}")

        np.random.set_state(np_state)
        torch.random.set_rng_state(torch_state)

        return mean_accuracy

    def _evaluate_batch(self, X, n_repeats: int):
        X = torch.as_tensor(X, dtype=torch.double)
        X_batch = X.reshape(1, -1) if X.ndim == 1 else X
        if X_batch.shape[-1] != self.num_dims:
            raise ValueError(
                f"Expected dimension {self.num_dims}, "
                f"got {X_batch.shape[-1]}."
            )
        res = torch.zeros((X_batch.shape[0], 1), dtype=torch.double)

        for i, x in enumerate(X_batch):
            key = (int(n_repeats), *tuple(np.round(x.detach().cpu().numpy(),12)),)
            if self.cache_evals and key in self.d_evaluations:
                res[i, 0] = self.d_evaluations[key]
                continue
            value = self._objective_function(x, n_repeats=n_repeats)
            res[i, 0] = value
            if self.cache_evals:
                self.d_evaluations[key] = value

        return res

    def f(self, X):
        return self._evaluate_batch(X, n_repeats=self.n_repeats_remove_noise)

    def f_noisy(self, X):
        return self._evaluate_batch(X, n_repeats=1)


if __name__ == "__main__":
    args = np.array([1.0, 0.0, 0.5, 0.5, 1.0])

    def time_objective_function(mlp_class, args, tag, n_repeats):
        start_time = timeit.default_timer()
        result = mlp_class._objective_function(args=args, n_repeats=n_repeats)
        elapsed_time = timeit.default_timer() - start_time
        print(f"{tag} result: {result}, Time elapsed: {elapsed_time:.5f} seconds, n_repeats={n_repeats}")

    mlp_problem_class = MLP_real_problems(dataset_id=Dataset.AUSTRALIAN_CREDIT.value, n_repeats=1)
    time_objective_function(mlp_problem_class, args, "AUSTRALIAN_CREDIT", n_repeats=1)
