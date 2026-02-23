#!/usr/bin/env python3
# coding: utf-8

import os
import sys
import numpy as np
import scipy as sp
import torch

from botorch.utils.sampling import draw_sobol_samples
from botorch.optim import optimize_acqf
from botorch.acquisition.joint_entropy_search import qJointEntropySearch

from renyi_entropy_search import qRenyiEntropySearch
from renyi_entropy_search_ensemble import qRenyiEntropySearchEnsemble
from renyi_entropy_search_hedge import qRenyiEntropySearchHedge

from util import reset_random_state, read_config, create_path
from plotter import Plotter
from synthetic_problem import Synthetic_problem
from vfe_sparse_gp import fit_model_vfe_sparse, pack_state_dict, as_botorch_model

# -----------------------------
# GLOBALS / DEFAULTS
# -----------------------------
SCALE_ACQ_VALS = True
RESOLUTION = 20
SIZE_GRID = 10000

# Defaults for BO optimize_acqf (override from config if present)
DEFAULT_NUM_RESTARTS = 20
DEFAULT_RAW_SAMPLES = 512

# Defaults for sparse GP training (override from config if present)
DEFAULT_M = 64
DEFAULT_TRAINING_ITER = 500
DEFAULT_LR = 0.01
DEFAULT_NOISE_EPS = 1e-6

# Defaults for conditioning to y*
DEFAULT_NUM_CONSTRAINT_POINTS = 100
DEFAULT_TAU = 0.05
DEFAULT_MC_SAMPLES = 16
DEFAULT_CONSTRAINT_WEIGHT = 10.0

# Debug (set False for final runs)
DEBUG = False


# -----------------------------
# MAXIMUM OF TRUE PROBLEM (y*)
# -----------------------------
def get_maximum_problem(num_dims, problem):
    grid = torch.rand(SIZE_GRID * num_dims, num_dims, dtype=torch.double)
    vals = problem(grid).view(-1)

    x0 = grid[vals.argmax()].detach().clone()
    y0 = vals.max().detach().clone()

    def f_np(x_np):
        x_t = torch.from_numpy(np.atleast_2d(x_np)).double()
        return -float(problem(x_t).view(-1)[0].detach().cpu().numpy())

    result = sp.optimize.fmin_l_bfgs_b(
        f_np,
        x0.cpu().numpy(),
        None,
        bounds=[(0.0, 1.0)] * num_dims,
        approx_grad=True,
    )

    x_opt = torch.from_numpy(result[0]).double()
    y_opt = problem(x_opt.view(1, num_dims)).view(-1)[0].double()
    return x_opt, y_opt


# -----------------------------
# OPTIMIZE POSTERIOR MEAN (optional recommendation)
# -----------------------------
def optimize_posterior_mean(model, num_dims):
    def f_np(x_np):
        x_t = torch.from_numpy(np.atleast_2d(x_np)).double()
        return -float(model(x_t).mean.view(-1)[0].detach().cpu().numpy())

    grid = torch.rand(SIZE_GRID * num_dims, num_dims, dtype=torch.double)
    vals = np.array([f_np(x) for x in grid.cpu().numpy()])

    x0 = grid[vals.argmin()].cpu().numpy()
    result = sp.optimize.fmin_l_bfgs_b(
        f_np, x0, None, bounds=[(0.0, 1.0)] * num_dims, approx_grad=True
    )
    return torch.from_numpy(result[0]).double()


# -----------------------------
# FIT MODEL (VFE sparse GP + ADAM)
# -----------------------------
def fit_model(
    train_X,
    train_Y,
    state_dict=None,
    *,
    M=DEFAULT_M,
    training_iter=DEFAULT_TRAINING_ITER,
    lr=DEFAULT_LR,
    noise_eps=DEFAULT_NOISE_EPS,
    # conditioning
    y_star=None,
    Xc = None,
    num_constraint_points=DEFAULT_NUM_CONSTRAINT_POINTS,
    tau=DEFAULT_TAU,
    mc_samples=DEFAULT_MC_SAMPLES,
    constraint_weight=DEFAULT_CONSTRAINT_WEIGHT,
):
    model, likelihood = fit_model_vfe_sparse(
        train_X=train_X,
        train_Y=train_Y,
        state_dict=state_dict,
        M=M,
        training_iter=training_iter,
        lr=lr,
        noise_eps=noise_eps,
        verbose=False,
        # conditioning
        y_star=y_star,
        Xc=Xc,
        num_constraint_points=num_constraint_points,
        tau=tau,
        mc_samples=mc_samples,
        constraint_weight=constraint_weight,
    )
    model.likelihood = likelihood
    return model


# -----------------------------
# Optimal samples for JES: grid MC (stable dtype)
# -----------------------------
def get_optimal_samples_grid_mc(model, bounds: torch.Tensor, num_optima: int, num_grid: int = 4096):
    bounds = bounds.double()
    X = draw_sobol_samples(bounds=bounds, n=num_grid, q=1).squeeze(1).double()

    model.eval()
    if hasattr(model, "likelihood"):
        model.likelihood.eval()

    with torch.no_grad():
        post = model(X)
        base_samples = torch.randn(
            num_optima, X.shape[0],
            device=post.mean.device,
            dtype=post.mean.dtype,
        )
        samples = post.rsample(base_samples=base_samples)  # (num_optima, num_grid)

        idx = samples.argmax(dim=-1)
        optimal_inputs = X[idx, :]  # (num_optima, d)
        optimal_outputs = samples[torch.arange(num_optima, device=X.device), idx].unsqueeze(-1)  # (num_optima,1)

    return optimal_inputs, optimal_outputs


# -----------------------------
# MAIN
# -----------------------------
def main():
    if len(sys.argv) != 2:
        print("Usage: python loop_BO.py <config.json>")
        sys.exit(1)

    config = read_config(sys.argv[1])

    seed = int(config["random_seed"])
    ls_model = float(config["lenghtscale_model_synthetic_problem"])
    num_initial_obs = int(config["num_initial_obs"])
    num_samples = int(config["num_samples_solution"])
    BO_iters = int(config["BO_iters"])
    acquisition_name = config["acquisition"]

    # This project assumes d=4 (keep as you had)
    num_dims = 4

    # Output path
    create_path(config["file_results"])
    reset_random_state(seed)

    # Bounds [0,1]^d
    bounds = torch.tensor([[0.0] * num_dims, [1.0] * num_dims], dtype=torch.double)

    if num_dims == 2:
        Plotter(num_dims=num_dims, bounds=bounds, resolution=RESOLUTION, path=config["file_results"])

    # Problem
    synthetic_problem = Synthetic_problem(num_dims=num_dims, lengthscale_model=ls_model, seed=seed)
    problem = synthetic_problem.f
    problem_noiseless = synthetic_problem.f

    # True optimum (assumed known)
    x_star, y_star = get_maximum_problem(num_dims=num_dims, problem=problem)

    np.savetxt(
        config["file_results"] + "/x_optimum_problem.txt",
        x_star.detach().cpu().numpy().reshape((1, num_dims)),
    )
    np.savetxt(
        config["file_results"] + "/y_optimum_problem.txt",
        np.array([float(y_star.detach().cpu().numpy())]),
    )

    # Initial observations
    x_observations = torch.rand(num_initial_obs, num_dims, dtype=torch.double)
    y_values_obs = problem(x_observations).double().view(-1, 1)  # IMPORTANT: (n,1)

    # Resume if exists
    if os.path.exists("points_evaluated.txt") and os.path.exists("y_values_evaluated.txt"):
        x_observations = torch.from_numpy(np.loadtxt("points_evaluated.txt", ndmin=2)).double()
        y_loaded = torch.from_numpy(np.loadtxt("y_values_evaluated.txt", ndmin=2)).double()
        y_values_obs = y_loaded.view(-1, 1)

        # Adjust remaining iterations
        BO_iters = BO_iters - (x_observations.shape[0] - num_initial_obs)
        BO_iters = max(0, BO_iters)

    # Read hyperparams from config if present
    num_restarts_opt = int(config.get("num_restarts_opt", DEFAULT_NUM_RESTARTS))
    raw_samples_opt_acq = int(config.get("raw_samples_opt_acq", DEFAULT_RAW_SAMPLES))

    M = int(config.get("sparse_M", DEFAULT_M))
    training_iter = int(config.get("training_iter", DEFAULT_TRAINING_ITER))
    lr = float(config.get("lr", DEFAULT_LR))
    noise_eps = float(config.get("noise_eps", DEFAULT_NOISE_EPS))

    num_constraint_points = int(config.get("num_constraint_points", DEFAULT_NUM_CONSTRAINT_POINTS))
    tau = float(config.get("tau", DEFAULT_TAU))
    mc_samples = int(config.get("mc_samples", DEFAULT_MC_SAMPLES))
    constraint_weight = float(config.get("constraint_weight", DEFAULT_CONSTRAINT_WEIGHT))

    model = None

    bounds = torch.tensor([[0.0] * num_dims, [1.0] * num_dims], dtype=torch.double)
    
    # -----------------------------
    # FIXED constraint points Xc (sample ONCE, reuse always)
    # -----------------------------
    Xc = draw_sobol_samples(
        bounds=bounds.double(),
        n=num_constraint_points,
        q=1,
    ).squeeze(1).double()
    # -----------------------------
    # BO LOOP
    # -----------------------------
    for bo_iteration in range(BO_iters):
        print(f"\n=== BO Iteration {bo_iteration} / {BO_iters - 1} ===")
        print("Fitting the model (VFE sparse GP, conditioned on y*) ...")

        if model is not None:
            prev_state = pack_state_dict(model, model.likelihood)
            model = fit_model(
                x_observations,
                y_values_obs.detach(),
                state_dict=prev_state,
                M=M,
                training_iter=training_iter,
                lr=lr,
                noise_eps=noise_eps,
                y_star=y_star,
                Xc=Xc,
                num_constraint_points=num_constraint_points,
                tau=tau,
                mc_samples=mc_samples,
                constraint_weight=constraint_weight,
            )
        else:
            model = fit_model(
                x_observations,
                y_values_obs.detach(),
                state_dict=None,
                M=M,
                training_iter=training_iter,
                lr=lr,
                noise_eps=noise_eps,
                y_star=y_star,
                Xc=Xc,
                num_constraint_points=num_constraint_points,
                tau=tau,
                mc_samples=mc_samples,
                constraint_weight=constraint_weight,
            )

        if DEBUG:
            print("train_X:", tuple(x_observations.shape), "train_Y:", tuple(y_values_obs.shape))
            print("y_star:", float(y_star))

        # -----------------------------
        # Prepare optimal samples (for entropy acquisitions)
        # -----------------------------
        print("Preparing optimal samples ...")
        optimal_inputs, optimal_outputs = get_optimal_samples_grid_mc(
            model, bounds=bounds, num_optima=num_samples, num_grid=4096
        )

        # -----------------------------
        # Choose acquisition + optimize
        # -----------------------------
        print("Optimizing acquisition ...")

        if acquisition_name == "JES":
            # Build BoTorch wrapper (must carry y_star for fantasy refits)
            model_botorch = as_botorch_model(
                model,
                y_star=y_star,
                Xc=Xc,
                num_constraint_points=num_constraint_points,
                tau=tau,
                mc_samples=mc_samples,
                constraint_weight=constraint_weight,
            )

            acq = qJointEntropySearch(
                model=model_botorch,
                optimal_inputs=optimal_inputs.double(),
                optimal_outputs=optimal_outputs.double(),
                estimation_type="LB",
                condition_noiseless=True,
            )

            candidate, acq_value = optimize_acqf(
                acq_function=acq,
                bounds=bounds,
                q=1,
                num_restarts=num_restarts_opt,
                raw_samples=raw_samples_opt_acq,
            )

        elif acquisition_name == "RES":
            alpha = float(config["alpha"])
            acq = qRenyiEntropySearch(
                model=model,
                optimal_inputs=optimal_inputs.double(),
                optimal_outputs=optimal_outputs.double(),
                alpha=alpha,
            )
            candidate, acq_value = optimize_acqf(
                acq_function=acq,
                bounds=bounds,
                q=1,
                num_restarts=num_restarts_opt,
                raw_samples=raw_samples_opt_acq,
            )

        elif acquisition_name == "RES_ENS":
            alphas = config.get("alphas", [0.001, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.999])
            l_max_acqs = []
            for a in alphas:
                acq_tmp = qRenyiEntropySearch(
                    model=model,
                    optimal_inputs=optimal_inputs.double(),
                    optimal_outputs=optimal_outputs.double(),
                    alpha=float(a),
                )
                cand_tmp, val_tmp = optimize_acqf(
                    acq_function=acq_tmp,
                    bounds=bounds,
                    q=1,
                    num_restarts=num_restarts_opt,
                    raw_samples=raw_samples_opt_acq,
                )
                l_max_acqs.append(val_tmp)

            acq = qRenyiEntropySearchEnsemble(
                model=model,
                optimal_inputs=optimal_inputs.double(),
                optimal_outputs=optimal_outputs.double(),
                alphas=alphas,
                weights_alphas=l_max_acqs,
            )
            candidate, acq_value = optimize_acqf(
                acq_function=acq,
                bounds=bounds,
                q=1,
                num_restarts=num_restarts_opt,
                raw_samples=raw_samples_opt_acq,
            )

        elif acquisition_name == "RES_Hedge":
            alphas = config.get("alphas", [0.001, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.999])

            hres = qRenyiEntropySearchHedge(
                model=model,
                filename_rewards="vector_of_rewards_hres.txt",
                filename_candidates="d_candidates.dat",
                filename_selected_alphas="selected_alphas.txt",
                optimal_inputs=optimal_inputs.double(),
                optimal_outputs=optimal_outputs.double(),
                alphas=alphas,
                num_iters=BO_iters,
            )

            candidate, acq_value = hres.optimize_acqf(
                bounds=bounds,
                q=1,
                num_restarts_opt=num_restarts_opt,
                raw_samples_opt_acq=raw_samples_opt_acq,
            )

        else:
            raise ValueError(f"Unknown acquisition_name='{acquisition_name}'")

        # Ensure candidate is (1, d)
        candidate = candidate.view(-1, num_dims).double()
        print(f"{acquisition_name}: candidate={candidate.detach().cpu().numpy()}, acq_value={float(acq_value)}")

        # -----------------------------
        # Recommendations (3 types)
        # -----------------------------
        print("Computing recommendations ...")

        # 1) Best predicted mean among observed X
        with torch.no_grad():
            idx_pred = model(x_observations).mean.view(-1).argmax().item()
        rec1 = x_observations[idx_pred:idx_pred + 1, :]
        val1 = problem_noiseless(rec1).double().view(-1, 1)

        with open(f'{config["file_results"]}/recommendations_obs.txt', 'a') as f:
            np.savetxt(f, rec1.detach().cpu().numpy())
        with open(f'{config["file_results"]}/objective_at_recommendations_obs.txt', 'a') as f:
            np.savetxt(f, val1.detach().cpu().numpy())

        # 2) Best observed Y
        idx_obs = y_values_obs.view(-1).argmax().item()
        rec2 = x_observations[idx_obs:idx_obs + 1, :]
        val2 = problem_noiseless(rec2).double().view(-1, 1)

        with open(f'{config["file_results"]}/recommendations_obs_obs.txt', 'a') as f:
            np.savetxt(f, rec2.detach().cpu().numpy())
        with open(f'{config["file_results"]}/objective_at_recommendations_obs_obs.txt', 'a') as f:
            np.savetxt(f, val2.detach().cpu().numpy())

        # 3) Optimize posterior mean (continuous)
        rec3 = optimize_posterior_mean(model, num_dims).view(1, num_dims)
        val3 = problem_noiseless(rec3).double().view(-1, 1)

        with open(f'{config["file_results"]}/recommendations_post_mean.txt', 'a') as f:
            np.savetxt(f, rec3.detach().cpu().numpy())
        with open(f'{config["file_results"]}/objective_at_recommendations_post_mean.txt', 'a') as f:
            np.savetxt(f, val3.detach().cpu().numpy())

        # -----------------------------
        # Evaluate objective at candidate and append
        # -----------------------------
        print("Evaluating objective at candidate ...")
        y_new = problem(candidate).double().view(-1, 1)  # IMPORTANT: (1,1)

        x_observations = torch.cat([x_observations, candidate], dim=0)
        y_values_obs = torch.cat([y_values_obs, y_new], dim=0)

        best_y = float(y_values_obs.max().detach().cpu().numpy())
        print("Best observed so far:", best_y)

        # Save points evaluated so far
        np.savetxt("points_evaluated.txt", x_observations.detach().cpu().numpy())
        np.savetxt("y_values_evaluated.txt", y_values_obs.detach().cpu().numpy())

        sys.stdout.flush()

    print("\nEnd. Have a nice day!")


if __name__ == "__main__":
    main()