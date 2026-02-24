#!/usr/bin/env python3
# coding: utf-8

import argparse
import time
import numpy as np
import scipy as sp
import torch

from botorch.utils.sampling import draw_sobol_samples
from botorch.optim import optimize_acqf
from botorch.acquisition.joint_entropy_search import qJointEntropySearch

from botorch.models.gp_regression import SingleTaskGP
from botorch.models.transforms.outcome import Standardize
from botorch.fit import fit_gpytorch_mll
from gpytorch.mlls.exact_marginal_log_likelihood import ExactMarginalLogLikelihood

from synthetic_problem import Synthetic_problem
from vfe_sparse_gp import fit_model_vfe_sparse, pack_state_dict, as_botorch_model


# -----------------------------
# Repro
# -----------------------------
def set_seed(seed: int):
    torch.manual_seed(seed)
    np.random.seed(seed)


# -----------------------------
# Ground truth optimum y* (oracle, for metrics)
# NOTE: this is only for evaluation/benchmarking.
# -----------------------------
def get_maximum_problem(num_dims: int, problem_callable, size_grid: int = 20000):
    grid = torch.rand(size_grid * num_dims, num_dims, dtype=torch.double)
    vals = problem_callable(grid).view(-1)

    x0 = grid[vals.argmax()].detach().clone()

    def f_np(x_np):
        x_t = torch.from_numpy(np.atleast_2d(x_np)).double()
        return -float(problem_callable(x_t).view(-1)[0].detach().cpu().numpy())

    result = sp.optimize.fmin_l_bfgs_b(
        f_np,
        x0.cpu().numpy(),
        None,
        bounds=[(0.0, 1.0)] * num_dims,
        approx_grad=True,
    )

    x_opt = torch.from_numpy(result[0]).double()
    y_opt = problem_callable(x_opt.view(1, num_dims)).view(-1)[0].double()
    return x_opt, float(y_opt.item())


# -----------------------------
# Optimal samples for JES (grid-MC)
# Works for both:
# - your variational GP (model(X) -> MVN)
# - baseline SingleTaskGP (model(X) -> MVN)
# -----------------------------
def get_optimal_samples_grid_mc(model, bounds: torch.Tensor, num_optima: int, num_grid: int = 4096):
    bounds = bounds.double()
    d = bounds.shape[-1]

    X = draw_sobol_samples(bounds=bounds, n=num_grid, q=1).squeeze(1).double()  # (num_grid, d)

    model.eval()
    if hasattr(model, "likelihood") and model.likelihood is not None:
        model.likelihood.eval()

    with torch.no_grad():
        post = model(X)  # MVN, mean shape (num_grid)
        base_samples = torch.randn(
            num_optima, X.shape[0],
            device=post.mean.device,
            dtype=post.mean.dtype,
        )
        samples = post.rsample(base_samples=base_samples)  # (num_optima, num_grid)

        idx = samples.argmax(dim=-1)  # (num_optima,)
        optimal_inputs = X[idx, :]  # (num_optima, d)
        optimal_outputs = samples[torch.arange(num_optima, device=X.device), idx].unsqueeze(-1)  # (num_optima, 1)

    return optimal_inputs, optimal_outputs


# -----------------------------
# Sanity: violations of ceiling y*
# We count how many random points have posterior mean > y*
# (Not a BO metric; only checks your constraint is doing something.)
# -----------------------------
def count_mean_violations(model_botorch, y_star: float, bounds: torch.Tensor, n_probe: int = 2000):
    Xp = draw_sobol_samples(bounds=bounds.double(), n=n_probe, q=1).squeeze(1).double()
    with torch.no_grad():
        mu = model_botorch.posterior(Xp).mean.view(-1)
    return int((mu > y_star).sum().item())


# -----------------------------
# Fit baseline exact GP
# -----------------------------
def fit_baseline_exact_gp(train_X: torch.Tensor, train_Y: torch.Tensor):
    model = SingleTaskGP(train_X, train_Y, outcome_transform=Standardize(m=1))
    mll = ExactMarginalLogLikelihood(model.likelihood, model)
    fit_gpytorch_mll(mll)
    return model


# -----------------------------
# Fit your conditioned VFE sparse GP
# -----------------------------
def fit_constrained_sparse(
    train_X: torch.Tensor,
    train_Y: torch.Tensor,
    prev_model,
    *,
    M: int,
    training_iter: int,
    lr: float,
    noise_eps: float,
    y_star: float,
    Xc: torch.Tensor,
    num_constraint_points: int,
    tau: float,
    mc_samples: int,
    constraint_weight: float,
):
    state = None
    if prev_model is not None:
        state = pack_state_dict(prev_model, prev_model.likelihood)

    model, likelihood = fit_model_vfe_sparse(
        train_X=train_X,
        train_Y=train_Y,
        state_dict=state,
        M=M,
        training_iter=training_iter,
        lr=lr,
        noise_eps=noise_eps,
        verbose=False,
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
# Build JES acquisition and optimize
# -----------------------------
def propose_candidate_via_jes(
    model_botorch,
    model_for_optima_sampling,
    bounds: torch.Tensor,
    *,
    num_optima: int,
    num_grid_optima: int,
    num_restarts: int,
    raw_samples: int,
    debug: bool,
):
    optimal_inputs, optimal_outputs = get_optimal_samples_grid_mc(
        model_for_optima_sampling,
        bounds=bounds,
        num_optima=num_optima,
        num_grid=num_grid_optima,
    )

    acq = qJointEntropySearch(
        model=model_botorch,
        optimal_inputs=optimal_inputs.double(),
        optimal_outputs=optimal_outputs.double(),
        estimation_type="LB",
        condition_noiseless=True,
    )

    if debug:
        Xdbg = torch.rand(64, 1, bounds.shape[-1], dtype=torch.double)
        with torch.no_grad():
            v = acq(Xdbg)
        print("  [DEBUG] acq finite:", bool(torch.isfinite(v).all()), "shape:", tuple(v.shape))

    candidate, acq_value = optimize_acqf(
        acq_function=acq,
        bounds=bounds.double(),
        q=1,
        num_restarts=num_restarts,
        raw_samples=raw_samples,
    )
    candidate = candidate.view(-1, bounds.shape[-1]).double()  # (1,d)
    return candidate, acq_value


# -----------------------------
# Main experiment runner
# -----------------------------
def run_one(
    *,
    tag: str,
    problem_callable,
    bounds: torch.Tensor,
    y_star: float,
    X_init: torch.Tensor,
    Y_init: torch.Tensor,
    T: int,
    num_optima: int,
    num_grid_optima: int,
    num_restarts: int,
    raw_samples: int,
    debug: bool,
    # constrained sparse settings (ignored if baseline)
    use_constrained_sparse: bool,
    Xc: torch.Tensor | None,
    num_constraint_points: int,
    tau: float,
    mc_samples: int,
    constraint_weight: float,
    M: int,
    training_iter: int,
    lr: float,
    noise_eps: float,
):
    X = X_init.clone()
    Y = Y_init.clone()

    best_hist = []
    viol_hist = []

    prev_sparse = None

    t0 = time.time()

    for it in range(T):
        print(f"\n[{tag}] iter {it}/{T-1}")

        # 1) Fit model
        if use_constrained_sparse:
            assert Xc is not None, "Xc must be provided for constrained sparse run"
            sparse = fit_constrained_sparse(
                X, Y, prev_sparse,
                M=M, training_iter=training_iter, lr=lr, noise_eps=noise_eps,
                y_star=y_star, Xc=Xc,
                num_constraint_points=num_constraint_points, tau=tau,
                mc_samples=mc_samples, constraint_weight=constraint_weight,
            )
            prev_sparse = sparse

            model_botorch = as_botorch_model(
                sparse,
                y_star=y_star,
                Xc=Xc,
                num_constraint_points=num_constraint_points,
                tau=tau,
                mc_samples=mc_samples,
                constraint_weight=constraint_weight,
            )

            # sanity: wrapper fantasy works?
            if debug and it == 0:
                Xf = torch.rand(3, 1, bounds.shape[-1], dtype=torch.double)
                Yf = torch.rand(3, 1, 1, dtype=torch.double)
                mf = model_botorch.condition_on_observations(Xf, Yf)
                print("  [DEBUG] fantasy ok; y_star base:", getattr(model_botorch, "y_star", None),
                      "y_star fantasy:", getattr(mf, "y_star", None))

            # sanity: gradients exist?
            if debug and it == 0:
                Xt = torch.rand(4, bounds.shape[-1], dtype=torch.double, requires_grad=True)
                post = model_botorch.posterior(Xt)
                loss = post.mean.sum()
                loss.backward()
                print("  [DEBUG] posterior grad norm:", float(Xt.grad.norm()))

            # propose via JES using wrapper + optima from the latent model
            candidate, acq_value = propose_candidate_via_jes(
                model_botorch=model_botorch,
                model_for_optima_sampling=sparse,  # (uses model(X)->MVN)
                bounds=bounds,
                num_optima=num_optima,
                num_grid_optima=num_grid_optima,
                num_restarts=num_restarts,
                raw_samples=raw_samples,
                debug=debug,
            )
        else:
            exact = fit_baseline_exact_gp(X, Y)
            model_botorch = exact  # already a BoTorch model
            candidate, acq_value = propose_candidate_via_jes(
                model_botorch=model_botorch,
                model_for_optima_sampling=exact,
                bounds=bounds,
                num_optima=num_optima,
                num_grid_optima=num_grid_optima,
                num_restarts=num_restarts,
                raw_samples=raw_samples,
                debug=debug,
            )

        print(f"  candidate: {candidate.detach().cpu().numpy()}  acq={float(acq_value)}")

        # 2) Evaluate objective ONCE and append ONCE (fixed vs your earlier debug script)
        y_new = problem_callable(candidate).double().view(-1, 1)
        X = torch.cat([X, candidate], dim=0)
        Y = torch.cat([Y, y_new], dim=0)

        best = float(Y.max().item())
        best_hist.append(best)

        # 3) Optional sanity: ceiling violations (only meaningful for constrained run)
        if use_constrained_sparse:
            viol = count_mean_violations(model_botorch, y_star, bounds, n_probe=2000)
            viol_hist.append(viol)
            print(f"  best_y={best:.6f}  simple_regret={y_star - best:.6f}  violations(mu>y*)={viol}")
        else:
            print(f"  best_y={best:.6f}  simple_regret={y_star - best:.6f}")

        # 4) Candidate novelty sanity
        dmin = float(torch.cdist(candidate, X[:-1]).min())
        print("  min dist to previous:", dmin)

    elapsed = time.time() - t0
    return {
        "tag": tag,
        "X": X,
        "Y": Y,
        "best_hist": best_hist,
        "best_final": best_hist[-1],
        "regret_final": y_star - best_hist[-1],
        "viol_hist": viol_hist,
        "elapsed_sec": elapsed,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--dims", type=int, default=4)
    p.add_argument("--T", type=int, default=12)
    p.add_argument("--n_init", type=int, default=8)
    p.add_argument("--ls_model", type=float, default=0.25)
    p.add_argument("--debug", action="store_true")

    # JES params
    p.add_argument("--num_optima", type=int, default=16)
    p.add_argument("--num_grid_optima", type=int, default=4096)
    p.add_argument("--raw_samples", type=int, default=256)
    p.add_argument("--num_restarts", type=int, default=10)

    # constrained sparse params
    p.add_argument("--M", type=int, default=64)
    p.add_argument("--training_iter", type=int, default=250)
    p.add_argument("--lr", type=float, default=0.01)
    p.add_argument("--noise_eps", type=float, default=1e-6)

    p.add_argument("--num_constraint_points", type=int, default=100)
    p.add_argument("--tau", type=float, default=0.05)
    p.add_argument("--mc_samples", type=int, default=16)
    p.add_argument("--constraint_weight", type=float, default=10.0)

    args = p.parse_args()

    set_seed(args.seed)

    # bounds [0,1]^d
    bounds = torch.tensor([[0.0] * args.dims, [1.0] * args.dims], dtype=torch.double)

    # Synthetic problem: use callable .f (matches your my_loop approach)
    synthetic_problem = Synthetic_problem(num_dims=args.dims, lengthscale_model=args.ls_model, seed=args.seed)
    problem = synthetic_problem.f  # callable

    # Oracle y* for evaluation (not needed for running BO, only for metrics)
    x_star, y_star = get_maximum_problem(args.dims, problem_callable=problem, size_grid=20000)
    print("Oracle approx y_star:", y_star)
    print("Oracle x_star:", x_star.detach().cpu().numpy())

    # Fixed initial design: identical for both runs (fair comparison)
    X_init = draw_sobol_samples(bounds=bounds, n=args.n_init, q=1).squeeze(1).double()
    Y_init = problem(X_init).double().view(-1, 1)

    # Fixed constraint points Xc (this is the key)
    Xc = draw_sobol_samples(bounds=bounds, n=args.num_constraint_points, q=1).squeeze(1).double()

    # Run baseline
    baseline = run_one(
        tag="baseline_exactGP_JES",
        problem_callable=problem,
        bounds=bounds,
        y_star=y_star,
        X_init=X_init,
        Y_init=Y_init,
        T=args.T,
        num_optima=args.num_optima,
        num_grid_optima=args.num_grid_optima,
        num_restarts=args.num_restarts,
        raw_samples=args.raw_samples,
        debug=args.debug,
        use_constrained_sparse=False,
        Xc=None,
        num_constraint_points=args.num_constraint_points,
        tau=args.tau,
        mc_samples=args.mc_samples,
        constraint_weight=args.constraint_weight,
        M=args.M,
        training_iter=args.training_iter,
        lr=args.lr,
        noise_eps=args.noise_eps,
    )

    # Run your method
    constrained = run_one(
        tag="constrained_sparse_JES",
        problem_callable=problem,
        bounds=bounds,
        y_star=y_star,
        X_init=X_init,
        Y_init=Y_init,
        T=args.T,
        num_optima=args.num_optima,
        num_grid_optima=args.num_grid_optima,
        num_restarts=args.num_restarts,
        raw_samples=args.raw_samples,
        debug=args.debug,
        use_constrained_sparse=True,
        Xc=Xc,
        num_constraint_points=args.num_constraint_points,
        tau=args.tau,
        mc_samples=args.mc_samples,
        constraint_weight=args.constraint_weight,
        M=args.M,
        training_iter=args.training_iter,
        lr=args.lr,
        noise_eps=args.noise_eps,
    )

    # Summary
    print("\n=======================")
    print("SUMMARY")
    print("=======================")
    print("baseline best_final:", baseline["best_final"])
    print("baseline regret_final:", baseline["regret_final"])
    print("baseline elapsed_sec:", baseline["elapsed_sec"])
    print("")
    print("constrained best_final:", constrained["best_final"])
    print("constrained regret_final:", constrained["regret_final"])
    print("constrained elapsed_sec:", constrained["elapsed_sec"])
    if constrained["viol_hist"]:
        print("constrained violations last:", constrained["viol_hist"][-1])

    # Quick verdict heuristic (one seed only; do multiple seeds for real conclusion)
    if constrained["regret_final"] < baseline["regret_final"]:
        print("\n[HEURISTIC] constrained seems better on this seed (lower final regret).")
    else:
        print("\n[HEURISTIC] baseline seems better on this seed (lower final regret). Try more seeds.")


if __name__ == "__main__":
    main()