import time
import numpy as np
import torch

from botorch.utils.sampling import draw_sobol_samples
from botorch.optim import optimize_acqf
from botorch.acquisition.joint_entropy_search import qJointEntropySearch

from botorch.models.gp_regression import SingleTaskGP
from botorch.models.transforms.outcome import Standardize
from gpytorch.mlls.exact_marginal_log_likelihood import ExactMarginalLogLikelihood
from botorch.fit import fit_gpytorch_mll

from synthetic_problem import Synthetic_problem
from vfe_sparse_gp import fit_model_vfe_sparse, as_botorch_model, pack_state_dict


# -----------------------------
# Helpers
# -----------------------------
def set_seed(seed: int = 0):
    torch.manual_seed(seed)
    np.random.seed(seed)


def eval_true(problem: Synthetic_problem, X: torch.Tensor, noisy: bool = False) -> torch.Tensor:
    """
    Evalúa tu función real usando problem.f / problem.f_noisy y devuelve (N, 1).
    Tu 'paths' suele devolver algo tipo (1, N) o (1, N, 1). Lo dejamos en (N, 1).
    """
    with torch.no_grad():
        y = problem.f_noisy(X) if noisy else problem.f(X)

    # y puede venir con dimensión de sample al principio: (1, N) o (1, N, 1)
    if y.dim() >= 2 and y.shape[0] == 1:
        y = y.squeeze(0)

    # ahora y típicamente es (N,) o (N,1)
    y = y.reshape(-1, 1)
    return y.double()


def oracle_y_star(problem, bounds: torch.Tensor, n_grid: int = 60000):
    """Aproximación al y* del problema real (solo para métricas)."""
    X = draw_sobol_samples(bounds=bounds, n=n_grid, q=1).squeeze(1).double()
    y = eval_true(problem, X, noisy=False).view(-1)
    return float(y.max().item())


def pick_optima_for_jes(model_botorch, bounds: torch.Tensor, num_optima: int = 64, grid: int = 4096):
    """
    Para JES necesitas (optimal_inputs, optimal_outputs).
    Aquí usamos una aproximación:
      - sampleamos una grilla Sobol
      - cogemos los top-k por posterior mean
    """
    Xg = draw_sobol_samples(bounds=bounds, n=grid, q=1).squeeze(1).double()
    with torch.no_grad():
        mu = model_botorch.posterior(Xg).mean.view(-1)

    k = min(num_optima, mu.numel())
    topk = torch.topk(mu, k=k).indices
    X_opt = Xg[topk].double()
    y_opt = mu[topk].double()
    return X_opt, y_opt


def eval_violations(model_botorch, y_star: float, bounds: torch.Tensor, n_test: int = 2000):
    """Cuenta cuántos puntos tienen posterior mean > y_star."""
    Xt = draw_sobol_samples(bounds=bounds, n=n_test, q=1).squeeze(1).double()
    with torch.no_grad():
        mu = model_botorch.posterior(Xt).mean.view(-1)
    return int((mu > y_star).sum().item())


def run_bo(
    *,
    name: str,
    problem,
    bounds: torch.Tensor,
    T: int = 12,
    n_init: int = 8,
    seed: int = 0,
    noisy_obj: bool = False,
    # constrained sparse GP
    use_constrained_sparse: bool = False,
    M: int = 64,
    training_iter: int = 250,
    lr: float = 0.01,
    noise_eps: float = 1e-6,
    y_star: float | None = None,
    Xc: torch.Tensor | None = None,
    num_constraint_points: int = 100,
    tau: float = 0.05,
    mc_samples: int = 16,
    constraint_weight: float = 10.0,
    # JES
    jes_num_optima: int = 64,
    jes_grid: int = 4096,
    num_restarts: int = 10,
    raw_samples: int = 256,
):
    set_seed(seed)

    # Inicialización
    X = draw_sobol_samples(bounds=bounds, n=n_init, q=1).squeeze(1).double()
    Y = eval_true(problem, X, noisy=noisy_obj)

    best_hist = []
    t0 = time.time()

    model_prev = None

    for it in range(T):
        # --- fit model ---
        if use_constrained_sparse:
            state = None
            if model_prev is not None:
                state = pack_state_dict(model_prev, model_prev.likelihood)

            model, likelihood = fit_model_vfe_sparse(
                train_X=X,
                train_Y=Y,
                state_dict=state,
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
            model_prev = model

            model_botorch = as_botorch_model(
                model,
                y_star=y_star,
                Xc=Xc,
                num_constraint_points=num_constraint_points,
                tau=tau,
                mc_samples=mc_samples,
                constraint_weight=constraint_weight,
            )
        else:
            # Baseline exact GP (tipo loop_BO.py)
            model = SingleTaskGP(X, Y, outcome_transform=Standardize(m=1))
            mll = ExactMarginalLogLikelihood(model.likelihood, model)
            fit_gpytorch_mll(mll)
            model_botorch = model

        # --- build JES ---
        optimal_inputs, optimal_outputs = pick_optima_for_jes(
            model_botorch, bounds=bounds, num_optima=jes_num_optima, grid=jes_grid
        )
        acq = qJointEntropySearch(
            model=model_botorch,
            optimal_inputs=optimal_inputs.double(),
            optimal_outputs=optimal_outputs.double(),
            estimation_type="LB",
        )

        # --- optimize acquisition ---
        candidate, _ = optimize_acqf(
            acq_function=acq,
            bounds=bounds.double(),
            q=1,
            num_restarts=num_restarts,
            raw_samples=raw_samples,
        )

        # --- evaluate true function ---
        y_new = eval_true(problem, candidate.double(), noisy=noisy_obj)

        X = torch.cat([X, candidate.double()], dim=0)
        Y = torch.cat([Y, y_new], dim=0)

        best = float(Y.max().item())
        best_hist.append(best)
        print(f"[{name}] iter={it:02d}  best_y={best:.6f}")

    elapsed = time.time() - t0
    return {
        "name": name,
        "best_hist": best_hist,
        "best_final": best_hist[-1],
        "elapsed_sec": elapsed,
        "X": X,
        "Y": Y,
    }


def main():
    device = torch.device("cpu")
    dtype = torch.double

    seed = 0
    num_dims = 2

    problem = Synthetic_problem(num_dims=num_dims, lengthscale_model=0.25, seed=seed)
    bounds = torch.tensor([[0.0] * num_dims, [1.0] * num_dims], dtype=dtype, device=device)

    # y* (solo para métrica / sanity)
    y_star = oracle_y_star(problem, bounds, n_grid=60000)
    print("Approx oracle y* =", y_star)

    # puntos de restricción fijos
    num_constraint_points = 100
    Xc = draw_sobol_samples(bounds=bounds, n=num_constraint_points, q=1).squeeze(1).double()

    # Baseline JES exact GP
    baseline = run_bo(
        name="baseline_JES_exactGP",
        problem=problem,
        bounds=bounds,
        T=12,
        n_init=8,
        seed=seed,
        noisy_obj=False,
        use_constrained_sparse=False,
    )

    # Constrained sparse + JES
    constrained = run_bo(
        name="constrainedSparse_JES",
        problem=problem,
        bounds=bounds,
        T=12,
        n_init=8,
        seed=seed,
        noisy_obj=False,
        use_constrained_sparse=True,
        M=64,
        training_iter=250,
        lr=0.01,
        noise_eps=1e-6,
        y_star=y_star,
        Xc=Xc,
        num_constraint_points=num_constraint_points,
        tau=0.05,
        mc_samples=16,
        constraint_weight=10.0,
    )

    # Métrica principal
    best_base = baseline["best_final"]
    best_cons = constrained["best_final"]
    simple_regret_base = y_star - best_base
    simple_regret_cons = y_star - best_cons

    print("\n=== RESULTS ===")
    print("baseline best_final:", best_base)
    print("constr.  best_final:", best_cons)
    print("baseline simple_regret (y* - best):", simple_regret_base)
    print("constr.  simple_regret (y* - best):", simple_regret_cons)
    print("baseline elapsed_sec:", baseline["elapsed_sec"])
    print("constr.  elapsed_sec:", constrained["elapsed_sec"])

    # Sanity check: violaciones del techo y* (posterior mean > y*)
    # baseline:
    model_base = SingleTaskGP(baseline["X"], baseline["Y"], outcome_transform=Standardize(m=1))
    mll = ExactMarginalLogLikelihood(model_base.likelihood, model_base)
    fit_gpytorch_mll(mll)
    v_base = eval_violations(model_base, y_star, bounds)

    # constrained:
    model_cons, lik_cons = fit_model_vfe_sparse(
        train_X=constrained["X"],
        train_Y=constrained["Y"],
        state_dict=None,
        M=64,
        training_iter=200,
        lr=0.01,
        noise_eps=1e-6,
        verbose=False,
        y_star=y_star,
        Xc=Xc,
        num_constraint_points=num_constraint_points,
        tau=0.05,
        mc_samples=16,
        constraint_weight=10.0,
    )
    model_cons.likelihood = lik_cons
    model_cons_botorch = as_botorch_model(
        model_cons,
        y_star=y_star,
        Xc=Xc,
        num_constraint_points=num_constraint_points,
        tau=0.05,
        mc_samples=16,
        constraint_weight=10.0,
    )
    v_cons = eval_violations(model_cons_botorch, y_star, bounds)

    print("\n=== Constraint sanity check ===")
    print("Violations baseline (mu(x) > y*):", v_base)
    print("Violations constrained (mu(x) > y*):", v_cons)


if __name__ == "__main__":
    main()