import io
import numpy as np
import requests
import torch
import matplotlib.pyplot as plt

import gpytorch

from vfe_sparse_gp import (
    VFESparseGP,
    StepConstraintVariationalELBO,
    train_model_ADAM,
    predictive_distribution,
    _normal_cdf,
)
from gpytorch.mlls import VariationalELBO
from gpytorch.constraints.constraints import GreaterThan


def load_snelson():
    url = "http://arantxa.ii.uam.es/~dhernan/MLAS2023/EdSnelson.npy"
    r = requests.get(url, timeout=30)
    r.raise_for_status()
    data = np.load(io.BytesIO(r.content), allow_pickle=False)

    x_train = torch.from_numpy(data[0]).double()
    y_train = torch.from_numpy(data[1]).double()
    if x_train.ndim == 1:
        x_train = x_train.unsqueeze(-1)
    return x_train, y_train


def sample_unit_box(n: int, d: int, method: str, seed: int = 0, dtype=torch.float64, device="cpu"):
    """Sample Xc in [0,1]^d deterministically."""
    if method == "rand":
        g = torch.Generator(device=device)
        g.manual_seed(seed)
        return torch.rand((n, d), generator=g, device=device, dtype=dtype)
    elif method == "sobol":
        # SobolEngine is deterministic given seed when scramble=True (if seed arg not available, manual_seed still helps)
        try:
            eng = torch.quasirandom.SobolEngine(dimension=d, scramble=True, seed=seed)
        except TypeError:
            torch.manual_seed(seed)
            eng = torch.quasirandom.SobolEngine(dimension=d, scramble=True)
        return eng.draw(n).to(device=device, dtype=dtype)
    else:
        raise ValueError("method must be 'rand' or 'sobol'")


@torch.no_grad()
def prob_f_below_y_star(model, Xc, y_star):
    """Average/min/max P(f(Xc) < y_star) under q(f(Xc))."""
    qf = model(Xc)
    m = qf.mean
    s = qf.variance.clamp_min(1e-12).sqrt()
    z = (y_star - m) / s
    p_less = _normal_cdf(z)
    return p_less.mean().item(), p_less.min().item(), p_less.max().item()


def plot_mean_and_band(model, likelihood, x_train, y_train, inducing_points, title, y_star=None):
    test_x = torch.linspace(x_train.min().item(), x_train.max().item(), 400).double().unsqueeze(-1)
    with torch.no_grad():
        pred = predictive_distribution(model, likelihood, test_x, observation_noise=False)
        mean = pred.mean
        std = pred.variance.sqrt()

    plt.figure(figsize=(9, 4))
    plt.plot(x_train.squeeze(-1).numpy(), y_train.numpy(), "k*", markersize=4, label="train")

    # Inducing points: show x-locations (y=0 only for visualization)
    Z = inducing_points.detach().cpu()
    plt.plot(Z.squeeze(-1).numpy(), np.zeros(Z.shape[0]), "rx", markersize=6, mew=2, label="inducing (x-locs)")

    plt.plot(test_x.squeeze(-1).numpy(), mean.numpy(), label="mean")
    plt.fill_between(
        test_x.squeeze(-1).numpy(),
        (mean - 2 * std).numpy(),
        (mean + 2 * std).numpy(),
        alpha=0.3,
        label="±2 std (latent)",
    )
    if y_star is not None:
        plt.axhline(float(y_star), color="red", linestyle="--", label="y*")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_mean_difference(res_std, res_con, x_train):
    test_x = torch.linspace(x_train.min().item(), x_train.max().item(), 400).double().unsqueeze(-1)
    with torch.no_grad():
        mean_std = predictive_distribution(res_std["model"], res_std["likelihood"], test_x).mean
        mean_con = predictive_distribution(res_con["model"], res_con["likelihood"], test_x).mean
    diff = mean_con - mean_std

    plt.figure(figsize=(9, 3))
    plt.plot(test_x.squeeze(-1).numpy(), diff.numpy())
    plt.axhline(0.0, color="black", linestyle="--", linewidth=1)
    plt.title("Mean difference: (constraint) - (standard)")
    plt.tight_layout()
    plt.show()

    print("Mean absolute difference:", diff.abs().mean().item())
    print("Max absolute difference:", diff.abs().max().item())


def fit_with_fixed_inducing(
    train_X: torch.Tensor,
    train_Y: torch.Tensor,
    inducing_points: torch.Tensor,
    *,
    training_iter: int,
    lr: float,
    noise: float,
    fix_noise: bool,
    verbose: bool,
    # constraint
    y_star=None,
    epsilon: float = 0.05,
    constraint_weight: float = 1.0,
    Xc: torch.Tensor | None = None,
    seed_for_init: int = 0,
):
    """Train VFE sparse GP with optional step-constraint, but with FIXED inducing points and FIXED Xc."""
    device = train_X.device
    dtype = train_X.dtype

    # Make sure y is vector
    if train_Y.ndim == 2 and train_Y.shape[-1] == 1:
        y_vec = train_Y.squeeze(-1)
    else:
        y_vec = train_Y

    # Likelihood
    likelihood = gpytorch.likelihoods.GaussianLikelihood(noise_constraint=GreaterThan(1e-8))
    likelihood = likelihood.to(dtype=dtype, device=device)
    likelihood.noise = torch.as_tensor(noise, dtype=dtype, device=device)
    if fix_noise:
        likelihood.raw_noise.requires_grad_(False)

    # Seed so both models start from same init (given same inducing_points)
    torch.manual_seed(seed_for_init)

    # Model with fixed inducing points
    Z = inducing_points.to(device=device, dtype=dtype).contiguous()
    model = VFESparseGP(inducing_points=Z).to(dtype=dtype, device=device)

    N = train_X.shape[0]

    if y_star is None:
        mll = VariationalELBO(likelihood, model, num_data=N)
    else:
        if Xc is None:
            raise ValueError("For constrained training you must pass Xc explicitly (fixed across runs).")
        y_star_t = torch.as_tensor(y_star, device=device, dtype=dtype)
        mll = StepConstraintVariationalELBO(
            likelihood=likelihood,
            model=model,
            num_data=N,
            Xc=Xc.to(device=device, dtype=dtype),
            y_star=y_star_t,
            epsilon=epsilon,
            constraint_weight=constraint_weight,
        )

    losses = train_model_ADAM(
        model=model,
        mll=mll,
        train_x=train_X,
        train_y=y_vec,
        training_iter=training_iter,
        likelihood=likelihood,
        lr=lr,
        verbose=verbose,
    )

    return {
        "model": model,
        "likelihood": likelihood,
        "mll": mll,
        "losses": losses,
        "inducing_points": Z.detach().clone(),
    }


def main():
    # ----------------------------
    # Global config (comparable)
    # ----------------------------
    torch.manual_seed(0)

    x_train, y_train = load_snelson()

    # Training hyperparams
    M = 10
    training_iter = 400
    lr = 5e-3
    noise = 1e-1
    fix_noise = False
    verbose = True

    # Constraint config
    n_c = 100
    constraint_sampling = "sobol"  # "rand" or "sobol"
    Xc_seed = 123  # fixed seed so Xc is identical across experiments
    y_star = -0.5   # choose a "high" threshold to see the effect for f(Xc) < y*
    epsilon = 0.05
    constraint_weight = 100000000.0

    # ----------------------------
    # 1) Fix inducing points ONCE (shared by both models)
    # ----------------------------
    # Use a separate generator so this is deterministic and independent of other randomness
    gZ = torch.Generator(device=x_train.device)
    gZ.manual_seed(999)
    perm = torch.randperm(x_train.shape[0], generator=gZ, device=x_train.device)
    inducing_points = x_train[perm[:M]].contiguous()

    # ----------------------------
    # 2) Fix Xc ONCE in [0,1]^d (shared by both models)
    # ----------------------------
    d = x_train.shape[1]
    Xc = sample_unit_box(n=n_c, d=d, method=constraint_sampling, seed=Xc_seed, dtype=x_train.dtype, device=x_train.device)

    # ----------------------------
    # 3) Fit baseline (no constraint) and constrained model
    #    Use same seed_for_init so they start "equivalently" given same inducing points
    # ----------------------------
    init_seed = 2024

    res_std = fit_with_fixed_inducing(
        train_X=x_train,
        train_Y=y_train,
        inducing_points=inducing_points,
        training_iter=training_iter,
        lr=lr,
        noise=noise,
        fix_noise=fix_noise,
        verbose=verbose,
        y_star=None,
        seed_for_init=init_seed,
    )

    res_con = fit_with_fixed_inducing(
        train_X=x_train,
        train_Y=y_train,
        inducing_points=inducing_points,   # SAME inducing points
        training_iter=training_iter,
        lr=lr,
        noise=noise,
        fix_noise=fix_noise,
        verbose=verbose,
        y_star=y_star,
        epsilon=epsilon,
        constraint_weight=constraint_weight,
        Xc=Xc,                              # SAME Xc (in [0,1]^d)
        seed_for_init=init_seed,            # SAME init seed
    )

    # ----------------------------
    # 4) Plots (show inducing points)
    # ----------------------------
    plot_mean_and_band(
        res_std["model"], res_std["likelihood"], x_train, y_train,
        inducing_points=res_std["inducing_points"],
        title="Standard ELBO (with noise)",
        y_star=y_star,
    )
    plot_mean_and_band(
        res_con["model"], res_con["likelihood"], x_train, y_train,
        inducing_points=res_con["inducing_points"],
        title=f"Step-constraint ELBO (f(Xc)<y*, y*={y_star}, Xc in [0,1]^d, {constraint_sampling})",
        y_star=y_star,
    )

    plot_mean_difference(res_std, res_con, x_train)

    # ----------------------------
    # 5) Quantitative check on the SAME Xc
    # ----------------------------
    y_star_t = torch.tensor(y_star, dtype=x_train.dtype, device=x_train.device)
    p_std_mean, p_std_min, p_std_max = prob_f_below_y_star(res_std["model"], Xc, y_star_t)
    p_con_mean, p_con_min, p_con_max = prob_f_below_y_star(res_con["model"], Xc, y_star_t)

    print("\nP(f(Xc) < y*) under q(f)  [SAME Xc in [0,1]^d]:")
    print(f"  Standard:   mean={p_std_mean:.3f}, min={p_std_min:.3f}, max={p_std_max:.3f}")
    print(f"  Constraint: mean={p_con_mean:.3f}, min={p_con_min:.3f}, max={p_con_max:.3f}")

    print("\nLearned noise:")
    print("  Standard noise:", res_std["likelihood"].noise.item())
    print("  Constraint noise:", res_con["likelihood"].noise.item())
    


if __name__ == "__main__":
    main()