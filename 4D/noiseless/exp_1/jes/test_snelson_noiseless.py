import io
import numpy as np
import requests
import torch
import matplotlib.pyplot as plt

from vfe_sparse_gp import fit_vfe_sparse_gp, predictive_distribution, _normal_cdf


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


@torch.no_grad()
def prob_f_below_y_star(model, Xc, y_star):
    """Compute average P(f(Xc) < y_star) under q(f(Xc))."""
    qf = model(Xc)
    m = qf.mean
    s = qf.variance.clamp_min(1e-12).sqrt()
    z = (y_star - m) / s
    p_less = _normal_cdf(z)
    return p_less.mean().item(), p_less.min().item(), p_less.max().item()


def plot_mean_and_band(model, likelihood, x_train, y_train, title, y_star=None):
    test_x = torch.linspace(x_train.min().item(), x_train.max().item(), 400).double().unsqueeze(-1)
    with torch.no_grad():
        pred = predictive_distribution(model, likelihood, test_x, observation_noise=False)
        mean = pred.mean
        std = pred.variance.sqrt()

    plt.figure(figsize=(9, 4))
    plt.plot(x_train.squeeze(-1).numpy(), y_train.numpy(), "k*", markersize=4, label="train")
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
        mean_std = predictive_distribution(res_std.model, res_std.likelihood, test_x).mean
        mean_con = predictive_distribution(res_con.model, res_con.likelihood, test_x).mean
    diff = mean_con - mean_std

    plt.figure(figsize=(9, 3))
    plt.plot(test_x.squeeze(-1).numpy(), diff.numpy())
    plt.axhline(0.0, color="black", linestyle="--", linewidth=1)
    plt.title("Mean difference: (constraint) - (standard)")
    plt.tight_layout()
    plt.show()

    print("Mean absolute difference:", diff.abs().mean().item())
    print("Max absolute difference:", diff.abs().max().item())


def main():
    torch.manual_seed(0)

    x_train, y_train = load_snelson()

    # ---- Build Xc over the REAL input domain ----
    # Option 1: Sobol points in [min,max] (more "uniform")
    n_c = 200
    x_min = x_train.min(dim=0).values
    x_max = x_train.max(dim=0).values
    sob = torch.quasirandom.SobolEngine(dimension=x_train.shape[1], scramble=True)
    U = sob.draw(n_c).double()
    Xc = x_min + (x_max - x_min) * U  # scale from [0,1]^d to [min,max]^d

    y_star = -1.5  # choose something that creates conflict (try also 0.0, -1.0)

    # ---- Standard model (NO constraint) ----
    res_std = fit_vfe_sparse_gp(
        train_X=x_train,
        train_Y=y_train,
        M=10,
        training_iter=400,
        lr=5e-3,
        noise=1e-2,
        fix_noise=False,
        verbose=False,
        y_star=None,  # IMPORTANT
    )

    # ---- Constraint model ----
    res_con = fit_vfe_sparse_gp(
        train_X=x_train,
        train_Y=y_train,
        M=10,
        training_iter=400,
        lr=5e-3,
        noise=1e-2,
        fix_noise=False,
        verbose=False,
        y_star=y_star,
        epsilon=0.05,
        constraint_weight=50.0,  # push harder to make it visible
        num_constraint_points=n_c,
        constraint_sampling="sobol",
        Xc=Xc,  # IMPORTANT: apply constraint across the domain
    )

    # ---- Plots ----
    plot_mean_and_band(res_std.model, res_std.likelihood, x_train, y_train,
                       title="Standard ELBO (with noise)", y_star=y_star)
    plot_mean_and_band(res_con.model, res_con.likelihood, x_train, y_train,
                       title=f"Step-constraint ELBO (y*={y_star})", y_star=y_star)

    # Difference plot (this makes the effect obvious)
    plot_mean_difference(res_std, res_con, x_train)

    # ---- Quantitative "constraint satisfaction" ----
    p_std_mean, p_std_min, p_std_max = prob_f_below_y_star(res_std.model, Xc, torch.tensor(y_star).double())
    p_con_mean, p_con_min, p_con_max = prob_f_below_y_star(res_con.model, Xc, torch.tensor(y_star).double())

    print("\nP(f(Xc) < y*) under q(f):")
    print(f"  Standard: mean={p_std_mean:.3f}, min={p_std_min:.3f}, max={p_std_max:.3f}")
    print(f"  Constraint: mean={p_con_mean:.3f}, min={p_con_min:.3f}, max={p_con_max:.3f}")

    print("\nLearned noise:")
    print("  Standard noise:", res_std.likelihood.noise.item())
    print("  Constraint noise:", res_con.likelihood.noise.item())


if __name__ == "__main__":
    main()