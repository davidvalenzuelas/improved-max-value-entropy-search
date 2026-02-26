import io
import numpy as np
import requests
import torch
import matplotlib.pyplot as plt

# Cambia esto al nombre real de tu fichero .py (sin .py)
from vfe_sparse_gp import fit_vfe_sparse_gp, predictive_distribution


def load_snelson():
    url = "http://arantxa.ii.uam.es/~dhernan/MLAS2023/EdSnelson.npy"
    r = requests.get(url, timeout=30)
    r.raise_for_status()
    data = np.load(io.BytesIO(r.content), allow_pickle=False)

    x_train = torch.from_numpy(data[0]).double()
    y_train = torch.from_numpy(data[1]).double()
    return x_train, y_train


def plot_predictive(model, likelihood, x_train, y_train, title):
    test_x = torch.linspace(-3.0, 9.0, 400).double()

    with torch.no_grad():
        pred = predictive_distribution(model, likelihood, test_x, observation_noise=False)
        mean = pred.mean
        std = pred.variance.sqrt()

    x_tr = x_train.cpu().numpy()
    y_tr = y_train.cpu().numpy()
    x_te = test_x.cpu().numpy()
    m = mean.cpu().numpy()
    s = std.cpu().numpy()

    plt.figure(figsize=(9, 4))
    plt.plot(x_tr, y_tr, "k*", markersize=4, label="train")
    plt.plot(x_te, m, label="predictive mean")
    plt.fill_between(x_te, m - 2*s, m + 2*s, alpha=0.3, label="±2 std (latent)")

    # mostrar inducing points (solo en x, y abajo)
    Z = model.variational_strategy.inducing_points.detach().cpu().numpy()
    plt.plot(Z, np.min(y_tr) * np.ones_like(Z), "ro", markersize=4, label="inducing points")

    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.show()


def main():
    torch.manual_seed(0)

    x_train, y_train = load_snelson()

    # 1) VFE estándar (sin constraint)
    res_std = fit_vfe_sparse_gp(
        train_X=x_train,
        train_Y=y_train,
        M=10,                 # en el notebook típico usan 10 inducing points en Snelson
        training_iter=500,
        lr=5e-3,              # más estable en "noiseless"
        noise=1e-6,
        fix_noise=True,
        verbose=True,
        y_star=None,          # IMPORTANT: sin constraint
    )

    plot_predictive(res_std.model, res_std.likelihood, x_train, y_train,
                    title="VFE Sparse GP (noiseless) - Standard ELBO")

    # 2) VFE con constraint (prueba rápida)
    # Elige un y_star razonable (ejemplo: 0.0). Ajusta según tu experimento.
    res_con = fit_vfe_sparse_gp(
        train_X=x_train,
        train_Y=y_train,
        M=10,
        training_iter=500,
        lr=5e-3,
        noise=1e-6,
        fix_noise=True,
        verbose=True,
        y_star=0.0,                 # activa constraint
        epsilon=0.05,
        constraint_weight=1.0,
        num_constraint_points=100,
        constraint_sampling="sobol" # o "rand"
    )

    plot_predictive(res_con.model, res_con.likelihood, x_train, y_train,
                    title="VFE Sparse GP (noiseless) - Step-constraint ELBO")


if __name__ == "__main__":
    main()