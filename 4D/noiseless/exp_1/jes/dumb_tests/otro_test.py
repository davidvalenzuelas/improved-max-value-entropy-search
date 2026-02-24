import torch
from vfe_sparse_gp import fit_model_vfe_sparse


def posterior_stats(model, Xt):
    """Devuelve media y var marginal aproximadas en Xt."""
    model.eval()
    with torch.no_grad():
        mvn = model(Xt)
        mu = mvn.mean
        var = mvn.variance.clamp_min(1e-12)
    return mu, var


def prob_exceed(mu, var, y_star: float):
    """P(f > y_star) asumiendo marginal Normal(mu, var)."""
    std = var.sqrt()
    z = (y_star - mu) / std  # queremos P(f>y*) = 1 - Phi((y*-mu)/std)
    normal = torch.distributions.Normal(
        loc=torch.tensor(0.0, dtype=mu.dtype, device=mu.device),
        scale=torch.tensor(1.0, dtype=mu.dtype, device=mu.device),
    )
    return (1.0 - normal.cdf(z)).clamp(0.0, 1.0)


def summarize(model, Xt, y_star: float):
    mu, var = posterior_stats(model, Xt)
    pexc = prob_exceed(mu, var, y_star)
    return {
        "mu_mean": float(mu.mean()),
        "mu_max": float(mu.max()),
        "pexc_mean": float(pexc.mean()),
        "pexc_max": float(pexc.max()),
    }


def main():
    torch.manual_seed(0)
    torch.set_default_dtype(torch.double)

    d = 1
    n = 20
    train_X = torch.rand(n, d)
    train_Y = torch.sin(6 * train_X)

    # Entrena normal
    model0, lik0 = fit_model_vfe_sparse(train_X, train_Y, training_iter=300, verbose=False)
    model0.likelihood = lik0

    # Entrena condicionado (moderado)
    y_star = train_Y.max().item()
    model1, lik1 = fit_model_vfe_sparse(
        train_X, train_Y,
        y_star=y_star,
        num_constraint_points=100,
        constraint_weight=5.0,
        tau=0.05,
        mc_samples=16,
        training_iter=300,
        verbose=False
    )
    model1.likelihood = lik1

    # Entrena condicionado fuerte
    model2, lik2 = fit_model_vfe_sparse(
        train_X, train_Y,
        y_star=y_star,
        num_constraint_points=100,
        constraint_weight=20.0,
        tau=0.05,
        mc_samples=16,
        training_iter=300,
        verbose=False
    )
    model2.likelihood = lik2

    # Puntos de test
    Xt = torch.rand(5000, d)

    s0 = summarize(model0, Xt, y_star)
    s1 = summarize(model1, Xt, y_star)
    s2 = summarize(model2, Xt, y_star)

    print("\n===== TEST PASO 5 (métricas) =====")
    print(f"y_star usado: {y_star:.6f}\n")

    def pr(name, s):
        print(f"[{name}]")
        print(f"  mean(mu):   {s['mu_mean']:.6f}")
        print(f"  max(mu):    {s['mu_max']:.6f}")
        print(f"  mean P(f>y*): {s['pexc_mean']:.6f}")
        print(f"  max  P(f>y*): {s['pexc_max']:.6f}\n")

    pr("normal", s0)
    pr("cond (w=5)", s1)
    pr("cond (w=20)", s2)

    # Criterios esperables:
    # - mean P(f>y*) debería bajar al aumentar weight
    # - max P(f>y*) a veces no baja mucho si hay puntos con mucha incertidumbre, pero suele bajar
    if s2["pexc_mean"] < s0["pexc_mean"]:
        print("✅ OK: la restricción reduce P(f>y*) en promedio.")
    else:
        print("⚠️  OJO: no bajó P(f>y*) en promedio. Ajusta tau/weight o sube iteraciones.")


if __name__ == "__main__":
    main()