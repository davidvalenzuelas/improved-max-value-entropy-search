import torch

# IMPORTA desde tu fichero real
from vfe_sparse_gp import fit_model_vfe_sparse


def count_violations(model, Xt, y_star: float) -> float:
    """Porcentaje de puntos con media posterior > y_star."""
    model.eval()
    with torch.no_grad():
        mvn = model(Xt)
        mu = mvn.mean
        return (mu > y_star).double().mean().item()


def main():
    torch.manual_seed(0)
    torch.set_default_dtype(torch.double)

    # 1) Datos simples (d=1) en [0,1]
    d = 1
    n = 20
    train_X = torch.rand(n, d)
    train_Y = torch.sin(6 * train_X)  # (n, 1)

    # (opcional) añade un pelín de ruido para que sea más realista
    # train_Y = train_Y + 0.01 * torch.randn_like(train_Y)

    # 2) Entrenamiento normal
    model0, lik0 = fit_model_vfe_sparse(
        train_X, train_Y,
        training_iter=200,
        verbose=False
    )
    model0.likelihood = lik0  # (no es estrictamente necesario para este test)

    # 3) Entrenamiento condicionado
    y_star = train_Y.max().item()  # solo para probar que el mecanismo funciona

    model1, lik1 = fit_model_vfe_sparse(
        train_X, train_Y,
        y_star=y_star,
        num_constraint_points=100,
        constraint_weight=5.0,
        tau=0.05,
        mc_samples=16,
        training_iter=200,
        verbose=False
    )
    model1.likelihood = lik1

    # 4) Check rápido: ¿se puede evaluar el modelo sin NaNs?
    Xt = torch.rand(2000, d)
    with torch.no_grad():
        mu0 = model0(Xt).mean
        mu1 = model1(Xt).mean

    if torch.isnan(mu0).any() or torch.isnan(mu1).any():
        raise RuntimeError("Hay NaNs en la media posterior. Revisa tau/eps/constraint_weight.")

    # 5) Métrica: porcentaje de "violaciones" (mu > y_star)
    v0 = count_violations(model0, Xt, y_star)
    v1 = count_violations(model1, Xt, y_star)

    print("\n===== TEST PASO 4 (y_star) =====")
    print(f"y_star usado: {y_star:.6f}")
    print(f"Violations (normal):       {v0:.4f}")
    print(f"Violations (conditioned):  {v1:.4f}")

    # 6) Test “fuerte” para estar seguros de que el término hace algo
    model2, lik2 = fit_model_vfe_sparse(
        train_X, train_Y,
        y_star=y_star,
        num_constraint_points=100,
        constraint_weight=20.0,
        tau=0.05,
        mc_samples=16,
        training_iter=200,
        verbose=False
    )
    Xt2 = torch.rand(2000, d)
    v2 = count_violations(model2, Xt2, y_star)

    print(f"Violations (strong cond):  {v2:.4f}")

    # 7) Criterio de “ok”
    # No siempre baja con weight=5.0, pero con weight=20 normalmente debería notarse.
    if v2 < v0:
        print("✅ OK: con constraint_weight alto, el modelo reduce violaciones. Paso 4 funciona.")
    else:
        print("⚠️  OJO: no veo reducción clara con weight alto. Puede ser tau/eps o la escala de la función.")
        print("   Prueba: tau=0.1 o constraint_weight=50, o más iteraciones. Si aún no cambia, revisamos.")

    print("Test finished\n")


if __name__ == "__main__":
    main()