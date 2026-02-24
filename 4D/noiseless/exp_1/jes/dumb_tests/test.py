import pytest
import torch
import numpy as np
from botorch.optim import optimize_acqf
from botorch.acquisition.joint_entropy_search import qJointEntropySearch
from botorch.utils.sampling import draw_sobol_samples
from vfe_sparse_gp import fit_model_vfe_sparse, as_botorch_model
from synthetic_problem import Synthetic_problem  # Asegúrate de tener la implementación adecuada de esta clase


# Simularemos un problema objetivo sencillo con un mínimo en [0, 1]^d
def toy_objective(X: torch.Tensor) -> torch.Tensor:
    """ Toy objective: simple quadratic bowl, max at (0.25, 0.80) """
    c = torch.tensor([0.25, 0.80], dtype=X.dtype, device=X.device)
    return -((X - c) ** 2).sum(dim=-1, keepdim=True)

from botorch.utils.sampling import draw_sobol_samples
import torch

def get_optimal_samples_grid_mc(model, bounds: torch.Tensor, num_optima: int, num_grid: int = 2048):
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
        idx = samples.argmax(dim=-1)                      # (num_optima,)
        optimal_inputs = X[idx, :]                        # (num_optima, d)
        optimal_outputs = samples[torch.arange(num_optima, device=X.device), idx].unsqueeze(-1)  # (num_optima, 1)

    return optimal_inputs, optimal_outputs

def test_my_loop():
    # -----------------------------
    # 1. Configuración inicial
    # -----------------------------
    device = torch.device("cpu")
    dtype = torch.double
    bounds = torch.tensor([[0.0, 0.0], [1.0, 1.0]], dtype=dtype, device=device)
    
    # Número de puntos iniciales y de iteraciones de optimización
    n_init = 6
    n_iters = 10
    
    # Inicializamos X_train con puntos aleatorios
    X_train = draw_sobol_samples(bounds=bounds, n=n_init, q=1).squeeze(1).to(dtype=dtype, device=device)
    Y_train = toy_objective(X_train).to(dtype=dtype, device=device)  # Evaluamos la función objetivo en esos puntos
    
    # El valor óptimo para el toy objective es 0 (en el centro del cuenco)
    y_star = torch.tensor(0.0, dtype=dtype, device=device)

    # Parámetros para el VFE Sparse GP y optimización
    M = 16
    training_iter = 60
    lr = 0.02
    noise_eps = 1e-6
    tau = 0.05
    mc_samples = 16
    constraint_weight = 5.0
    num_constraint_points = 32
    
    # -----------------------------
    # 2. Iteración de Optimización Bayesiana
    # -----------------------------
    best_y_hist = []
    
    for it in range(n_iters):
        print(f"=== Iteración {it+1}/{n_iters} ===")

        # Entrenamos el modelo GP con los datos actuales
        model, likelihood = fit_model_vfe_sparse(
            train_X=X_train,
            train_Y=Y_train,
            M=M,
            training_iter=training_iter,
            lr=lr,
            noise_eps=noise_eps,
            verbose=False,
            y_star=y_star,
            Xc=None,
            num_constraint_points=num_constraint_points,
            tau=tau,
            mc_samples=mc_samples,
            constraint_weight=constraint_weight
        )

        model.likelihood = likelihood
        
        
        # Verificamos que el posterior sobre los puntos entrenados es finito
        with torch.no_grad():
            post_train = model(X_train)
            assert torch.isfinite(post_train.mean).all(), "Posterior mean is not finite"
            assert torch.isfinite(post_train.variance).all(), "Posterior variance is not finite"

        # Generamos puntos óptimos para la búsqueda de adquisición JES
        optimal_inputs, optimal_outputs = get_optimal_samples_grid_mc(model, bounds, num_optima=10)

        # Envolvemos el modelo para usarlo con BoTorch
        model_botorch = as_botorch_model(
            model,
            y_star=y_star,
            Xc=None,
            num_constraint_points=num_constraint_points,
            tau=tau,
            mc_samples=mc_samples,
            constraint_weight=constraint_weight,
        )

        # Construimos la función de adquisición usando JES
        acq = qJointEntropySearch(
            model=model_botorch,
            optimal_inputs=optimal_inputs.double(),
            optimal_outputs=optimal_outputs.double(),
            estimation_type="LB",
            condition_noiseless=True,
        )

        # Optimizamos la función de adquisición
        candidate, acq_value = optimize_acqf(
            acq_function=acq,
            bounds=bounds,
            q=1,
            num_restarts=5,
            raw_samples=32,
        )

        candidate = candidate.view(1, 2).double()
        print("Candidato:", candidate.detach().cpu().numpy())
        print("Valor de adquisición:", float(acq_value))

        # Evaluamos la función objetivo en el nuevo punto candidato
        y_new = toy_objective(candidate).double()  # (1,1)
        X_train = torch.cat([X_train, candidate], dim=0)
        Y_train = torch.cat([Y_train, y_new], dim=0)

        # Guardamos el mejor valor observado
        best_y = float(Y_train.max().detach().cpu().numpy())
        best_y_hist.append(best_y)
        print("Mejor valor hasta ahora:", best_y)

        # Verificamos que el valor observado esté mejorando (al menos no empeorando)
        if it > 0 and best_y_hist[it] < best_y_hist[it - 1]:
            raise AssertionError(f"El valor observado ha empeorado en la iteración {it+1}.")

    print("Mejor valor encontrado:", best_y_hist[-1])
    assert best_y_hist[-1] >= 0, "El mejor valor encontrado debe ser mayor o igual a 0"

    # Verificamos que el mejor valor esté en línea con el óptimo (el máximo de la función objetivo)
    assert torch.isclose(torch.tensor(best_y_hist[-1]), torch.tensor(0.0), atol=1e-1), \
        f"El mejor valor encontrado no está cerca del óptimo. Mejor: {best_y_hist[-1]}"

    print("El test ha pasado correctamente.")
