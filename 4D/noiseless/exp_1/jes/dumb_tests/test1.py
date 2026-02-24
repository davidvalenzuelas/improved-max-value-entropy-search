import torch
from vfe_sparse_gp import fit_model_vfe_sparse, as_botorch_model

def main():
    torch.manual_seed(0)
    torch.set_default_dtype(torch.double)

    d = 4
    n = 20
    train_X = torch.rand(n, d)
    train_Y = torch.sin(6 * train_X[:, :1])  # (n,1)

    y_star = train_Y.max().item()

    model, lik = fit_model_vfe_sparse(
        train_X, train_Y,
        y_star=y_star,
        num_constraint_points=100,
        tau=0.05,
        mc_samples=16,
        constraint_weight=10.0,
        training_iter=200,
        verbose=False
    )
    model.likelihood = lik

    m = as_botorch_model(
        model,
        y_star=y_star,
        num_constraint_points=100,
        tau=0.05,
        mc_samples=16,
        constraint_weight=10.0,
    )

    # Simula fantasías con batch y q
    X_f = torch.rand(7, 3, d)      # [batch=7, q=3, d]
    Y_f = torch.rand(7, 3, 1)      # [batch=7, q=3, 1]

    mf = m.condition_on_observations(X_f, Y_f)
    Xtest = torch.rand(5, d)
    post = mf.posterior(Xtest)

    print("OK. posterior mean shape:", post.mean.shape)
    print("OK. y_star base:", m.y_star, "y_star fantasy:", mf.y_star)

if __name__ == "__main__":
    main()