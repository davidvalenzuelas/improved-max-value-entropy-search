import torch
from vfe_sparse_gp import fit_model_vfe_sparse, as_botorch_model

def main():
    torch.manual_seed(0)
    torch.set_default_dtype(torch.double)

    d = 2
    train_X = torch.rand(30, d)
    train_Y = (torch.sin(6 * train_X[:, :1]) + torch.cos(4 * train_X[:, 1:2]))

    y_star = train_Y.max().item()

    model, lik = fit_model_vfe_sparse(
        train_X, train_Y,
        y_star=y_star,
        num_constraint_points=100,
        constraint_weight=10.0,
        tau=0.05,
        mc_samples=16,
        training_iter=300,
        verbose=False
    )
    model.likelihood = lik

    m = as_botorch_model(
        model,
        y_star=y_star,
        num_constraint_points=100,
        constraint_weight=10.0,
        tau=0.05,
        mc_samples=16,
    )

    Xtest = torch.rand(5, d)
    post = m.posterior(Xtest)
    print("posterior mean shape:", post.mean.shape)

    Xnew = torch.rand(3, d)
    Ynew = (torch.sin(6 * Xnew[:, :1]) + torch.cos(4 * Xnew[:, 1:2]))

    mf = m.condition_on_observations(Xnew, Ynew)
    post2 = mf.posterior(Xtest)
    print("fantasy posterior mean shape:", post2.mean.shape)

    print("✅ Wrapper BoTorch OK")

if __name__ == "__main__":
    main()