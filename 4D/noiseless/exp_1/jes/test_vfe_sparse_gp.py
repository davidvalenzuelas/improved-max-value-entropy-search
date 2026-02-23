import torch
from vfe_sparse_gp import fit_model_vfe_sparse

torch.manual_seed(0)

# Datos falsos simples
N, d = 20, 4
X = torch.rand(N, d, dtype=torch.double)
y = torch.sin(6 * X[:, :1]).sum(dim=1, keepdim=True)  # (N,1)

model, likelihood = fit_model_vfe_sparse(
    train_X=X,
    train_Y=y,
    state_dict=None,
    M=16,
    training_iter=200,
    lr=0.01,
    noise_eps=1e-6,
    verbose=False,
)

model.eval(); likelihood.eval()

Xt = torch.rand(8, d, dtype=torch.double)
with torch.no_grad():
    post = model(Xt)
    print("mean finite:", torch.isfinite(post.mean).all().item())
    print("var  finite:", torch.isfinite(post.variance).all().item())
    print("var  min   :", float(post.variance.min()))