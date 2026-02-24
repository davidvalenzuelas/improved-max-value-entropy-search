#!/usr/bin/env python3
# coding: utf-8

import torch
import gpytorch

from gpytorch.models import ApproximateGP
from gpytorch.variational import CholeskyVariationalDistribution, UnwhitenedVariationalStrategy
from gpytorch.mlls import VariationalELBO

from botorch.models.model import Model
from botorch.posteriors.gpytorch import GPyTorchPosterior


# Defines our approximate GP method based on the VFE approach
class VFESparseGP(ApproximateGP):
    
    def __init__(self, inducing_points: torch.Tensor):
        # Zero mean and RBF kernel for covariances
        mean_module = gpytorch.means.ZeroMean()
        covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())
        
        # Variational approximate distribution q
        # We use a cholesky factorization to represent its parameters
        variational_distribution = CholeskyVariationalDistribution(
            inducing_points.size(0),
            mean_init_std=0.0,
        )
        
        # Smooth initialization of the variational distribution with this prior-like gaussian
        init_dist = gpytorch.distributions.MultivariateNormal(
            torch.zeros(inducing_points.size(0), dtype=inducing_points.dtype, device=inducing_points.device),
            covar_module(inducing_points) * 1e-5,
        )
        variational_distribution.initialize_variational_distribution(init_dist)
        
        # Variational strategy, defining how the inducing points ares used to approximate the full GP
        variational_strategy = UnwhitenedVariationalStrategy(
            self,
            inducing_points,
            variational_distribution,
            learn_inducing_locations=True, # this makes inducing points trainable
        )
        
        # Avoids internal random reinitialization of the variational parameters, because we have already
        # initialized them with the prior-like distribution above
        variational_strategy.variational_params_initialized = torch.tensor(1)
        
        # Initializes base ApproximateGP class
        super().__init__(variational_strategy)
        
        # Stores mean and covariances
        self.mean_module = mean_module
        self.covar_module = covar_module
        
    # Defines the GP prior for a given input x
    def forward(self, x: torch.Tensor):
        # Computes mean vector and covariance matrix for the input x
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        
        # Returns a gaussian distribution
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)
    
    
# Training function using ADAM optimizer
def train_model_ADAM(model: torch.nn.Module, mll: torch.nn.Module, train_x: torch.Tensor, train_y: torch.Tensor,
                    training_iter: int = 500, likelihood: torch.nn.Module | None = None, lr: float = 0.01,
                    verbose: bool = True,):
    """ Trains the variational GP model by maximizing the elbo using the ADAM optimizer"""
    
    # Sets the model and likelihood in training mode
    model.train()
    if likelihood is not None:
        likelihood.train()
        
    # Determines which parameters to optimize
    if likelihood is None:
        parameters = model.parameters()
    else:
        parameters = list(model.parameters()) + list(likelihood.parameters())
        
    # Defines ADAM optimizer
    optimizer = torch.optim.Adam(parameters, lr=lr)
    
    # Closure function to compute loss and gradients
    def closure():
        optimizer.zero_grad()
        output = model(train_x)
        # we maximize ELBO, so we minimize -ELBO
        loss = -mll(output, train_y)
        loss.backward()
        return loss
    
    losses = []
    # This is the main training loop, we call the closure function here to compute loss and gradients, and
    # then we use the optimizer to update the parameters
    for i in range(training_iter):
        loss = closure()
        # The closure is called explicitly and not passed to optimizer.step(), because Adam does not require it
        # Updates parameters
        optimizer.step()
        
        losses.append(loss.item())
        if verbose and ((i + 1) % 50 == 0 or i == 0):
            print(f"Iter {i+1}/{training_iter} - Loss: {loss.item():.6f}")
            
    # Sets the model and likelihood in evaluation mode after training        
    model.eval()
    if likelihood is not None:
        likelihood.eval()
        
    return losses


class ConstrainedVariationalELBO(VariationalELBO):
    def __init__(
        self,
        likelihood,
        model,
        num_data: int,
        Xc: torch.Tensor,
        y_star: torch.Tensor,
        tau: float = 0.02,
        mc_samples: int = 32,
        constraint_weight: float = 1.0,
        eps: float = 1e-12,
    ):
        super().__init__(likelihood=likelihood, model=model, num_data=num_data)
        self.register_buffer("Xc", Xc)
        self.register_buffer("y_star", y_star.view(1))
        self.tau = float(tau)
        self.mc_samples = int(mc_samples)
        self.constraint_weight = float(constraint_weight)
        self.eps = float(eps)

        self._std_normal = torch.distributions.Normal(
            loc=torch.tensor(0.0, dtype=Xc.dtype, device=Xc.device),
            scale=torch.tensor(1.0, dtype=Xc.dtype, device=Xc.device),
        )

    def _constraint_term(self) -> torch.Tensor:
        mvn_c = self.model(self.Xc)  # q(f(Xc))
        fs = mvn_c.rsample(sample_shape=torch.Size([self.mc_samples]))  # (S, K)

        z = (self.y_star - fs) / self.tau
        phi = self._std_normal.cdf(z).clamp_min(self.eps)
        log_phi = torch.log(phi)  # (S, K)

        return log_phi.mean(dim=0).sum()  # scalar

    def forward(self, output, target, **kwargs):
        base = super().forward(output, target, **kwargs)
        extra = self._constraint_term()
        return base + self.constraint_weight * extra
    
    
# Fits the VFE sparse GP model to the training data, and returns the trained model and likelihood
def fit_model_vfe_sparse(train_X: torch.Tensor, train_Y: torch.Tensor, state_dict: dict | None = None,
    M: int = 64, training_iter: int = 500, lr: float = 0.01,
    noise_eps: float = 1e-6,  # tiny fixed noise for numerical stability
    verbose: bool = True, y_star=None, Xc: torch.Tensor | None = None,num_constraint_points: int = 100, tau: float = 0.02, 
    mc_samples: int = 32, constraint_weight: float = 1.0,):
    """Fits a variational sparse GP model with tiny observation noise """
    
    # Uses double precision for better numeical stability, important for Cholesk decompositions
    train_X = train_X.double()
    train_Y = train_Y.double()
    
    # Target vector should be 1D
    y_vec = train_Y.squeeze(-1) if train_Y.ndim == 2 and train_Y.shape[1] == 1 else train_Y
    
    # Noiseless likelihood
    # Fixes the noise to a small eps to approximate noiseless observations
    fixed_noise = torch.full_like(y_vec, noise_eps)
    likelihood = gpytorch.likelihoods.FixedNoiseGaussianLikelihood(noise=fixed_noise)
    
    # Selects inducing points
    N = train_X.shape[0]

    if state_dict is not None:
        # Get previous inducing point count from checkpoint
        prev_inducing = state_dict["model"]["variational_strategy.inducing_points"]
        M_eff = prev_inducing.shape[0]
        M_eff = min(M_eff, N)  # safety
    else:
        M_eff = min(M, N)
    perm = torch.randperm(N, device=train_X.device)
    inducing_points = train_X[perm[:M_eff]].contiguous()
    
    # Instantiate variational sparse GP model
    model = VFESparseGP(inducing_points=inducing_points)
    
    # Loads state dict if provided, it allows resuming training from checkpoints
    if state_dict is not None:
        model.load_state_dict(state_dict["model"])
        try:
            likelihood.load_state_dict(state_dict["likelihood"])
        except Exception:
            pass  # allows loading checkpoints created with GaussianLikelihood
        
    # Defines the variational ELBO loss, our objective to maximize during training
    # --- build the MLL (ELBO) ---
    if y_star is not None:
        # Option A: constraint points in [0,1]^d
        d = train_X.shape[-1]
        
        if Xc is None:
            Xc = torch.rand(num_constraint_points, d, device=train_X.device, dtype=train_X.dtype)
        else:
            Xc = Xc.to(device=train_X.device, dtype=train_X.dtype)
            
        y_star_t = torch.as_tensor(y_star, device=train_X.device, dtype=train_X.dtype)
        
        mll = ConstrainedVariationalELBO(
            likelihood=likelihood,
            model=model,
            num_data=train_X.size(0),
            Xc=Xc,
            y_star=y_star_t,
            tau=tau,
            mc_samples=mc_samples,
            constraint_weight=constraint_weight,
        )
    else:
        mll = VariationalELBO(likelihood, model, num_data=train_X.size(0))
    
    # Training optimizes model parameters using ADAM to minimize -ELBO
    # ELBO already includes likelihood
    train_model_ADAM(model=model, mll=mll, train_x=train_X, train_y=y_vec, training_iter=training_iter,
        likelihood=None, lr=lr, verbose=verbose,)
    
    # saves training data on the model for later conditioning
    model._train_X = train_X.detach()
    # Stores as (N,1) always
    model._train_Y = y_vec.detach().view(-1, 1)
    
    return model, likelihood

# Packs model and likelihood state dicts into a single dictionary for checkpointing
def pack_state_dict(model, likelihood) -> dict:
    return {"model": model.state_dict(), "likelihood": likelihood.state_dict()}

# --- BoTorch wrapper for ApproximateGP (needed by JES) ---



class BoTorchVFEWrapper(Model):
    """
    Wrap a gpytorch ApproximateGP + likelihood to satisfy BoTorch's Model interface
    required by qJointEntropySearch (posterior + condition_on_observations).
    """
    def __init__(
        self,
        gp: torch.nn.Module,
        likelihood: gpytorch.likelihoods.Likelihood,
        cond_training_iter: int = 25,
        cond_lr: float = 0.01,
        cond_noise_eps: float = 1e-6,

        # --- NEW: constraint config ---
        y_star=None,
        Xc: torch.Tensor | None = None,
        num_constraint_points: int = 100,
        tau: float = 0.05,
        mc_samples: int = 16,
        constraint_weight: float = 1.0,
    ):
        super().__init__()
        self.gp = gp
        self.likelihood = likelihood
        self.cond_training_iter = cond_training_iter
        self.cond_lr = cond_lr
        self.cond_noise_eps = cond_noise_eps

        # --- NEW: store constraint config ---
        self.y_star = y_star
        self.Xc = Xc
        self.num_constraint_points = num_constraint_points
        self.tau = tau
        self.mc_samples = mc_samples
        self.constraint_weight = constraint_weight
        
    @property
    def num_outputs(self) -> int:
        return 1
    
    def posterior(
    self,
    X: torch.Tensor,
    output_indices=None,
    observation_noise: bool = False,
    posterior_transform=None,
    **kwargs,
    ):
        X = X.double()
        self.gp.eval()
        self.likelihood.eval()

        # 1) LIMPIAR CACHÉS para evitar choques cuando cambia el batch size
        if hasattr(self.gp, "_clear_cache"):
            self.gp._clear_cache()
        if hasattr(self.gp, "variational_strategy") and hasattr(self.gp.variational_strategy, "_clear_cache"):
            self.gp.variational_strategy._clear_cache()

        # 2) NO usar no_grad (optimize_acqf necesita gradientes w.r.t X)
        mvn = self.gp(X)
        if observation_noise:
            mvn = self.likelihood(mvn)

        post = GPyTorchPosterior(mvn)
        if posterior_transform is not None:
            return posterior_transform(post)
        return post

    def condition_on_observations(
        self,
        X: torch.Tensor,
        Y: torch.Tensor,
        noise: torch.Tensor | None = None,
        **kwargs,
    ) -> "BoTorchVFEWrapper":
        """
        For variational sparse GP there is no closed-form conditioning like exact GP.
        We do a warm-start refit on the augmented dataset (fast, few Adam steps).
        This is enough to make JES run.
        """
        X = X.double()
        Y = Y.double()
        if Y.ndim == 1:
            Y = Y.unsqueeze(-1)
        if Y.shape[-1] != 1:
            # JES in your setup is 1-output
            Y = Y[..., :1]

        # old training data (from gpytorch model)
        # Retrieve stored training data (we store it in fit_model_vfe_sparse)
        if not hasattr(self.gp, "_train_X") or not hasattr(self.gp, "_train_Y"):
            raise RuntimeError(
                "VFESparseGP has no stored training data. "
                "Make sure fit_model_vfe_sparse sets model._train_X and model._train_Y."
            )

        train_X_old = self.gp._train_X.detach().double()
        train_y_old = self.gp._train_Y.detach().double()
        if train_y_old.ndim == 1:
            train_y_old = train_y_old.unsqueeze(-1)

        # new points
        X_new = X.view(-1, train_X_old.shape[-1])
        Y_new = Y.view(-1, 1)

        train_X = torch.cat([train_X_old, X_new], dim=0)
        train_Y = torch.cat([train_y_old.view(-1, 1), Y_new], dim=0)

        # warm-start state
        state = {"model": self.gp.state_dict(), "likelihood": self.likelihood.state_dict()}

        # refit quickly (calls your own fit function already defined in this file)
        gp_new, lik_new = fit_model_vfe_sparse(
            train_X=train_X,
            train_Y=train_Y,
            state_dict=state,
            M=min(self.gp.variational_strategy.inducing_points.shape[0], train_X.shape[0]),
            training_iter=self.cond_training_iter,
            lr=self.cond_lr,
            noise_eps=self.cond_noise_eps,
            verbose=False,
            y_star=self.y_star,
            Xc=self.Xc,
            num_constraint_points=self.num_constraint_points,
            tau=self.tau,
            mc_samples=self.mc_samples,
            constraint_weight=self.constraint_weight,
        )
        gp_new.likelihood = lik_new
        
        return BoTorchVFEWrapper(
            gp=gp_new,
            likelihood=lik_new,
            cond_training_iter=self.cond_training_iter,
            cond_lr=self.cond_lr,
            cond_noise_eps=self.cond_noise_eps,

            y_star=self.y_star,
            Xc=self.Xc,
            num_constraint_points=self.num_constraint_points,
            tau=self.tau,
            mc_samples=self.mc_samples,
            constraint_weight=self.constraint_weight,
        )


def as_botorch_model(
    model: torch.nn.Module,
    y_star=None,
    Xc = None,
    num_constraint_points: int = 100,
    tau: float = 0.05,
    mc_samples: int = 16,
    constraint_weight: float = 1.0,
) -> Model:
    if not hasattr(model, "likelihood"):
        raise RuntimeError("Model has no .likelihood. In fit_model, do: model.likelihood = likelihood")
    return BoTorchVFEWrapper(
        gp=model,
        likelihood=model.likelihood,
        y_star=y_star,
        Xc=Xc,
        num_constraint_points=num_constraint_points,
        tau=tau,
        mc_samples=mc_samples,
        constraint_weight=constraint_weight,
    )