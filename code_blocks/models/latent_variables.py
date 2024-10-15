import torch
import numpy as np
from torch.distributions import kl_divergence

from gpytorch.module import Module
from gpytorch.mlls.added_loss_term import AddedLossTerm

class LatentVariable(Module):
    """
    Base class for all latent variables

    :param n_outputs: how many outputs we are considering
    :param Q: how many latent variables we assign for each output
    :param latent_dim: dimensionality used to depict a latent variable
    """

    def __init__(self, n_outputs, Q, latent_dim):
        super().__init__()
        self.n_outputs = n_outputs
        self.Q = Q
        self.latent_dim = latent_dim

    def forward(self, x):
        raise NotImplementedError
    
class VariationalLatentVariable(LatentVariable):

    def __init__(self, Q, n_outputs, latent_dim, prior, **kwargs):
        super().__init__(n_outputs=n_outputs, Q=Q, latent_dim=latent_dim)

        self.prior = prior
        self.Q = Q
        self.q_mu = torch.nn.Parameter(torch.randn(Q, n_outputs, latent_dim))
        self.q_log_sigma = torch.nn.Parameter(torch.randn(Q, n_outputs, latent_dim))

        self.register_added_loss_term("latents_kl")

    def forward(self, batch_idx=None, **kwargs):

        if batch_idx is None:
            batch_idx = np.arange(self.n_outputs) 
        
        q_mu_batch = self.q_mu[:, batch_idx, :]
        q_log_sigma_batch = self.q_log_sigma[:, batch_idx, :]

        q_x = torch.distributions.Normal(q_mu_batch, q_log_sigma_batch.exp())

        p_mu_batch = self.prior.loc[:, batch_idx, :]
        p_var_batch = self.prior.variance[:, batch_idx, :]

        p_x = torch.distributions.Normal(p_mu_batch, p_var_batch)
        # NOTE: latents_kl is KL averaged over mini-batch ONLY (sum over Q and latent_dim)
        latents_kl = kl_gaussian_loss_term(q_x, p_x, len(batch_idx))
        self.update_added_loss_term('latents_kl', latents_kl)
        
        return q_x.rsample()

class kl_gaussian_loss_term(AddedLossTerm):
    
    def __init__(self, q_x, p_x, batch_size):
        self.q_x = q_x
        self.p_x = p_x
        self.batch_size = batch_size
       
    def loss(self): 
        # NOTE: this implementation only works when p_x and q_x both diagonal. ONLY KL between scalar distributions are computed.
        # The following gives KLs of same shape as q_x or p_x, sum over axis=-1 gives Q by batch_size KLs, each KL correspond to
        # one pair of multi-dim latent variables.
        kls = kl_divergence(self.q_x, self.p_x).sum(axis=-1)

        return kls.sum() / self.batch_size