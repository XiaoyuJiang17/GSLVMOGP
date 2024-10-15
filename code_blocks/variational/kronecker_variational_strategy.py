from typing import Optional
from abc import ABC

import torch
from torch import Tensor

from gpytorch.module import Module
from gpytorch import settings
from gpytorch.utils.errors import CachingError
from gpytorch.settings import _linalg_dtype_cholesky, trace_mode
from gpytorch.distributions import MultivariateNormal, Distribution
from gpytorch.utils.memoize import cached, clear_cache_hook, pop_from_cache_ignore_args

from linear_operator.operators import (
    DenseLinearOperator,
    DiagLinearOperator,
    LinearOperator,
    MatmulLinearOperator,
    SumLinearOperator,
    TriangularLinearOperator,
    KroneckerProductLinearOperator,
)
from linear_operator.utils.cholesky import psd_safe_cholesky
from linear_operator import to_dense

from .cholesky_kronecker_variational_distribution import CholeskyKroneckerVariationalDistributionV1

class KroneckerVariationalStrategyV1(Module, ABC):
    """
    Our own variational strategy, cann't inherit from GpyTorch's _variational_strategy as more than one groups of 
    inducing points/model inputs (TODO). This variaional strategy is based on 'whitening' parametrization of q(u), i.e., 
    variational_distribution depicts q(u_0) instead of q(u) directly, where 
        u = L u_0, L is the cholesky factor of K_uu.
    
    Each inducing point on latent space has Q parts, i.e. it is formed by Q different latent variables.

    :param inducing_points_latent: tensor of shape ... * Q * M_q * D_q; 
        M_q: number of inducing points in q-th part, 
        D_q: dimensionality of q-th part latent variable, 
    """
    def __init__(
        self,
        model,
        inducing_points_latent: Tensor,
        inducing_points_input: Tensor,
        variational_distribution: CholeskyKroneckerVariationalDistributionV1,
        learn_inducing_locations_latent: bool = True,
        learn_inducing_locations_input: bool = True,
        jitter_val: Optional[float] = None,
    ):
        super().__init__()
        self._jitter_val = jitter_val

        # Model
        object.__setattr__(self, "model", model)

        # Inducing points latent in space and input space
        inducing_points_latent = inducing_points_latent.clone()
        inducing_points_input = inducing_points_input.clone()

        if inducing_points_latent.dim() == 1:
            raise ValueError("Invalid shape for inducing points in latent space, at least 2 dims, one refers to Q, one refers to M_q")
        
        elif inducing_points_latent.dim() == 2:
            inducing_points_latent = inducing_points_latent.unsqueeze(-1)

        if inducing_points_input.dim() == 1:
            inducing_points_input = inducing_points_input.unsqueeze(-1)

        if learn_inducing_locations_latent:
            self.register_parameter(name="inducing_points_latent", parameter=torch.nn.Parameter(inducing_points_latent))
        else:
            self.register_buffer("inducing_points_latent", inducing_points_latent)

        if learn_inducing_locations_input:
            self.register_parameter(name="inducing_points_input", parameter=torch.nn.Parameter(inducing_points_input))
        else:
            self.register_buffer("inducing_points_input", inducing_points_input)
        
        # Variational distribution
        self._variational_distribution = variational_distribution

    @cached(name="cholesky_factor", ignore_args=True)
    def _cholesky_factor(self, induc_induc_covar: LinearOperator) -> TriangularLinearOperator:
        L = psd_safe_cholesky(to_dense(induc_induc_covar).type(_linalg_dtype_cholesky.value()), max_tries=4)
        return TriangularLinearOperator(L)

    @property
    @cached(name="prior_distribution_memo")
    def prior_distribution(self) -> MultivariateNormal:
        zeros = torch.zeros(
            self._variational_distribution.shape(),
            dtype=self._variational_distribution.dtype,
            device=self._variational_distribution.device,
        )
        ones = torch.ones_like(zeros)
        res = MultivariateNormal(zeros, DiagLinearOperator(ones))
        return res

    @property
    @cached(name="variational_distribution_memo")
    def variational_distribution(self) -> Distribution:
        return self._variational_distribution()

    @property
    def jitter_val(self) -> float:
        if self._jitter_val is None:
            return settings.variational_cholesky_jitter.value(dtype=self.inducing_points_latent.dtype)
        return self._jitter_val

    @jitter_val.setter
    def jitter_val(self, jitter_val: float):
        self._jitter_val = jitter_val
    
    def _clear_cache(self) -> None:
        clear_cache_hook(self)

    def kl_divergence(self) -> Tensor:
        r"""
        Compute the KL divergence between the variational inducing distribution :math:`q(\mathbf u)`
        and the prior inducing distribution :math:`p(\mathbf u) = N(0,I)`.
        NOTE: p(\mathbf u) MUST be a standard normal distribution (specially designed for whitening). 
        """
        def kl_divergence_kronecker_wrt_identity(chol_variational_covar_latent: Tensor,
                                                 chol_variational_covar_input: Tensor,
                                                 variational_mean: Tensor):
            """Compute KL between q with Kronecker product covariance and p with 0 mean and Indentity covariance"""
            M_X, M_H = chol_variational_covar_input.shape[0], chol_variational_covar_latent.shape[0]
            _variational_mean = variational_mean.reshape(M_H, M_X)
            log_det_varcov_latent = (torch.prod(torch.diag(chol_variational_covar_latent)) ** 2).log()
            log_det_varcov_input = (torch.prod(torch.diag(chol_variational_covar_input)) ** 2).log()
            tr_varcov_latent = torch.sum(chol_variational_covar_latent ** 2)
            tr_carcov_input = torch.sum(chol_variational_covar_input ** 2)
            tr_MTM = torch.norm(_variational_mean, p='fro') ** 2 # trace(M^TM) = F_norm(M)^2
            
            res = - M_X * log_det_varcov_latent - M_H * log_det_varcov_input + tr_MTM + tr_carcov_input*tr_varcov_latent - M_X*M_H 
            return res / 2
        
        return kl_divergence_kronecker_wrt_identity(
                                    chol_variational_covar_latent=self._variational_distribution.chol_variational_covar_latent,
                                    chol_variational_covar_input=self._variational_distribution.chol_variational_covar_input,
                                    variational_mean=self._variational_distribution.variational_mean)
    def forward(
        self,
        latents: Tensor,
        inputs: Tensor,
        inducing_points_latent: Tensor,
        inducing_points_input: Tensor,
        inducing_values: Tensor, 
        variational_inducing_covar: Optional[LinearOperator] = None,
        full_covar: bool = True,
        **kwargs,
    ) -> MultivariateNormal:
        """
        :param full_covar: whether we want full covariance matrix in q(f) predictive distribution, if false, only diagonal elements are computed.
        """
        # Ensure latents, inputs has the same length, i.e. a (latents[i],inputs[i]) pair jointly determines a prediction value / target value.
        assert latents.shape[-2] == inputs.shape[-2]
        n_induc_latent, n_induc_input = inducing_points_latent.shape[-2], inducing_points_input.shape[-2]
        mini_batch_size = latents.shape[-2]
        Q = latents.shape[-3]
        
        # NOTE: following two tensors might contains repeting elements!
        full_latent = torch.cat([latents, inducing_points_latent], dim=-2)
        full_input = torch.cat([inputs, inducing_points_input], dim=-2)
        # of shape ... * Q * N * N
        # NOTE: 
        Q_full_covar_latent = self.model.Q_covar_latent(full_latent)
        Q_full_covar_input = self.model.Q_covar_input(full_input)
        # Covariance terms (for each q)
        Q_induc_latent_covar = Q_full_covar_latent[:, mini_batch_size:, mini_batch_size:]
        Q_induc_input_covar = Q_full_covar_input[:, mini_batch_size:, mini_batch_size:]

        if full_covar: # 3d tensor
            Q_data_data_covar = Q_full_covar_latent[:, :mini_batch_size, :mini_batch_size].mul(Q_full_covar_input[:, :mini_batch_size, :mini_batch_size]) # elementwise product
        else: # 2d tensor
            Q_data_data_covar = torch.diagonal(Q_full_covar_latent[:, :mini_batch_size, :mini_batch_size], dim1=1, dim2=2).mul(
                torch.diagonal(Q_full_covar_input[:, :mini_batch_size, :mini_batch_size], dim1=1, dim2=2))

        Q_induc_latent_data_latent_covar = Q_full_covar_latent[:, mini_batch_size:, :mini_batch_size] # (n_induc_latent, mini_batch_size)
        Q_induc_input_data_input_covar = Q_full_covar_input[:, mini_batch_size:, :mini_batch_size]    # (n_induc_input, mini_batch_size)
        
        # broadcasting
        Q_induc_data_covar = Q_induc_latent_data_latent_covar.to_dense().unsqueeze(-2) * Q_induc_input_data_input_covar.to_dense().unsqueeze(-3)
        Q_induc_data_covar = Q_induc_data_covar.reshape(Q, (n_induc_latent*n_induc_input), mini_batch_size)

        # Sum over Q 
        induc_data_covar = Q_induc_data_covar.sum(0)
        data_data_covar = Q_data_data_covar.sum(0)

        # Compute interpolation terms
        # K_uu^{-1/2} K_uf

        K_uu = KroneckerProductLinearOperator(Q_induc_latent_covar, Q_induc_input_covar).sum(0) # K_uu = LL^T
        L = self._cholesky_factor(K_uu)

        if L.shape != K_uu.shape:
            # Aggressive caching can cause nasty shape incompatibilies when evaluating with different batch shapes
            # TODO: Use a hook fo this
            try:
                pop_from_cache_ignore_args(self, "cholesky_factor")
            except CachingError:
                pass
            L = self._cholesky_factor(K_uu)

        interp_term = L.solve(induc_data_covar.type(_linalg_dtype_cholesky.value()))

        # Compute the mean of q(f), K_fu K_uu^{-1/2} u + test_mean (assume zero here)
        predictive_mean = (interp_term.transpose(-1, -2) @ (inducing_values.to(interp_term.dtype).unsqueeze(-1))).squeeze(-1)

        # Compute the covariance of q(f)
        middle_term = self.prior_distribution.lazy_covariance_matrix.mul(-1).to(interp_term.dtype)
        if variational_inducing_covar is not None:
            middle_term = SumLinearOperator(variational_inducing_covar, middle_term)

        # TODO: avoid multiplication with big matrices, use kronecker product properties ... 
        if full_covar:
            if trace_mode.on():
                predictive_covar = (
                    data_data_covar.add_jitter(self.jitter_val).to_dense()
                    + interp_term.transpose(-1, -2) @ middle_term.to_dense() @ interp_term
                )
            else:
                predictive_covar = SumLinearOperator(
                    DenseLinearOperator(data_data_covar).add_jitter(self.jitter_val),
                    MatmulLinearOperator(interp_term.transpose(-1, -2), middle_term.to(interp_term.dtype) @ interp_term),
                )
        else:
            # Use property diag(AB) = sum(A * B.T, axis=1)
            A, B = interp_term.transpose(-1, -2), middle_term.to(interp_term.dtype) @ interp_term
            predictive_covar = (
                data_data_covar + self.jitter_val + 
                A.mul(B.transpose(-1, -2)).sum(-1)
            )

            predictive_covar = torch.diag(predictive_covar)
        
        # Return the distribution
        return MultivariateNormal(predictive_mean, predictive_covar)
    
    def __call__(self, latents: Tensor, inputs: Tensor, prior: bool = False, **kwargs) -> MultivariateNormal:
        if prior:
            return self.model.forward(latents, inputs, **kwargs)

        if self.training:
            self._clear_cache()
        
        inducing_points_latent = self.inducing_points_latent
        inducing_points_input = self.inducing_points_input

        # Get p(u)/q(u)
        variational_dist_u = self.variational_distribution

        # Get q(f)
        if isinstance(variational_dist_u, MultivariateNormal): 
            return super().__call__(
                latents,
                inputs,
                inducing_points_latent,
                inducing_points_input,
                inducing_values=variational_dist_u.mean,
                variational_inducing_covar=variational_dist_u.lazy_covariance_matrix,
                **kwargs,
            )
        else:
            raise RuntimeError(
                f"Invalid variational distribuition ({type(variational_dist_u)}). "
                "Expected a multivariate normal or a delta distribution (NOT IMPLEMENTED YET)."
            )

        