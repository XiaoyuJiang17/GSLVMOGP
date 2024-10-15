import torch
from gpytorch.distributions import MultivariateNormal
from gpytorch.variational.cholesky_variational_distribution import _VariationalDistribution
from linear_operator.operators import CholLinearOperator, TriangularLinearOperator, KroneckerProductLinearOperator

class CholeskyKroneckerVariationalDistributionV1(_VariationalDistribution):
    """
    Single variational distribution q(u), the covariance matrix has kronecker product structure.
    
    :param n_inducing_input: the number of inducing points on input space.
    :param n_inducing_latent: the number of inducing points on latent space.
    """
    def __init__(self, 
                 n_inducing_input: int, 
                 n_inducing_latent: int,
                 batch_shape: torch.Size = torch.Size([]), 
                 mean_init_std: float = 1e-3,
                 **kwargs,
    ):  
        total_num_inducing_points = int(n_inducing_input*n_inducing_latent)

        super().__init__(total_num_inducing_points, batch_shape, mean_init_std)
        mean_init = torch.zeros(total_num_inducing_points)
        covar_input_init = torch.eye(n_inducing_input)
        covar_latent_init = torch.eye(n_inducing_latent)

        mean_init = mean_init.repeat(*batch_shape, 1)
        covar_input_init = covar_input_init.repeat(*batch_shape, 1, 1)
        covar_latent_init = covar_latent_init.repeat(*batch_shape, 1, 1)

        self.register_parameter(name='variational_mean', parameter=torch.nn.Parameter(mean_init))
        self.register_parameter(name='chol_variational_covar_input', parameter=torch.nn.Parameter(covar_input_init))
        self.register_parameter(name='chol_variational_covar_latent', parameter=torch.nn.Parameter(covar_latent_init))
    
    def forward(self) -> MultivariateNormal:
        chol_variational_covar_input = self.chol_variational_covar_input
        chol_variational_covar_latent = self.chol_variational_covar_latent
        dtype = chol_variational_covar_input.dtype
        device = chol_variational_covar_latent.device

        # First make the cholesky factor is upper triangular
        lower_mask_latent = torch.ones(self.chol_variational_covar_latent.shape[-2:], dtype=dtype, device=device).tril(0)
        chol_variational_covar_latent = TriangularLinearOperator(chol_variational_covar_latent.mul(lower_mask_latent))

        lower_mask_input = torch.ones(self.chol_variational_covar_input.shape[-2:], dtype=dtype, device=device).tril(0)
        chol_variational_covar_input = TriangularLinearOperator(chol_variational_covar_input.mul(lower_mask_input))

        # Now construct the actual covariance matrix
        variational_covar_latent = CholLinearOperator(chol_variational_covar_latent)
        variational_covar_input = CholLinearOperator(chol_variational_covar_input)

        self.variational_covar = KroneckerProductLinearOperator(variational_covar_latent, variational_covar_input)

        return MultivariateNormal(self.variational_mean, self.variational_covar)
    
    def initialize_variational_distribution(self, prior_dist: MultivariateNormal) -> None:
        raise NotImplementedError


class CholeskyKroneckerVariationalDistributionV2(_VariationalDistribution):
    """
    Composing a variational distribution for every q. Overall q(u) has block-diagonal covariance matrix, each block has kronecker product structure.
    
    :param n_inducing_input_per_q: number of inducing points on input space per q
    :param q: the number of latent variables corresponding to each output 
    """
    def __init__(
            self,
            n_inducing_input_per_q: int,
            n_inducing_latent_per_q: int,
            q: int,
            batch_shape: torch.Size=torch.Size([]),
            mean_init_std: float=1e-3,
            **kwargs, 
    ):
        total_n_inducing_points_per_q = int(n_inducing_input_per_q*n_inducing_latent_per_q)

        super().__init__(
            num_inducing_points=total_n_inducing_points_per_q,
            batch_shape=batch_shape,
            mean_init_std=mean_init_std
        )

        mean_init = torch.zeros(total_n_inducing_points_per_q)
        covar_input_init = torch.eye(n_inducing_input_per_q)
        covar_latent_init = torch.eye(n_inducing_latent_per_q)

        mean_init = mean_init.repeat(*batch_shape, q, 1)
        covar_input_init = covar_input_init.repeat(*batch_shape, q, 1, 1)
        covar_latent_init = covar_latent_init.repeat(*batch_shape, q, 1, 1)

        self.register_parameter(name='variational_mean', parameter=torch.nn.Parameter(mean_init))
        self.register_parameter(name='chol_variational_covar_input', parameter=torch.nn.Parameter(covar_input_init))
        self.register_parameter(name='chol_variational_covar_latent', parameter=torch.nn.Parameter(covar_latent_init))
    
    def forward(self) -> MultivariateNormal:
        chol_variational_covar_input = self.chol_variational_covar_input
        chol_variational_covar_latent = self.chol_variational_covar_latent
        dtype = chol_variational_covar_input.dtype
        device = chol_variational_covar_latent.device

        # First make the cholesky factor is upper triangular
        lower_mask_latent = torch.ones(self.chol_variational_covar_latent.shape[-2:], dtype=dtype, device=device).tril(0)
        chol_variational_covar_latent = TriangularLinearOperator(chol_variational_covar_latent.mul(lower_mask_latent))
        
        lower_mask_input = torch.ones(self.chol_variational_covar_input.shape[-2:], dtype=dtype, device=device).tril(0)
        chol_variational_covar_input = TriangularLinearOperator(chol_variational_covar_input.mul(lower_mask_input))

        # Now construct the actual covariance matrix
        variational_covar_latent = CholLinearOperator(chol_variational_covar_latent)
        variational_covar_input = CholLinearOperator(chol_variational_covar_input)

        self.variational_covar = KroneckerProductLinearOperator(variational_covar_latent, variational_covar_input)

        return MultivariateNormal(self.variational_mean, self.variational_covar)

    def shape(self) -> torch.Size:
        r"""
        Event + q + batch shape of VariationalDistribution object
        :rtype: torch.Size
        """
        return torch.Size([*self.batch_shape, self.q, self.num_inducing_points])
    