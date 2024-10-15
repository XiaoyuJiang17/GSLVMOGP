from typing import Any, Optional, Union
import math

import torch
from torch import Tensor
from torch.distributions import Normal
from torch.distributions import Distribution as _Distribution

from gpytorch.priors import Prior
from gpytorch.constraints import Interval
from gpytorch.distributions import MultivariateNormal
from gpytorch.likelihoods import MultitaskGaussianLikelihood

class MyMultitaskGaussianLikelihood(MultitaskGaussianLikelihood):
    """
    MultitaskGaussianLikelihood with different noise scale parameter for every output. Support mini-batch sampling among all outputs. 
    """
    def __init__(self,
                num_tasks: int,
                rank: int = 0,
                batch_shape: torch.Size = torch.Size(),
                task_prior: Optional[Prior] = None,
                noise_prior: Optional[Prior] = None,
                noise_constraint: Optional[Interval] = None
                ) -> None:
        # NOTE: We assume no global noises, only task related noises.
        super().__init__(num_tasks, rank, batch_shape, task_prior, noise_prior, noise_constraint, has_global_noise=False, has_task_noise=True)
    
    def expected_log_prob(self, target: Tensor, input: MultivariateNormal, task_indices: Tensor = None, *params: Any, **kwargs: Any) -> Tensor:
        # TODO: support more complex situations, like missing data and batch-based computing ... 
        assert target.shape[0] == input.mean.shape[0] == task_indices.shape[0]
        selected_noises = torch.index_select(self.task_noises, 0, task_indices)

        mean, variance = input.mean, input.variance
        res = ((target - mean).square() + variance) / selected_noises + selected_noises.log() + math.log(2 * math.pi)
        res = res.mul(-0.5)

        return res

    def marginal(self, function_dist: MultivariateNormal, task_indices: Tensor = None, *args: Any, **kwargs: Any) -> _Distribution:
        assert function_dist.mean.shape[0] == task_indices.shape[0]
        mean, covar = function_dist.mean, function_dist.lazy_covariance_matrix
        # pick some task noises and diagonalize them
        selected_noises = torch.index_select(self.task_noises, 0, task_indices)
        noise_matrix = torch.diag_embed(selected_noises)
        full_covar = covar + noise_matrix
        
        return function_dist.__class__(mean, full_covar)
