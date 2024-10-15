import torch
from gpytorch.likelihoods import _OneDimensionalLikelihood
from gpytorch.distributions import base_distributions

class PoissonLikelihood(_OneDimensionalLikelihood):
    def __init__(self):
        super().__init__()

    def forward(self, function_samples, **kwargs):
        return base_distributions.Poisson(rate=function_samples.exp())
