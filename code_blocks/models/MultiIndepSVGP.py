from typing import Union

import numpy as np

import torch
import torch.nn as nn

import gpytorch
from gpytorch.distributions import MultivariateNormal
from gpytorch.variational import CholeskyVariationalDistribution
from gpytorch.variational import VariationalStrategy
from gpytorch.models import ApproximateGP
from gpytorch.likelihoods.gaussian_likelihood import GaussianLikelihood
from gpytorch.module import Module

from ..likelihoods.poisson_likelihood import PoissonLikelihood
from utils import (
    helper_specify_kernel_by_name,
    helper_init_kernel,
)

class SVGP(ApproximateGP):
    """
    SVGP for one output
    """
    def __init__(self,
                 n_inducing_points: int,
                 kernel_type: str='Scale_RBF',
                 likelihood=GaussianLikelihood(),
                 learn_inducing_locations: bool=True,
                 input_dim: int=1,
                ):
        
        variational_distribution = CholeskyVariationalDistribution(num_inducing_points=n_inducing_points)
        inducing_points = torch.randn(n_inducing_points, input_dim)
        variational_strategy = VariationalStrategy(self, 
                                                   inducing_points=inducing_points, 
                                                   variational_distribution=variational_distribution,
                                                   learn_inducing_locations=learn_inducing_locations)
        super().__init__(variational_strategy)
        self.mean_module = gpytorch.means.ZeroMean()
        self.covar_module = helper_specify_kernel_by_name(kernel_type, input_dim)
        self.likelihood = likelihood
    
    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return MultivariateNormal(mean_x, covar_x)

class Multi_IndepSVGP(Module):
    """
    for every output, we have an independent SVGP with its own (gaussian) likelihood.   

    :param n_outputs: number of outputs (note that possibly some outputs have no test data, i.e. the model has no need to be trained)
    :param n_inducing_points: if int, all SVGP use same number of inducing points, otherwise every output is associated with a model with different number of inducing points.
    :param learn_inducing_locations: if int, whether or not the inducing points is trained for all SVGPs is specified the same.
    :param input_kernels: if length = 1, use the same type of kernel for all outputs (NOT mean sharing the kernel!) 
    """
    def __init__(self,
                 n_outputs: int,
                 input_dim: int,
                 n_inducing_points: Union[int, list],
                 learn_inducing_locations: Union[bool, list],
                 input_kernels: list=['Scale_RBF'],
                 likelihood_type: str='GaussianLikelihood', # Other options: 'PoissonLikelihood'
                 **kwargs):
        
        super().__init__()

        # sanity checks
        if isinstance(n_inducing_points, list):
            assert len(n_inducing_points) == n_outputs
        elif isinstance(n_inducing_points, int):
            n_inducing_points = [n_inducing_points for _ in range(n_outputs)]
        
        if isinstance(learn_inducing_locations, list):
            assert len(learn_inducing_locations) == n_outputs
        elif isinstance(learn_inducing_locations, bool):
            learn_inducing_locations = [learn_inducing_locations for _ in range(n_outputs)]

        # make sure kernels are specified for all outputs 
        if len(input_kernels) == 1:
            # repeat the same kind of kernel
            input_kernels = [input_kernels[0] for _ in range(n_outputs)]

        self.n_outputs = n_outputs
        self.input_dim = input_dim
        self.n_inducing_points = n_inducing_points
        self.learn_inducing_locations = learn_inducing_locations
        self.input_kernels = input_kernels
        self.likelihood_type = likelihood_type
        
        self.init_models()

    def init_models(self):
        """
        define a SVGP model for every output
        """
        models = []
        for id_output in range(self.n_outputs):
            if self.likelihood_type == 'GaussianLikelihood':
                curr_SVGP = SVGP(n_inducing_points=self.n_inducing_points[id_output],
                                kernel_type=self.input_kernels[id_output],
                                likelihood=GaussianLikelihood(),
                                learn_inducing_locations=self.learn_inducing_locations[id_output],
                                input_dim=self.input_dim)
                
            elif self.likelihood_type == 'PoissonLikelihood':
                curr_SVGP = SVGP(n_inducing_points=self.n_inducing_points[id_output],
                                kernel_type=self.input_kernels[id_output],
                                likelihood=PoissonLikelihood(),
                                learn_inducing_locations=self.learn_inducing_locations[id_output],
                                input_dim=self.input_dim)
                
            models.append(curr_SVGP)

        self.model_list = nn.ModuleList(models)

    def get_model(self, output_id):
        """
        get the corresponding SVGP model given output id
        """
        if 0 <= output_id <= self.n_outputs - 1:
            return self.model_list[output_id]
        
        else: 
            raise ValueError(f"Model number must be between 0 and {self.n_outputs}")

    def init_inducing_points(self, **kwargs):
        """
        Notice that all SVGPs have the same initialization
        """
        config = kwargs['config']
        for id_output in range(self.n_outputs):
            my_model = self.get_model(id_output)

            with torch.no_grad():
                my_model.variational_strategy.inducing_points.copy_(
                    torch.tensor(np.linspace(
                        config['init_inducing_input_LB'],
                        config['init_inducing_input_UB'],
                        config['n_inducing_points']).reshape(-1, 1)))

    def init_kernels(self, **kwargs):
        """
        Notice that kernels for all SVGP models are using the same initialization 
        """
        config = kwargs['config']
        for id_output in range(self.n_outputs):
            my_model = self.get_model(id_output)
            helper_init_kernel(my_model.covar_module, config['kernel_init'], self.input_kernels[id_output])

    def init_likelihoods(self, **kwargs):
        """
        Notice that likelihoods for all SVGP models are using same initialization
        """
        try:
            assert self.likelihood_type == 'GaussianLikelihood'
            config = kwargs['config']
            for id_output in range(self.n_outputs):
                my_model = self.get_model(id_output)
                with torch.no_grad():
                    my_model.likelihood.noises = torch.tensor(config['init_likelihood_noise'])

        except AssertionError:
            pass
        