import torch
import torch.nn as nn

import gpytorch
from gpytorch.models import ExactGP
from gpytorch.likelihoods.gaussian_likelihood import GaussianLikelihood
from gpytorch.module import Module

from utils import (
    helper_specify_kernel_by_name,
    helper_init_kernel,
)

class ExactGPModel(ExactGP):
    def __init__(self, 
                 train_x, 
                 train_y, 
                 kernel_type: str,
                 likelihood=GaussianLikelihood(), 
                 ):
        """
        Exact GP model for single output. We MUST have GaussianLikelihood to ensure exactness.
        """
        super(ExactGPModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ZeroMean()
        input_dim = train_x.shape[-1]
        self.covar_module = helper_specify_kernel_by_name(kernel_type, input_dim)

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

class Multi_IndepExactGP(Module):
    """
    We model every output by an exact GP (with Gaussian Likelihood).

    :param n_outputs: number of outputs (number of indep GP models)
    :param ls_train_inputs: list of tensors, inputs for every output
    :param ls_train_targets: list of tensors, targets for every output
    :param input_kernels: list of strings, specify what type of kernel for each output. 
    """
    def __init__(self, 
                 n_outputs: int,
                 ls_train_inputs: list,         # list of tensors
                 ls_train_targets: list,        # list of tensors
                 input_kernels: list=['Scale_RBF'],
                 **kwargs,
                 ):
        
        super().__init__()

        # make sure kernels are specified for all outputs 
        if len(input_kernels) == 1:
            # repeat the same kind of kernel
            input_kernels = [input_kernels[0] for _ in range(n_outputs)]
        else:
            assert len(input_kernels) == n_outputs

        assert len(ls_train_inputs) == len(ls_train_targets) == n_outputs
        assert torch.is_tensor(ls_train_inputs[0]) == torch.is_tensor(ls_train_targets[0]) == True

        self.n_outputs = n_outputs
        self.ls_train_inputs = ls_train_inputs
        self.ls_train_targets = ls_train_targets
        self.input_kernels_list = input_kernels

        self.init_models()

    def init_models(self):
        """
        define an exact model for every output
        """
        models = []
        for id_output in range(self.n_outputs):
            curr_GPmodel = ExactGPModel(
                train_x = self.ls_train_inputs[id_output],
                train_y = self.ls_train_targets[id_output],
                kernel_type = self.input_kernels_list[id_output]
            )
            models.append(curr_GPmodel)

        self.model_list = nn.ModuleList(models)

    def get_model(self, output_id):
        """
        get the corresponding GP model given output id
        """
        if 0 <= output_id <= self.n_outputs - 1:
            return self.model_list[output_id]
        
        else: 
            raise ValueError(f"Model number must be between 0 and {self.n_outputs}")

    def init_kernels(self, **kwargs):
        """
        Notice that kernels for all GP models are using the same initialization 
        """
        config = kwargs['config']
        for id_output in range(self.n_outputs):
            my_model = self.get_model(id_output)
            helper_init_kernel(my_model.covar_module, config['kernel_init'], self.input_kernels_list[id_output])


    def init_likelihoods(self, **kwargs):
        """
        Notice that likelihoods for all GP models are using same initialization
        """
        config = kwargs['config']
        for id_output in range(self.n_outputs):
            my_model = self.get_model(id_output)
            with torch.no_grad():
                my_model.likelihood.noises = torch.tensor(config['init_likelihood_noise'])
