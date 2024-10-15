import numpy as np

import torch
from torch import Tensor
import torch.nn as nn

from gpytorch.means import ZeroMean
from gpytorch.models.approximate_gp import ApproximateGP
from gpytorch.priors import NormalPrior

from .latent_variables import VariationalLatentVariable
from ..variational.cholesky_kronecker_variational_distribution import CholeskyKroneckerVariationalDistributionV1
from ..variational.kronecker_variational_strategy import KroneckerVariationalStrategyV1
from ..likelihoods.my_multitask_gaussian_likelihood import MyMultitaskGaussianLikelihood
from ..likelihoods.poisson_likelihood import PoissonLikelihood

from utils import helper_init_kernel, helper_dimension_reduction, greedy_select_distant_points
from utils import helper_generate_list_kernels

class ApproximateGP_Kron(ApproximateGP):
    """
    ApproximateGP for LVMOGP, i.e. two sets of inputs are given.

    :param inputs:  tensor of shape ... * N * D
    :param latents: tensor of shape ... * Q * N_q * D_q
    """
    def __call__(self, latents: Tensor, inputs: Tensor, prior=False, **kwargs):
        if inputs.dim() == 1:
            inputs = inputs.unsqueeze(-1)

        if latents.dim() == 1:
            raise ValueError("Invalid latents. At least 2 dims, one refers to Q, one refers to N_q")
        
        elif latents.dim == 2:
            latents = latents.unsqueeze(-1)

        return self.variational_strategy(latents=latents, inputs=inputs, prior=prior, **kwargs)

class BayesianGPLVM(ApproximateGP_Kron):
    """
    Base class to construct different variants of LVMOGPs.
    : param H: Latent Variables
    """

    def __init__(self, H, variational_strategy):
        super().__init__(variational_strategy)

        # Assigning Latent Variable
        self.H = H

    def forward(self):
        raise NotImplementedError

    def sample_latent_variable(self, *args, **kwargs):
        sample = self.H(*args, **kwargs)
        return sample

class LVMOGP_SVI_v1(BayesianGPLVM):
    """
    Latent Variable MOGP with Stochastic Variational Inference, version 1. 

    For each output: a collection of Q latent variables together with Q kernels (possibly same) in latent space are used to model cross-output
    correlation. We may have Q kernels on input space, or share only 1 kernel. Overall covariance matrix is: \Sum_{q=1}^{Q} K_q(H_q, H_q) \otimes K_q(X, X)

    Inducing output is also collection of inducing latent variables. We have M_H inducing outputs (each corresponds to a collection of Q variables), 
    M_X inducing inputs. Inducing values are denoted as u, of length M_H * M_X.

    The variational distribution q(u)'s covariance matrix is assumed to have kronecker product structure. q(f) = \int q(f|u)q(u) du
    Each output has its own likelihood noise.

    :param Q: how many latent variables (latent kernels/input kernels) are introduced for each output.
    :list_input_kernels: either of length 1, or of length Q. length=1 suggests only 1 (input space) kernels is shared across different Q
    :list_latent_kernels: either of length 1, or of length Q. length=1 suggests only 1 (latent space) kernels is shared across different Q

    """
    
    def __init__(self,
                 n_outputs: int,
                 input_dim: int,
                 latent_dim: int,
                 n_inducing_latent: int,
                 n_inducing_input: int,
                 learn_inducing_locations_latent: bool=True,
                 learn_inducing_locations_input: bool=True, 
                 Q: int=3,
                 list_input_kernels: list=['Scale_RBF'],   # if len=1, sharing the same kernel 
                 list_latent_kernels: list=['Scale_RBF'],  # if len=1, sharing the same kernel 
                 **kwargs,
                 ):

        self.Q = Q
        # sanity check
        if len(list_input_kernels) != Q:
            assert len(list_input_kernels) == 1
        
        if len(list_latent_kernels) != Q:
            assert len(list_latent_kernels) == 1

        # q(f)
        self.inducing_outputs = torch.randn(Q, n_inducing_latent, latent_dim)
        self.inducing_inputs = torch.randn(n_inducing_input, input_dim)

        q_u = CholeskyKroneckerVariationalDistributionV1(n_inducing_latent=n_inducing_latent,
                                                         n_inducing_input=n_inducing_input)
        q_f = KroneckerVariationalStrategyV1(self, 
                                             inducing_points_latent=self.inducing_outputs,
                                             inducing_points_input=self.inducing_inputs,
                                             variational_distribution=q_u,
                                             learn_inducing_locations_latent=learn_inducing_locations_latent,
                                             learn_inducing_locations_input=learn_inducing_locations_input)
        
        # Define Prior for Latent Variables
        if 'latent_prior_mean' in kwargs and kwargs['latent_prior_mean'] != None:
            assert kwargs['latent_prior_mean'].dim() >= 2
            assert kwargs['latent_prior_mean'].shape[-1] == latent_dim
            assert kwargs['latent_prior_mean'].shape[-2] == n_outputs
            if kwargs['latent_prior_mean'].dim() == 2:
                latent_prior_mean = kwargs['latent_prior_mean'].unsqueeze(0).expand(self.Q, -1, -1)
            else:
                latent_prior_mean = kwargs['latent_prior_mean']
        else:    
            latent_prior_mean = torch.zeros(Q, n_outputs, latent_dim)

        # Notice that the prior for latent variables are fully factorized
        prior_latent_noise_scale = kwargs['prior_latent_noise_scale'] if 'prior_latent_noise_scale' in kwargs else 1.
        prior_latent = NormalPrior(latent_prior_mean, prior_latent_noise_scale * torch.ones_like(latent_prior_mean))

        H = VariationalLatentVariable(Q=Q, n_outputs=n_outputs, latent_dim=latent_dim, prior=prior_latent)

        super().__init__(H, q_f)

        # Means and Kernels
        self.mean_module = ZeroMean()
        self.covar_module_latent = nn.ModuleList(helper_generate_list_kernels(list_latent_kernels, list_of_input_dim=[latent_dim for _ in range(len(list_latent_kernels))]))
        self.covar_module_input = nn.ModuleList(helper_generate_list_kernels(list_input_kernels,  list_of_input_dim=[input_dim for _ in range(len(list_input_kernels))]))

        # Likelihood, 
        # (1) default choice: gaussain likelihood with one noise scale for every output.
        # (2) Poisson likelihood for integer (count) 0-255 data, the case for rotatedMNIST integer dataset ... 
        if 'possion_likelihood' in kwargs and kwargs['possion_likelihood'] == True:
            self.likelihood = PoissonLikelihood()

        else:
            self.likelihood = MyMultitaskGaussianLikelihood(num_tasks=n_outputs)

        # add config
        self.config = kwargs['config']
        
    def Q_covar_latent(self, latents: Tensor) -> Tensor:
        """
        desired shape of latents is ...* Q * N * D, aim to compute Q covariance matrices (N * N).  
        """
        if latents.dim() < 2:
            raise ValueError('Invalid latents. At least 2 dims, one refers to Q, one refers to N')

        elif latents.dim() == 2:
            latents = latents.unsqueeze(-1)
        
        if len(self.covar_module_latent) == 1:
            # sharing one kernel on latent space for all Q
            kernel = self.covar_module_latent[0]
            return kernel(latents).to_dense()

        elif len(self.covar_module_latent) == self.Q:
            list_Q_covar = [self.covar_module_latent[q](latents[q]).to_dense().unsqueeze(0) for q in range(self.Q)]
            return torch.cat(list_Q_covar, dim=0)

        else:
            raise ValueError('Incorrect covar_module_latent, go back and check __init__ step what wrong happens.')

    def Q_covar_input(self, inputs: Tensor) -> Tensor:
        """
        desired shape of inputs is ...* N * D, aim to compute Q covariance matrices (N * N), D refers to number of features.
        NOTE inputs are the same for all q \in {1, 2, ..., Q}, if sharing kernel, we only need to compute it once.
        """

        if inputs.dim() == 1:
            inputs = inputs.unsqueeze(-1)
        
        if len(self.covar_module_input) == 1:
            # sharing one kernel on input space for all Q
            kernel = self.covar_module_input[0]
            return kernel(inputs).to_dense().unsqueeze(0).expand(self.Q, -1, -1)
        
        elif len(self.covar_module_input) == self.Q:
            list_Q_covar = [self.covar_module_input[q](inputs).to_dense().unsqueeze(0) for q in range(self.Q)]
            return torch.cat(list_Q_covar, dim=0)

        else:
            raise ValueError('Incorrect covar_module_input, go back and check __init__ step what wrong happens.')
        

    def init_latent_variables(self, **kwargs):
        """
        if spatio_temporal dataset, we initailize model with normalized (lon, lat) tensor.
        """
        if self.config['dataset_type'] == 'spatio_temporal':
            lon_lat_tensor_reshape = kwargs['data_dict']['lon_lat_tensor']
            with torch.no_grad():
                self.H.q_mu[...] = lon_lat_tensor_reshape.unsqueeze(0).expand(self.config['Q'], -1, -1)
            
            if self.config['fix_latents'] == True:
                self.H.q_mu.requires_grad = False

        elif self.config['dataset_type'] == 'video' or self.config['dataset_type'] == 'simple_video':
            normed_coords = kwargs['data_dict']['normed_coords']
            with torch.no_grad():
                self.H.q_mu[...] = normed_coords.unsqueeze(0).expand(self.config['Q'], -1, -1)
            
            if self.config['fix_latents'] == True:
               self.H.q_mu.requires_grad = False

        elif self.config['dataset_type'] == 'human_Prostate_Cancer':
            if 'dim_reduction_method' in self.config and self.config['dim_reduction_method'] != 'random_initialization':
                if isinstance(self.likelihood, PoissonLikelihood):
                    # NOTE the inverse link function is exp(), it is better for us to first log the data_Y before 
                    # performing dimension reduction
                    data_Y = (kwargs['data_dict']['data_target'] + 1e-3).log()      # avoid the appearance of log(0)
                else: 
                    data_Y = kwargs['data_dict']['data_target']

                data_reduced_Y = helper_dimension_reduction(data_Y=data_Y,
                                                            method=self.config['dim_reduction_method'],
                                                            final_dim=self.config['latent_dim'],
                                                            config=self.config)
                # before using these data points as initialization, we need to normalize them
                # so that they spread near 0, and has variance near 1.
                normed_data_reduced_Y = (data_reduced_Y - data_reduced_Y.mean()) / data_reduced_Y.std()

                for q in range(self.config['Q']):
                    with torch.no_grad():
                        self.H.q_mu[q, ...] = normed_data_reduced_Y
            
                # whether we want to fix these latent variables
                if self.config['fix_latents'] == True:
                    self.H.q_mu.requires_grad = False

                # We will intialize inducing latent variables at the same time
                _, selected_latents = greedy_select_distant_points(normed_data_reduced_Y, num_points=self.config['n_inducing_latent'])
                self.init_inducing_latent_points(selected_latents)

            else: # random initialization 
                pass

        else: # for other dataset, we go into this general case ...
            if 'gplvm_init' in self.config and self.config['gplvm_init'] == True:
                from misc.gplvm_init import specify_gplvm, train_gplvm
                from gpytorch.likelihoods.gaussian_likelihood import GaussianLikelihoodWithMissingObs
                # run gplvm to initialize our model
                gplvm_model = specify_gplvm(self.config)
                gplvm_likelihood = GaussianLikelihoodWithMissingObs()
                data_Y = kwargs['data_dict']['data_target']
                gplvm_model, gplvm_likelihood, losses = train_gplvm(gplvm_model, 
                                                                    gplvm_likelihood,
                                                                    data_Y=data_Y,
                                                                    hyper_parameters={'training_steps': 5000,
                                                                                        'batch_size': min(100, data_Y.shape[0]),
                                                                                        'lr': 0.01})
                for q in range(self.config['Q']):
                    with torch.no_grad():
                        self.H.q_mu[q, ...] = gplvm_model.X().detach()

        
    def init_inducing_points(self, **kwargs):
        """
        init inducing locs in the input space
        """
        # init inducing points on input space
        if self.config['input_dim'] == 1:
            with torch.no_grad():
                self.variational_strategy.inducing_points_input.copy_(
                                torch.tensor(np.linspace(self.config['init_inducing_input_LB'], 
                                                        self.config['init_inducing_input_UB'], 
                                                        self.config['n_inducing_input']).reshape(-1, 1)))
        elif self.config['input_dim'] > 1:
            if 'init_inducing_points' in kwargs:
                with torch.no_grad():
                    self.variational_strategy.inducing_points_input.copy_(kwargs['init_inducing_points'])

                print('We have initialized input inducing point!')

            else:
                # randomly initialized by samples from standard Gaussian distribution.
                with torch.no_grad():
                    self.variational_strategy.inducing_points_input.copy_(torch.randn(self.config['n_inducing_input'], self.config['input_dim']))

        ### ------------------------------------------------------------------------------------------------------------------------------------------------------------

        # init inducing locs on latent space

        # if self.config['dataset_type'] == 'human_Prostate_Cancer':
        
        # if self.config['dataset_type'] == 'video':
        #     normed_coords = kwargs['init_latent_locs']
        #     assert normed_coords.shape[0] == self.config['n_inducing_latent']
        #     assert normed_coords.shape[1] == self.config['latent_dim']

        #     with torch.no_grad():
        #         self.variational_strategy.inducing_points_latent.copy_(
        #             normed_coords.unsqueeze(0).expand(self.config['Q'], -1, -1)
        #         )
    
    def init_inducing_latent_points(self, data, **kwargs):
        """
        """
        print('We have initialized latent inducing point!')
        for q in range(self.config['Q']):
            with torch.no_grad():
                self.variational_strategy.inducing_points_latent[q, ...] = data

    def init_kernels(self, **kwargs):
        """
        """
        for i, kernel in enumerate(self.covar_module_latent):
            helper_init_kernel(kernel, self.config['latent_kernels_init'][f'kernel{i+1}'], self.config['list_latent_kernels'][i])
        
        for j, kernel in enumerate(self.covar_module_input):
            helper_init_kernel(kernel, self.config['input_kernels_init'][f'kernel{j+1}'], self.config['list_input_kernels'][j])

    def init_likelihoods(self, **kwargs):
        """
        NOTE here we only consider MultitaskGaussianlikelihood not Poisson or orthers ... 
        """
        if isinstance(self.config['init_likelihood_noise'], list):
            with torch.no_grad():
                self.likelihood.task_noises = torch.tensor(self.config['init_likelihood_noise'])
        
        elif isinstance(self.config['init_likelihood_noise'], float):
            with torch.no_grad():
                self.likelihood.task_noises = torch.tensor([self.config['init_likelihood_noise'] for _ in range(self.config['n_outputs'])])
    
    def freeze_parameters(self, name_list: list):
        """
        Freeze part of the model parameters during training bu setting requires_grad = False.

        name_list contains strs, possible values are 'kernels', 'likelihoods', 'latent variables'
        """



