preimport os
import itertools
from tqdm import tqdm
import random
import numpy as np

import pandas as pd
import matplotlib.pyplot as plt

import torch
from torch import Tensor
from torchvision import datasets, transforms
from gpytorch.kernels import ScaleKernel, RBFKernel, MaternKernel, PeriodicKernel, LinearKernel
from gpytorch.distributions import MultivariateNormal
from gpytorch.settings import _linalg_dtype_cholesky

from linear_operator import to_dense
from linear_operator.operators import LinearOperator, TriangularLinearOperator
from linear_operator.operators.kronecker_product_linear_operator import KroneckerProductLinearOperator, KroneckerProductTriangularLinearOperator
from linear_operator.utils.cholesky import psd_safe_cholesky

from code_blocks.kernels.periodic_inputs_maternkernel import PeriodicInputsMaternKernel

def _cholesky_factor(induc_induc_covar: LinearOperator) -> TriangularLinearOperator:
    L = psd_safe_cholesky(to_dense(induc_induc_covar).type(_linalg_dtype_cholesky.value()), max_tries=4)
    return TriangularLinearOperator(L)

def prepare_large_scale_synthetic_regression_data(config):
    """
    Generate MOGP dataset (Q=1, only one latent variable per output) following the procedure:
        (1). Sample latent variables from N(0, I).
        (2). Specify (ground truth) kernels on input and output space, specify kernel hyper-parameters and likelihood noise scale.
        (3). Specify Input range, how many data points in total, how many for train and how many for test ... 
        (4). Generate dataset by sampling from grond truth distribution (ground truth kronecker product covariance matrix) ... 
    """

    assert config['Q'] == 1
    data_random_seed = config['data_random_seed'] if 'data_random_seed' in config else 1
    torch.manual_seed(data_random_seed)

    # function specific hyper-parameters
    num_total_available_inputs = 100

    # Sample H
    ground_truth_H = torch.randn(config['n_outputs'], config['latent_dim']) * config['prior_latent_noise_scale']

    # Specify kernels
    latent_kernel = helper_specify_kernel_by_name(config['list_latent_kernels'][0], config['latent_dim'])
    input_kernel = helper_specify_kernel_by_name(config['list_input_kernels'][0], config['input_dim'])

    # Init the kernels
    helper_init_kernel(latent_kernel, config['latent_kernels_init']['kernel1'], kernel_type=config['list_latent_kernels'][0])
    helper_init_kernel(input_kernel, config['input_kernels_init']['kernel1'], kernel_type=config['list_input_kernels'][0])

    # Specify inputs
    data_inputs = torch.tensor(np.linspace(config['min_input_bound'], config['max_input_bound'], num_total_available_inputs))

    # Ground Truth covariance matrices
    covar_input = input_kernel(data_inputs)
    covar_latent = latent_kernel(ground_truth_H)

    cholesky_factor_input = _cholesky_factor(covar_input)
    cholesky_factor_latent = _cholesky_factor(covar_latent)

    cholesky_factor_overall = KroneckerProductTriangularLinearOperator(cholesky_factor_latent, cholesky_factor_input)

    assert num_total_available_inputs*config['n_outputs'] == cholesky_factor_overall.shape[0]
    sample = cholesky_factor_overall._matmul(torch.randn(cholesky_factor_overall.shape[0])).detach()
    sample_final = sample + torch.randn_like(sample) * config['ground_truth_likelihood_noise']

    data_target = sample_final.reshape(config['n_outputs'], num_total_available_inputs)
    
    # Heterogeneous inputs for different outputs 
    
    input_full_list = [i for i in range(num_total_available_inputs)]

    ls_of_ls_train_input, ls_of_ls_test_input = [], []

    for output_id in range(config['n_outputs']):

        if config['n_train_inputs']  <= len(input_full_list):
            indices_perm = torch.randperm(len(input_full_list))
            train_indices = indices_perm[:config['n_train_inputs']]
            test_indices = indices_perm[config['n_train_inputs']: config['n_train_inputs'] + config['n_test_inputs']]
        else:
            raise ValueError("n_inputs is larger than the length of input_full_list")
        
        curr_train_id_list = [input_full_list[i] for i in train_indices]
        curr_test_id_list = [input_full_list[j] for j in test_indices]

        ls_of_ls_train_input.append(curr_train_id_list)
        ls_of_ls_test_input.append(curr_test_id_list)

    means, stds = torch.zeros(config['n_outputs']), torch.ones(config['n_outputs'])

    results_dict = {'ground_truth_H': ground_truth_H,
                    'data_inputs': data_inputs,
                    'data_target': data_target,
                    'ls_of_ls_train_input': ls_of_ls_train_input,
                    'ls_of_ls_test_input': ls_of_ls_test_input,
                    'means': means,
                    'stds': stds}
    
    return results_dict

def generate_grid_latents(min_bound, max_bound, latent_dim: int, num_points_per_dim: int, n_outputs: int, **kwargs):
    """
    Generate grided latent variables. All dims of the latent variables are between [min_bound, max_bound].
    There are num_points_per_dim points EVENLY spaced along each dim.

    Return:
        ground_H : 2d Tensor, of shape (config['n_outputs'], latent_dim)
    """
    assert n_outputs == num_points_per_dim**latent_dim
    # e.g, latent space is 2d, total number of outputs is 25 ----> only 5 points per dim is valid

    # Generate evenly spaced points for each dimension
    points_per_dim = np.linspace(min_bound, max_bound, num_points_per_dim)

    # Generate all combinations of points in the latent space
    grid_points = list(itertools.product(points_per_dim, repeat=latent_dim))

    # Check if the number of generated points matches the expected number of outputs
    assert len(grid_points) == n_outputs, "Generated number of grid points does not match the expected number."

    # Convert the list of points to a tensor
    ground_H = torch.tensor(grid_points, dtype=torch.double)

    return ground_H

def prepare_synthetic_demo_data(config, **kwargs):
    """
    Check if the LVMOGP can recover true covariance on latent space.

    1. We first generate ground true latent variable H, and use user-specified kernel (with well chosen hyperparameters) to construct ground truth covariance matrix K(H, H),
        we save the plot as the 'ground truth' plots.
    2. Input locations (for every output) are evenly spaced between min_input_bound and max_input_bound given in config. The number of input points is also user-defined.
        by doing this, we defined K(X, X), with user-chosen kernel function.
    3. We generate dataset by 
        (1) sample f using N(0, K_X(X, X) \otimes K_H(H, H)) 
        (2) get y using f + \epsilon, where \epsilon have user-defined std noise with Gaussian Distribution.

    GOAL: we want to apply LVMOGP model on this synthetic dataset and get well-trained latent vectors H, and we would like to check if groud truth K(H, H) is recovered.
    NOTE every output holds a different group of inputs (Heterogeneous inputs).
    """
    data_random_seed = config['data_random_seed'] if 'data_random_seed' in config else 1
    torch.manual_seed(data_random_seed)

    # How to get ground truth latents?

    # Approach 1. Random Samples
    # ground_H = torch.randn(config['n_outputs'], config['latent_dim'])

    # Approach 2. Grids
    
    try:
        num_points_per_dim = int(config['n_outputs'] ** (1 / config['latent_dim']))
    except:
        raise ValueError('Invalid config for generating ground_H via grids !')

    ground_H = generate_grid_latents(min_bound = config['min_latent_bound'],
                                     max_bound = config['max_latent_bound'],
                                     latent_dim=config['latent_dim'],
                                     num_points_per_dim=num_points_per_dim,
                                     n_outputs=config['n_outputs'])


    latent_kernel = helper_specify_kernel_by_name(config['list_latent_kernels'][0], config['latent_dim'])
    input_kernel = helper_specify_kernel_by_name(config['list_input_kernels'][0], config['input_dim'])

    # Inputs are 1-dimensional
    num_total_available = 2 * config['n_inputs']
    data_inputs = torch.tensor(np.linspace(config['min_input_bound'], config['max_input_bound'], num_total_available))

    # Init the kernels
    helper_init_kernel(latent_kernel, config['latent_kernels_init']['kernel1'], kernel_type=config['list_latent_kernels'][0])
    helper_init_kernel(input_kernel, config['input_kernels_init']['kernel1'], kernel_type=config['list_input_kernels'][0])

    covar_input = input_kernel(data_inputs)
    covar_latent = latent_kernel(ground_H)

    # Save ground truth plot of covar_latent
    results_folder_path = kwargs['results_folder_path'] if 'results_folder_path' in kwargs else None
    if results_folder_path != None:
        helper_synthetic_demo_plot(covar_latent, results_folder_path)

    covar_all = KroneckerProductLinearOperator(covar_latent, covar_input)
    covar_final = covar_all.add_jitter(config['init_likelihood_noise'])
    mean_all = torch.zeros(covar_all.shape[0])

    # mvn = MultivariateNormal(mean=mean_all, covariance_matrix=covar_final)
    mvn = torch.distributions.MultivariateNormal(mean_all, covariance_matrix=covar_final.to_dense().detach())
    sample = mvn.sample()

    data_target = sample.reshape(config['n_outputs'], num_total_available)

    # Heterogeneous inputs 
    
    input_full_list = [i for i in range(num_total_available)]

    ls_of_ls_train_input, ls_of_ls_test_input = [], []

    for output_id in range(config['n_outputs']):

        n = config['n_inputs'] 
        if n <= len(input_full_list):
            indices = torch.randperm(len(input_full_list))[:n]
        else:
            raise ValueError("n_inputs is larger than the length of input_full_list")
        
        curr_train_id_list = [input_full_list[i] for i in indices]
        curr_test_id_list = [j for j in input_full_list if j not in curr_train_id_list]
        assert len(curr_train_id_list) == len(curr_test_id_list) == config['n_inputs']

        ls_of_ls_train_input.append(curr_train_id_list)
        ls_of_ls_test_input.append(curr_test_id_list)

    means, stds = torch.zeros(config['n_outputs']), torch.ones(config['n_outputs'])

    # ls_of_ls_train_input = [input_full_list for _ in range(config['n_outputs'])]
    # NO test points actually.
    # ls_of_ls_test_input = [[] for _ in range(config['n_outputs'])]

    results_dict = {'ground_H': ground_H,
                    'data_inputs': data_inputs,
                    'data_target': data_target,
                    'ls_of_ls_train_input': ls_of_ls_train_input,
                    'ls_of_ls_test_input': ls_of_ls_test_input,
                    'means': means,
                    'stds': stds}
    
    return results_dict

# helper functions for following func ... 
def f1(x: Tensor, param_dict: dict):
    '''
    This is the ground truth function f1. x is the inputs to the function, param_dict is a dict describing function parameters.
    function form copied from GPAR paper f1. 
    '''
    term1 = - torch.sin(param_dict['a']*torch.pi*( param_dict['b']*x + param_dict['c'] )) / (param_dict['d']*x + param_dict['e'])
    term2 = - x**param_dict['f']
    return term1 + term2

def f2(x: Tensor, param_dict: dict):
    '''
    Similar to f1 func above, this is copied from GPAR paper f2.
    '''
    term1 = param_dict['a'] * torch.exp( param_dict['b'] * x)
    term2 = param_dict['c'] * torch.cos( param_dict['d'] * torch.pi * x) + param_dict['e'] * torch.cos( param_dict['f'] * torch.pi * x)
    term3 = torch.sqrt( param_dict['g'] * x)
    term4 = param_dict['h']

    return term1 * term2 + term3 + term4

def prepare_synthetic_regression_data(config):
    """
    Show why large number of outputs is benefical.

    Assume ground truth func f(x), x in [0, 1], only few input-output pairs for this func is observed, but observations for many noisy func f^{noisy} (f^{noisy} if func with some 
    perturbation applied on func parameters) are also available, we want to show the ability to recover the truth function becomes stronger and stronger as we have more and more
    related outputs.

    Return: 
        data_target: the first output for the ground truth function f, all others are relevant outputs.
    """
    data_random_seed = config['data_random_seed'] if 'data_random_seed' in config else 1
    torch.manual_seed(data_random_seed)

    data_inputs = torch.tensor(np.linspace(0, 1, config['n_total_input']))

    # all target data (before any data normalization...)
    all_data = torch.zeros(config['n_outputs'], config['n_total_input'])

    for id_output in range(config['n_outputs']):
        # The first output is the Ground Truth function
        if config['func_choice'] == 'f1':
            if id_output == 0:
                param_dict = {'a':10, 'b':1, 'c': 1, 'd': 2, 'e':1, 'f': 4}
                output_1 = f1(data_inputs, param_dict)
                obs_noise = torch.randn(output_1.shape[0]) * config['obs_noise'] # observation noise
                all_data[id_output, :] = output_1 + obs_noise
            else:
                e1, e2, e3, e4, _, e6 = (torch.rand(6) - 0.5) * config['data_param_noise']
                # 5 func parameters are randomly perturbed
                param_dict = {'a':10+e1, 'b':1+e2, 'c': 1+e3, 'd': 2+e4, 'e':1, 'f': 4+e6}
                curr_output = f1(data_inputs, param_dict)
                obs_noise = torch.randn(curr_output.shape[0]) * config['obs_noise'] # observation noise
                all_data[id_output, :] = curr_output + obs_noise

        elif config['func_choice'] == 'f2':
            if id_output == 0:
                param_dict = {'a':0.2, 'b':0.5, 'c':1, 'd':9, 'e':1, 'f':7, 'g':11, 'h':-2}
                output_1 = f2(data_inputs, param_dict)
                obs_noise = torch.randn(output_1.shape[0]) * config['obs_noise'] # observation noise
                all_data[id_output, :] = output_1 +  obs_noise
            else:
                e1, e2, e3, e4, e5, e6, e7, e8 = (torch.rand(8) - 0.5) * config['data_param_noise']
                perturb_param_dict = {'a':0.2+e1, 'b':0.5+e2, 'c':1+e3, 'd':9+e4, 'e':1+e5, 'f':7+e6, 'g':11+e7, 'h':-2+e8 }
                curr_output = f2(data_inputs, perturb_param_dict)
                obs_noise = torch.randn(curr_output.shape[0]) * config['obs_noise'] # observation noise
                all_data[id_output, :] = curr_output + obs_noise
        else: 
            raise NotImplementedError

    # Train/Test data split 
    input_ids_full_list = [i for i in range(config['n_total_input'])]
    ls_of_ls_train_input, ls_of_ls_test_input = [], []
    for output_id in range(config['n_outputs']):
        # this is how many training points for each output
        n = config['n_inputs'] 
        if n <= len(input_ids_full_list):
            indices = torch.randperm(len(input_ids_full_list))[:n]
        else:
            raise ValueError("n_inputs is larger than the length of input_ids_full_list")
        
        if output_id == 0: # Manually define input locations for first output (the one we are interested in)
            curr_train_id_list = [3, 18, 99, 165, 188]
            curr_test_id_list = [j for j in input_ids_full_list if j not in curr_train_id_list]
        else: 
            # THIS actually trivial! As we are not interested in eval results on these outputs.
            curr_train_id_list = [input_ids_full_list[i] for i in indices]
            curr_test_id_list = [j for j in input_ids_full_list if j not in curr_train_id_list][:195] # NOTE keep the same length of ls_of_ls_test_input ! 

        # assert len(curr_train_id_list) == config['n_inputs']
        assert len(curr_test_id_list) == config['n_total_input'] - 5

        ls_of_ls_train_input.append(curr_train_id_list)
        ls_of_ls_test_input.append(curr_test_id_list)

    # Optionally, we apply data normalization based on statistics on train split to get data_target, or no normalization at all.
    data_target = all_data
    means, stds = torch.zeros(config['n_outputs']), torch.ones(config['n_outputs'])

    results_dict = {'data_inputs': data_inputs,
                    'data_target': data_target,
                    'ls_of_ls_train_input': ls_of_ls_train_input,
                    'ls_of_ls_test_input': ls_of_ls_test_input,
                    'means': means,
                    'stds': stds}
    
    return results_dict


def prepare_mocap_data(config):
    """
    """
    results_dict = {}
    return results_dict

def prepare_gusto_1_indiv_data(config):
    """
    Pick only 1 individual data in arthor's datasets, and filter out all rows with missing values (NaN values). ------> 366 rows
    Each row responds to an output. That is 366 outputs (if id=010-04004), every output is a 4-elements time series.
        The dataset contains 10 columns, but only columns ['3Output', '9Output', '48Output', '72Output'] are what we are interested in. 
    
    There are several settings:
        (1). 'Inference_for_72month' mode: We randomly pick some '72Output' as test (masked during training), all other data are available during training
        (2). 'Imputation' mode: For every output (row), we randomly pick 1 or 2 data as test, all other data are available during training
    """
    data_random_seed = config['data_random_seed'] if 'data_random_seed' in config else 1
    np.random.seed(data_random_seed)
    # torch.manual_seed(data_random_seed)

    columns_as_targets = ['3Output', '9Output', '48Output', '72Output']

    _all_data = pd.read_csv(config['gusto_1_indiv_data_path'])[columns_as_targets] 
    all_data = torch.tensor(_all_data.values) # first numpy ndarray, then tensor
    assert all_data.shape[0] == config['n_outputs']
   
    translate_bias = config['min_input_bound']
    translate_scale = (config['max_input_bound'] - config['min_input_bound']) / 72
    data_inputs =  translate_bias + translate_scale * ( torch.tensor([3, 9, 48, 72]) )

    # Collect lists of indices of input points for every output
    ls_of_ls_train_input, ls_of_ls_test_input = [], []
    input_full_list = [i for i in range(4)]

    # At the same time, compute statistics and standarize orignal data by statistics from train split ... 
    means, stds, data_target = torch.zeros(config['n_outputs']), torch.zeros(config['n_outputs']), torch.zeros_like(all_data)

    if config['data_split_setting'] == 'Inference_for_72month':
        # Randomly pick #config['n_test_72Output'] outputs from output with id 1,2,...,config['n_outputs']
        # data_random_seed = config['data_random_seed'] if 'data_random_seed' in config else 1
        # np.random.seed(data_random_seed)
        selected_output_ids = np.random.choice(range(config['n_outputs']), config['n_test_72Output'], replace=False)

        for id_output in range(config['n_outputs']):
            # for some output, as one element is used for test ... 
            if id_output not in selected_output_ids:
                ls_of_ls_train_input.append(input_full_list)
                ls_of_ls_test_input.append([])

            elif id_output in selected_output_ids:
                ls_of_ls_train_input.append(input_full_list[:3])
                ls_of_ls_test_input.append(input_full_list[-1:])

            # Compute statistics and normalize data ... 
            means[id_output] = all_data[id_output, ls_of_ls_train_input[-1]].mean()
            stds[id_output] = all_data[id_output, ls_of_ls_train_input[-1]].std()

            data_target[id_output, :] = (all_data[id_output, :] - means[id_output]) / (stds[id_output] + 1e-8)

    elif config['data_split_setting'] == 'Imputation':
        raise NotImplementedError
    
    else:
        raise NotImplementedError

    results_dict = {'data_target': data_target,
                    'data_inputs':data_inputs,
                    'ls_of_ls_train_input': ls_of_ls_train_input,
                    'ls_of_ls_test_input': ls_of_ls_test_input,
                    'means': means,
                    'stds': stds}
    
    return results_dict

def prepare_gusto_multiple_indiv_data(config):
    """
    Take all individuals data in horvath & blood train split in arthor's dataset, and filter out all rows with missing values (NaN values). ----> 35044 rows
    Each row responds to an output.  Every output is a 4-elements time series.
        The dataset contains 10 columns, but only columns ['3Output', '9Output', '48Output', '72Output'] are what we are interested in. 

    NOTE: Almost all code from prepare_gusto_1_indiv_data can be reused, except data_path: 
        for single individual, data_path is "gusto_1_indiv_data_path", but now "gusto_multiple_indiv_data_path"
    """
    # This is the only extra thing we need to do, and reuse (all) code from prepare_gusto_1_indiv_data
    config['gusto_1_indiv_data_path'] = config['gusto_multiple_indiv_data_path']

    results_dict = prepare_gusto_1_indiv_data(config)

    return results_dict

def prepare_gusto_pick1CpG_data(config):

    """
    In config, there is a path for 'entire dataset' with multiple individuals and multiple genes. And there is a variable called CpG_name, 
        which used to select rows we want (which CpG gene we want).
    Pick the data related to one CpG from given entire dataset.

    NOTE entire dataset should be a clearn dataset, no missing values.
    """

    data_random_seed = config['data_random_seed'] if 'data_random_seed' in config else 1
    np.random.seed(data_random_seed)

    columns_as_targets = ['3Output', '9Output', '48Output', '72Output']

    # entire dataset: multiple individuals, multiple CpGs
    _all_data = pd.read_csv(config['gusto_entire_dataset'])
    _pick_CpG_data = _all_data[_all_data['ProbeID'] == config['CpG_name']]
    pick_CpG_data = torch.tensor(_pick_CpG_data[columns_as_targets].values)
    print(f'Dataset with CpG: {config["CpG_name"]} has shape: ', pick_CpG_data.shape)
    assert pick_CpG_data.shape[0] == config['n_outputs']

    translate_bias = config['min_input_bound']
    translate_scale = (config['max_input_bound'] - config['min_input_bound']) / 72
    data_inputs =  translate_bias + translate_scale * ( torch.tensor([3, 9, 48, 72]) )

    # Collect lists of indices of input points for every output
    ls_of_ls_train_input, ls_of_ls_test_input = [], []
    input_full_list = [i for i in range(4)]
    input_full_tensor = torch.tensor(input_full_list)

    # At the same time, compute statistics and standarize orignal data by statistics from train split ... 
    means, stds, data_target = torch.zeros(config['n_outputs']), torch.zeros(config['n_outputs']), torch.zeros_like(pick_CpG_data)

    if config['data_split_setting'] == 'Inference_for_72month':
        selected_output_ids = np.random.choice(range(config['n_outputs']), config['n_test_72Output'], replace=False)

        for id_output in range(config['n_outputs']):
            # for some output, the last one element is used for test ... 
            if id_output not in selected_output_ids:
                ls_of_ls_train_input.append(input_full_list)
                ls_of_ls_test_input.append([])

            elif id_output in selected_output_ids:
                ls_of_ls_train_input.append(input_full_list[:3])
                ls_of_ls_test_input.append(input_full_list[-1:])

            # Compute statistics and normalize data ... 
            means[id_output] = pick_CpG_data[id_output, ls_of_ls_train_input[-1]].mean()
            stds[id_output] = pick_CpG_data[id_output, ls_of_ls_train_input[-1]].std()

            data_target[id_output, :] = (pick_CpG_data[id_output, :] - means[id_output]) / (stds[id_output] + 1e-10)

    elif config['data_split_setting'] == 'Imputation':
        # select a certain number of outputs, and do imputation (randomly mask 1 of 4 target values during training and left for testing.)
        selected_output_ids = np.random.choice(range(config['n_outputs']), config['n_test_72Output'], replace=False)

        for id_output in range(config['n_outputs']):
            # no test data for some output
            if id_output not in selected_output_ids:
                ls_of_ls_train_input.append(input_full_list)
                ls_of_ls_test_input.append([])

            elif id_output in selected_output_ids:
                perm = torch.randperm(input_full_tensor.size(0))
                selected = input_full_tensor[perm[0:1]]
                remaining = input_full_tensor[perm[1:]]
                ls_of_ls_train_input.append(input_full_tensor[remaining].tolist())
                ls_of_ls_test_input.append(input_full_tensor[selected].tolist())

            # Compute statistics and normalize data ... 
            means[id_output] = pick_CpG_data[id_output, ls_of_ls_train_input[-1]].mean()
            stds[id_output] = pick_CpG_data[id_output, ls_of_ls_train_input[-1]].std()

            data_target[id_output, :] = (pick_CpG_data[id_output, :] - means[id_output]) / (stds[id_output] + 1e-10)
    
    else:
        raise NotImplementedError

    results_dict = {'data_target': data_target,
                    'data_inputs':data_inputs,
                    'ls_of_ls_train_input': ls_of_ls_train_input,
                    'ls_of_ls_test_input': ls_of_ls_test_input,
                    'means': means,
                    'stds': stds}
    
    return results_dict

def prepare_gusto_pick1CpG_v2_data(config):
    """
    For test outputs, the input data are splitted into 3 parts: training (used in ELBO optimization), reference (conditioning for prediction), testing (for evaluation).
        MOST code copied from function prepare_gusto_pick1CpG_data.
    Returns:
    :param ls_of_ls_train_input: list of list, index for training data.
    :param ls_of_ls_ref_input: list of list, index for reference data (for conditioning); note that if there are reference inputs for certain output, there MUST have test input as well.
    :param ls_of_ls_test_input: list of list, index for testing data; note that if there are test input for certain output, there MUST have reference inputs as well.
    """
    data_random_seed = config['data_random_seed'] if 'data_random_seed' in config else 1
    np.random.seed(data_random_seed)

    columns_as_targets = ['3Output', '9Output', '48Output', '72Output']

    # entire dataset: multiple individuals, multiple CpGs
    _all_data = pd.read_csv(config['gusto_entire_dataset'])
    _pick_CpG_data = _all_data[_all_data['ProbeID'] == config['CpG_name']]
    pick_CpG_data = torch.tensor(_pick_CpG_data[columns_as_targets].values)
    print(f'Dataset with CpG: {config["CpG_name"]} has shape: ', pick_CpG_data.shape)
    assert pick_CpG_data.shape[0] == config['n_outputs']

    translate_bias = config['min_input_bound']
    translate_scale = (config['max_input_bound'] - config['min_input_bound']) / 72
    data_inputs =  translate_bias + translate_scale * ( torch.tensor([3, 9, 48, 72]) )

    # Collect lists of indices of input points for every output
    ls_of_ls_train_input, ls_of_ls_ref_input, ls_of_ls_test_input = [], [], []
    input_full_list = [i for i in range(4)]

    # At the same time, compute statistics and standarize orignal data by statistics from train split ... 
    means, stds, data_target = torch.zeros(config['n_outputs']), torch.zeros(config['n_outputs']), torch.zeros_like(pick_CpG_data)

    if config['data_split_setting'] == 'Inference_for_72month':
        selected_output_ids = np.random.choice(range(config['n_outputs']), config['n_test_72Output'], replace=False)

        for id_output in range(config['n_outputs']):
            # for some output, as one element is used for test ... 
            if id_output not in selected_output_ids:
                ls_of_ls_train_input.append(input_full_list)
                ls_of_ls_ref_input.append([])
                ls_of_ls_test_input.append([])

                # Compute statistics and normalize data ... 
                means[id_output] = pick_CpG_data[id_output, input_full_list].mean()
                stds[id_output] = pick_CpG_data[id_output, input_full_list].std()

            elif id_output in selected_output_ids:
                # TODO avoid this hard seperation! 
                ls_of_ls_train_input.append(input_full_list[0:-2]) # at least 2 data points to have a valid variance!
                ls_of_ls_ref_input.append(input_full_list[-2:-1])
                ls_of_ls_test_input.append(input_full_list[-1:])

                # Compute statistics and normalize data ... 
                means[id_output] = pick_CpG_data[id_output, input_full_list[:-1]].mean()
                stds[id_output] = pick_CpG_data[id_output, input_full_list[:-1]].std()

            assert stds.isnan().any() == False

            data_target[id_output, :] = (pick_CpG_data[id_output, :] - means[id_output]) / (stds[id_output] + 1e-10)

    elif config['data_split_setting'] == 'Imputation':
        raise NotImplementedError
    
    else:
        raise NotImplementedError

    results_dict = {'data_target': data_target,
                    'data_inputs':data_inputs,
                    'ls_of_ls_train_input': ls_of_ls_train_input,
                    'ls_of_ls_ref_input': ls_of_ls_ref_input,
                    'ls_of_ls_test_input': ls_of_ls_test_input,
                    'means': means,
                    'stds': stds}
    
    return results_dict

def prepare_exchange_data(config):
    """
    exchange datasets from OILMM paper.
        The dataset all_df contains 14 columns: 13 target time series, 1 input series. Each time series is of 251 rows,
        though some of the elements are missing for USD/XAG (#missing=8), USD/XAU (#missing=9), USD/XPT (#missing=42).
        We are interested in making predict on (artificially masked targets) of 3 time series: USD/CAD (#test = 51), USD/JPY (#test = 51), and USD/AUD (#test = 51).
    """
    columns_as_targets = ['USD/CHF', 'USD/EUR', 'USD/GBP', 'USD/HKD', 'USD/KRW', 'USD/MXN', 'USD/NZD', 'USD/XAG', 'USD/XAU', 'USD/XPT', 'USD/CAD', 'USD/JPY', 'USD/AUD']

    all_df = pd.read_csv(config['exchange_all_data_path'])
    train_df = pd.read_csv(config['exchange_train_data_path']) # test points are replaced as NaN.
    assert set(train_df.iloc[:, 1:].columns.to_list()) == set(columns_as_targets) 
    assert all_df.shape[0] == 251

    n_inputs_per_output = 251
    assert all_df.shape[1] == config['n_outputs'] + 1 # due to first column 'year' are inputs to the model.

    # Original inputs, without transformation
    data_inputs = torch.tensor(all_df['year']) 

    # Collect lists of indices of input points for every output
    ls_of_ls_train_input, ls_of_ls_test_input = [], []

    for col_name in all_df.columns:
        if col_name != 'year':
            if col_name != 'USD/CAD' and col_name != 'USD/JPY' and col_name != 'USD/AUD':
                # all data used for training, no test points for these outputs
                ls_of_ls_train_input.append(all_df.index[all_df[col_name].notna()].to_list())
                ls_of_ls_test_input.append([])
            
            elif col_name == 'USD/CAD':
                test_index_1 = [i for i in range(49, 100)]
                train_index_1 = [i for i in range(n_inputs_per_output) if i not in test_index_1]
                ls_of_ls_train_input.append(train_index_1)
                ls_of_ls_test_input.append(test_index_1)
            
            elif col_name == 'USD/JPY':
                test_index_2 = [i for i in range(99, 150)]
                train_index_2 = [i for i in range(n_inputs_per_output) if i not in test_index_2]
                ls_of_ls_train_input.append(train_index_2)
                ls_of_ls_test_input.append(test_index_2)
            
            elif col_name == 'USD/AUD':
                test_index_3 = [i for i in range(149, 200)]
                train_index_3 = [i for i in range(n_inputs_per_output) if i not in test_index_3]
                ls_of_ls_train_input.append(train_index_3)
                ls_of_ls_test_input.append(test_index_3)

    assert len(ls_of_ls_train_input) == 13 == len(ls_of_ls_test_input)

    # Compute statistics: make sure no test data statistics are leaked
    means, stds = torch.tensor(train_df.iloc[:, 1:].mean().to_numpy()), torch.tensor(train_df.iloc[:, 1:].std().to_numpy())

    # Normalize datasets
    norm_all_df = (all_df - train_df.mean()) / train_df.std()
    data_target = torch.tensor(norm_all_df.iloc[:, 1:].to_numpy().T)  # of shape: num_outputs * 251
    assert data_target.shape[0] == config['n_outputs']
    assert data_target.shape[1] == 251

    results_dict = {'data_target': data_target,
                    'data_inputs': data_inputs,
                    'ls_of_ls_train_input': ls_of_ls_train_input,
                    'ls_of_ls_test_input': ls_of_ls_test_input,
                    'means': means,
                    'stds': stds}
    
    return results_dict

def prepare_eeg_data(config):
    """
    EGG dataset from OILMM paper.
        The egg_data dataset has 8 columns (column names: time, F1, F2, F3, F4, F5, F6, FZ) and 256 rows, no missing data.
        'time' column serves as the inputs, other columns serve as target time series.
        The last 100 rows of F1, F2, FZ are masked during training, they serve as test data.
    """
    columns_as_targets = ['F1', 'F2', 'F3', 'F4', 'F5', 'F6', 'FZ']

    all_df = pd.read_csv(config['eeg_all_data_path'])
    train_df = pd.read_csv(config['eeg_train_data_path'])
    assert set(train_df.iloc[:, 1:].columns.to_list()) == set(columns_as_targets)
    assert all_df.shape[0] == 256

    n_inputs_per_output = 256
    assert all_df.shape[1] == config['n_outputs'] + 1 # due to first column 'time' are inputs to the model.

    # Original inputs, without transformation
    data_inputs = torch.tensor(all_df['time'])

    # Collect lists of indices of input points for every output
    ls_of_ls_train_input, ls_of_ls_test_input = [], []
    input_full_list = [i for i in range(n_inputs_per_output)]

    for col_name in all_df.columns:
        if col_name != 'time':
            if col_name == 'F1' or col_name == 'F2' or col_name == 'FZ':
                # Last 100 datapoints used for testing, rest of them for training
                ls_of_ls_train_input.append(input_full_list[:-100])
                ls_of_ls_test_input.append(input_full_list[-100:])

            else:
                # All datapoints for training, NO test data
                ls_of_ls_train_input.append(input_full_list)
                ls_of_ls_test_input.append([])

    assert len(ls_of_ls_train_input) == 7 == len(ls_of_ls_test_input)    

    # Compute statistics: make sure no test data statistics are leaked
    means, stds = torch.tensor(train_df.iloc[:, 1:].mean().to_numpy()), torch.tensor(train_df.iloc[:, 1:].std().to_numpy())

    # Normalize datasets
    norm_all_df = (all_df - train_df.mean()) / train_df.std()
    data_target = torch.tensor(norm_all_df[columns_as_targets].to_numpy().T)  # of shape: num_outputs * 256
    
    assert data_target.shape[0] == config['n_outputs']
    assert data_target.shape[1] == 256

    results_dict = {'data_target': data_target,
                    'data_inputs': data_inputs,
                    'ls_of_ls_train_input': ls_of_ls_train_input,
                    'ls_of_ls_test_input': ls_of_ls_test_input,
                    'means': means,
                    'stds': stds}
    
    return results_dict

def prepare_spatio_temporal_data(config):
    """
    Spatio-Temporal data download from :
        https://cds.climate.copernicus.eu/cdsapp#!/dataset/projections-cmip5-monthly-single-levels?tab=form
    
    Basicly the (target) data is a 3D tensor, (#times, #lat, #lon) = (363, 145, 192), we are interested in subset of it,
    in particular, subsampling via "::rate".

    The setting is: 
        We pick some grids of (lat, lon) pairs as the outputs we are interested in. For every output we picked, the last 100 datapoints are left for testing.
        The training data can be 
            (1). all remaining datapoints (263) ----> easy
            (2). a random batch of remaining points (of user chosen size) , the batches for every output might be different. ----> hard
    """
    all_data = torch.load(config['all_data_path'])
    lat = torch.load(config['lat_path'])
    lon = torch.load(config['lon_path'])

    assert all_data.shape[1] == lat.shape[0]
    assert all_data.shape[2] == lon.shape[0]

    n_inputs_per_output = all_data.shape[0] #times

    translate_bias = config['min_input_bound']
    translate_scale = (config['max_input_bound'] - config['min_input_bound']) / n_inputs_per_output
    data_inputs =  translate_bias + translate_scale * ( torch.tensor([i for i in range(n_inputs_per_output)]) )

    # Pick some (lat,lon) pairs from original data
    picked_data = all_data[:, ::config['rate'], ::config['rate']]
    picked_lat = lat[::config['rate']]
    picked_lon = lon[::config['rate']]

    num_picked_lat, num_picked_lon = picked_data.shape[-2], picked_data.shape[-1]
    picked_data_permute = picked_data.permute(2,1,0)
    # first lon with all lats, second lon with all lats, ..., last lon with all lats.
    picked_data_reshape = picked_data_permute.reshape(num_picked_lat*num_picked_lon, n_inputs_per_output)
    num_outputs = picked_data_reshape.shape[0]

    print(f'This Spatio-Temporal dataset has {num_outputs} outputs!')
    assert num_outputs == config['n_outputs']

    # Collect lists of indices of input points for every output
    ls_of_ls_train_input, ls_of_ls_test_input = [], []
    input_full_list = [i for i in range(n_inputs_per_output)]

    # At the same time, compute statistics and standarize orignal data by statistics from train split ... 
    means, stds, data_target = torch.zeros(num_outputs), torch.zeros(num_outputs), torch.zeros_like(picked_data_reshape)

    if config['data_split_setting'] == 'hard':
        data_random_seed = config['data_random_seed'] if 'data_random_seed' in config else 1
        np.random.seed(data_random_seed)
        list_expri_random_seeds = np.random.randn(num_outputs)

    for id_output in range(num_outputs):
        # We always have last 100 datapoints for all outputs for test ... 
        if config['data_split_setting'] == 'easy':
            ls_of_ls_train_input.append(input_full_list[:-100])
            ls_of_ls_test_input.append(input_full_list[-100:])

        elif config['data_split_setting'] == 'hard':
            random.seed(list_expri_random_seeds[id_output])
            train_index = random.sample(range(263), config['n_input_we_want_per_output'])
            ls_of_ls_train_input.append(train_index)
            ls_of_ls_test_input.append(input_full_list[-100:])
        
        else:
            raise NotImplementedError
        
        # Compute statistics and normalize data ... 
        means[id_output] = picked_data_reshape[id_output, ls_of_ls_train_input[-1]].mean()
        stds[id_output] = picked_data_reshape[id_output, ls_of_ls_train_input[-1]].std()

        data_target[id_output, :] = (picked_data_reshape[id_output, :] - means[id_output]) / (stds[id_output] + 1e-8)

    # For spatio-temporal data, we are also interested in (lon, lat) pairs.
    lon_lat_tensor = torch.zeros(num_picked_lon, num_picked_lat, 2)
    for id_lon in range(num_picked_lon):
        for id_lat in range(num_picked_lat): 
            lon_lat_tensor[id_lon, id_lat, :] = torch.tensor([picked_lon[id_lon].item(), picked_lat[id_lat].item()])
    
    lon_lat_tensor_reshape = lon_lat_tensor.reshape(-1, 2)
    lon_lat_means, lon_lat_stds = lon_lat_tensor_reshape.mean(dim=0), lon_lat_tensor_reshape.std(dim=0, unbiased=False)
    lon_lat_tensor = (lon_lat_tensor_reshape - lon_lat_means.unsqueeze(0)) / lon_lat_stds.unsqueeze(0)

    results_dict = {'data_target': data_target,
                    'data_inputs': data_inputs,
                    'ls_of_ls_train_input': ls_of_ls_train_input,
                    'ls_of_ls_test_input': ls_of_ls_test_input,
                    'means': means,
                    'stds': stds,
                    'lon_lat_tensor': lon_lat_tensor,
                    'lon_lat_means': lon_lat_means,
                    'lon_lat_stds': lon_lat_stds,
                    }
    
    return results_dict

def prepare_spatio_temporal_data_v2(config):
    """
    Version 2 of the above prepare_spatio_temporal_data func.

    Spatio-Temporal data download from :
        https://cds.climate.copernicus.eu/cdsapp#!/dataset/projections-cmip5-monthly-single-levels?tab=form
    
    Basicly the (target) data is a 3D tensor, (#times, #lat, #lon) = (363, 145, 192).
    We are interested in subset of it, in particular, the european part.
        lat: 100 (35.)   ---- 130 (72.5)      Wiki: (36)     --- (70.8)
        lon: 90 (168.75) ---- 132 (247.5)     Wiki: (170.69) --- (246.1)
    This leads to totally 31 * 43 = 1333 outputs, we pick some (within-domain) outputs as testing locations: 
    (lat, lon) pairs on grids: choose lat among {115}, choose lon among {110}, Thus, 30 * 42 = 1260 outputs for training.
    
    For each training output, dataset is splitted into 3 parts: training, within-test, extrapolation-test.    
    """
    all_data = torch.load(config['all_data_path'])
    lat = torch.load(config['lat_path'])
    lon = torch.load(config['lon_path'])

    assert all_data.shape[1] == lat.shape[0]
    assert all_data.shape[2] == lon.shape[0]

    n_inputs_per_output = all_data.shape[0] #times

    translate_bias = config['min_input_bound']
    translate_scale = (config['max_input_bound'] - config['min_input_bound']) / n_inputs_per_output
    data_inputs =  translate_bias + translate_scale * ( torch.tensor([i for i in range(n_inputs_per_output)]) )

    # Pick some (lat, lon) pairs from original data
    lats_ids = torch.tensor([x for x in range(100, 131) if x not in [115]])
    lons_ids = torch.tensor([x for x in range(90, 133) if x not in [110]])

    _picked_data = torch.index_select(all_data, 1, lats_ids)
    picked_data = torch.index_select(_picked_data, 2, lons_ids)
    # picked_data = all_data[:, lats_ids, lons_ids]
    picked_lat = lat[lats_ids]
    picked_lon = lon[lons_ids]

    assert picked_data.shape[1] == picked_lat.shape[0] == 30
    assert picked_data.shape[2] == picked_lon.shape[0] == 42

    num_picked_lat, num_picked_lon = picked_data.shape[-2], picked_data.shape[-1]
    picked_data_permute = picked_data.permute(2,1,0)
    # first lon with all lats, second lon with all lats, ..., last lon with all lats.
    picked_data_reshape = picked_data_permute.reshape(num_picked_lat*num_picked_lon, n_inputs_per_output)
    num_outputs = picked_data_reshape.shape[0]

    print(f'This Spatio-Temporal dataset has {num_outputs} outputs!')
    assert num_outputs == config['n_outputs']

    # Collect lists of indices of input points for every output
    ls_of_ls_train_input, ls_of_ls_within_test_input, ls_of_ls_extrapolate_test_input = [], [], []
    input_full_list = [i for i in range(n_inputs_per_output)]

    # At the same time, compute statistics and standarize orignal data by statistics from train split ... 
    means, stds, data_target = torch.zeros(num_outputs), torch.zeros(num_outputs), torch.zeros_like(picked_data_reshape)

    if config['data_split_setting'] == 'hard':
        data_random_seed = config['data_random_seed'] if 'data_random_seed' in config else 1
        np.random.seed(data_random_seed)
        list_expri_random_seeds = np.random.randn(num_outputs)
    
    for id_output in range(num_outputs):

        # We always have last 100 datapoints for all outputs for test ... 
        if config['data_split_setting'] == 'easy':
            raise NotImplementedError

        elif config['data_split_setting'] == 'hard':
            random.seed(list_expri_random_seeds[id_output])
            train_index = random.sample(range(263), config['n_input_we_want_per_output'])
            within_test_index = [x for x in range(263) if x not in train_index]
            ls_of_ls_train_input.append(train_index)
            ls_of_ls_within_test_input.append(within_test_index)
            ls_of_ls_extrapolate_test_input.append(input_full_list[-100:])
        
        else:
            raise NotImplementedError
        
        # Compute statistics and normalize data ... 
        means[id_output] = picked_data_reshape[id_output, ls_of_ls_train_input[-1]].mean()
        stds[id_output] = picked_data_reshape[id_output, ls_of_ls_train_input[-1]].std()

        data_target[id_output, :] = (picked_data_reshape[id_output, :] - means[id_output]) / (stds[id_output] + 1e-8)


    # For spatio-temporal data, we are also interested in (lon, lat) pairs.
    lon_lat_tensor = torch.zeros(num_picked_lon, num_picked_lat, 2)
    for id_lon in range(num_picked_lon):
        for id_lat in range(num_picked_lat): 
            lon_lat_tensor[id_lon, id_lat, :] = torch.tensor([picked_lon[id_lon].item(), picked_lat[id_lat].item()])
    
    lon_lat_tensor_reshape = lon_lat_tensor.reshape(-1, 2)
    lon_lat_means, lon_lat_stds = lon_lat_tensor_reshape.mean(dim=0), lon_lat_tensor_reshape.std(dim=0, unbiased=False)
    lon_lat_tensor = (lon_lat_tensor_reshape - lon_lat_means.unsqueeze(0)) / lon_lat_stds.unsqueeze(0)

    results_dict = {'data_target': data_target,
                    'data_inputs': data_inputs,
                    'ls_of_ls_train_input': ls_of_ls_train_input,
                    'ls_of_ls_within_test_input': ls_of_ls_within_test_input,
                    'ls_of_ls_extrapolate_test_input': ls_of_ls_extrapolate_test_input, 
                    'means': means,
                    'stds': stds,
                    'all_lon_lat_tensor': all_data,
                    'all_lat': lat,
                    'all_lon': lon,
                    'lon_lat_tensor': lon_lat_tensor,
                    'lon_lat_means': lon_lat_means,
                    'lon_lat_stds': lon_lat_stds,
                    }
    
    return results_dict

import cv2
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset

class VideoDataset(Dataset):
    def __init__(self, video_path, transform=None):
        self.cap = cv2.VideoCapture(video_path)
        self.transform = transform
        self.length = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = self.cap.read()
        if ret:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            if self.transform:
                frame = self.transform(frame)
            return frame
        else:
            return None
        
class SubsetVideoDataset(torch.utils.data.Dataset):
    def __init__(self, original_dataset, rate, start=None, end=None):
        self.original_dataset = original_dataset
        self.rate = rate

        # start and end determines the range for which we subset the original video
        if start == None:
            start = 0
        if end == None:
            end = len(original_dataset)
        
        self.indices = [i for i in range(start, end, rate)]
        
    def __len__(self):
        return len(self.indices)
    
    def __getitem__(self, idx):
        original_idx = self.indices[idx]
        return self.original_dataset[original_idx]

def helper_count_row_unique_values(tensor_2d: torch.tensor):
    """
    Counts the number of unique values for each row of the given tensor.

    :param tensor_2d: 2d tensor, each row represent the time series for one output (pixel).

    Return:
        a 1d tensor containing the number of unique values for all outputs.
    """
    import pandas as pd
    df = pd.DataFrame(tensor_2d.numpy())
    unique_counts = df.apply(lambda x: x.nunique(), axis=1)
    
    return torch.tensor(unique_counts.values)

def prepare_video_data(config):
    """
    Prepare a timeseries (n_timeframes) of (3, n_rows, n_columns) video, which result to a 3 * n_rows * n_columns outputs.
    We apply a filtering to the video data, only keep outputs (timeseries) with a large number of unique values (more smooth; less unique value ---> looks more piecewise constant).
    
    """
    n_rows, n_columns = 200, 200
    original_video = torch.load(config['video_path']) # 2d tensor, (#outputs, #inputs)

    n_inputs_per_output = original_video.shape[1]
    num_original_outputs = original_video.shape[0]
    print(f'This Video dataset originally has {num_original_outputs} outputs!\n')
    assert num_original_outputs == 3 * n_rows * n_columns

    # We aim at filtering the outputs, only keep smooth ones
    unique_counts = helper_count_row_unique_values(original_video)
    rows_filter = unique_counts > config['output_unique_values_threshold']
    filtered_video = original_video[rows_filter, :]
    num_filtered_outputs = filtered_video.shape[0]
    print(f'After filtering, dataset has {num_filtered_outputs} outputs!')
    assert num_filtered_outputs == config['n_outputs']

    # Then we focus on the filtered outputs .... 
    translate_bias = config['min_input_bound']
    translate_scale = (config['max_input_bound'] - config['min_input_bound']) / n_inputs_per_output
    data_inputs =  translate_bias + translate_scale * ( torch.tensor([i for i in range(n_inputs_per_output)]) )

    ls_of_ls_train_input, ls_of_ls_test_input = [], []

    # At the same time, compute statistics and standarize orignal data by statistics from train split ... 
    means, stds, data_target = torch.zeros(num_filtered_outputs), torch.zeros(num_filtered_outputs), torch.zeros_like(filtered_video)

    # Make random seeds ready to be used
    data_random_seed = config['data_random_seed'] if 'data_random_seed' in config else 1
    np.random.seed(data_random_seed)
    list_expri_random_seeds = np.random.randn(num_filtered_outputs)

    print('Prepare dataset: splitting train/test')
    for id_output in tqdm(range(num_filtered_outputs)):
        random.seed(list_expri_random_seeds[id_output])
        train_index = random.sample(range(n_inputs_per_output), config['n_input_we_want_per_output'])
        within_test_index = [x for x in range(n_inputs_per_output) if x not in train_index]
        ls_of_ls_train_input.append(train_index)
        ls_of_ls_test_input.append(within_test_index)

        # Compute statistics and normalize data ... 
        means[id_output] = filtered_video[id_output, ls_of_ls_train_input[-1]].mean()
        stds[id_output] = filtered_video[id_output, ls_of_ls_train_input[-1]].std()

        data_target[id_output, :] = (filtered_video[id_output, :] - means[id_output]) / (stds[id_output] + 1e-8)
    
    # for video dataset, we are also interested in coordinates to describe every output (channel, pixel, pixel)
    all_coords = torch.cartesian_prod(torch.tensor([0, 1, 2]), torch.arange(n_rows), torch.arange(n_columns)).float()

    filtered_coords = all_coords[rows_filter]
    coords_means, coords_stds = filtered_coords.mean(dim=0), (filtered_coords.std(dim=0, unbiased=False) + 1e-10)
    normed_coords = (filtered_coords - coords_means.unsqueeze(0)) / coords_stds.unsqueeze(0)

    results_dict = {
        'data_target': data_target, # filtered outputs
        'data_inputs': data_inputs, # common
        'ls_of_ls_train_input': ls_of_ls_train_input,   # common
        'ls_of_ls_test_input': ls_of_ls_test_input,     # common
        'means': means,             # filtered outputs
        'stds': stds,               # filtered outputs
        'normed_coords': normed_coords, # filtered outputs
        'coords_means': coords_means,   # filtered outputs
        'coords_stds': coords_stds,      # filtered outputs
        # the following are for original outputs
        'all_coords': all_coords,
        'original_video': original_video 
    }

    return results_dict

def find_factors(m):
    """
    Find factor pairs of n, returning a list of possible dimensions for the matrices.
    """
    factors = [(3, 5), (5, 3), 
               (5, 2), (2, 5), 
               (3, 3), 
               (2, 2)]
    
    return factors

def place_rectangles(m, grid_size=100):
    """
    Place rectangles randomly on a grid until m squares are selected.

    :param m: number of grids to pick
    """
    selected = torch.zeros(grid_size, grid_size, dtype=torch.bool)
    factors = find_factors(m)
    count = 0  # Counter for selected squares

    while count < m:
        # Randomly choose a matrix size
        a, b = random.choice(factors)
        
        # Randomly determine the starting position of the matrix
        start_x = random.randint(0, grid_size - a)
        start_y = random.randint(0, grid_size - b)
        
        # Attempt to place the matrix
        for x in range(start_x, min(start_x + a, grid_size)):
            for y in range(start_y, min(start_y + b, grid_size)):
                if not selected[x, y] and count < m:
                    selected[x, y] = True
                    count += 1

                # Stop immediately if m squares are selected
                if count == m:
                    break
            if count == m:
                break

    return selected

def select_squares_for_frames(n_frames, m, grid_size=100):
    """
    Select m squares in each of the n_frames, resulting in a boolean tensor.
    
    Return:
        frames_selected: of shape (n_frames, grid_size, grid_size)
    """
    frames_selected = torch.zeros((n_frames, grid_size, grid_size), dtype=torch.bool)

    for i in range(n_frames):
        frames_selected[i] = place_rectangles(m, grid_size)

    return frames_selected

def helper_sample_locs_per_frame(video: torch.Tensor, num_locs_per_frame: int=100, sample_mode: str='rectangle', **kwargs):
    """
    for a given video, we sample a certain number of locs for each frame and used them as training indices, the rest of them are test indices. 

    The indices are organized as ls_of_ls.

    Return:
        masks4train: of shape (n_frames, grid_size, grid_size)
    """
    num_outputs, num_frames = video.shape[0], video.shape[1]
    n_grids = int(num_outputs**0.5)
    print(f'The video dataset has {num_frames} frames.\n')

    if sample_mode == 'random':
        random_2d_tensor = torch.randn_like(video) # (#outputs, #inputs)
        threshold_values, _ = torch.topk(random_2d_tensor, k=num_locs_per_frame, dim=0) # order from large to small
        threshold_values = threshold_values[-1, :]
        assert threshold_values.shape[0] == num_frames
        # NOTE masks4train serves as an extractor (for index filtering)
        # same shape as the video data, True/False represents whether it is selected for train/test.
        masks4train = random_2d_tensor >= threshold_values 
    
    elif sample_mode == 'rectangle':
        masks4train_ = select_squares_for_frames(num_frames, m=num_locs_per_frame, grid_size=n_grids)
        masks4train = masks4train_.reshape(num_frames, n_grids*n_grids).permute(1, 0) # (#outputs, #inputs)

        # NOTE make sure at least 2 points in each output ...
        print('Sampling locations for each frame: ')
        for id_output in range(num_outputs):
            if masks4train[id_output, :].sum() >= 2:
                pass

            else:
                # if the number of data points for certain output is less than 2: manually pick first and last point.
                masks4train[id_output, 0] = True
                masks4train[id_output, -1] = True
    else: 
        raise NotImplementedError
    
    ls_of_ls_train_input, ls_of_ls_test_input = [], []
    ls_num_training_samples = [] # a list of length num_outputs, which records the number of training samples per output

    for id_output in tqdm(range(num_outputs)):

        curr_train_id_list = torch.where(masks4train[id_output, :])[0].tolist() 
        curr_test_id_list = [x for x in range(num_frames) if x not in curr_train_id_list]
        ls_num_training_samples.append(len(curr_train_id_list))
        ls_of_ls_train_input.append(curr_train_id_list)
        ls_of_ls_test_input.append(curr_test_id_list)

    return masks4train, ls_of_ls_train_input, ls_of_ls_test_input, ls_num_training_samples

def prepare_simple_video(config):
    """
    Given an (n_frames, n_rows, n_columns) video, prepare into (#n_row*n_columns, n_frames) ready for multi-output modelling.

    where   (1). video path is specified in 'video_path'.
            (2). we are free to choose how to do train/test split: subsampling for each output or for each frame.
            (3). masks4train of shape (#outputs, #inputs).

    """
    n_rows, n_columns = config['n_grids'], config['n_grids']
    train_test_split_mode = 'mode2'

    video = torch.load(config['video_path']) # 2d tensor, (#outputs, #inputs)
    num_outputs,  n_inputs_per_output = video.shape[0], video.shape[1]
    assert num_outputs == config['n_outputs'] == n_rows*n_columns

    translate_bias = config['min_input_bound']
    translate_scale = (config['max_input_bound'] - config['min_input_bound']) / n_inputs_per_output
    data_inputs =  translate_bias + translate_scale * ( torch.tensor([i for i in range(n_inputs_per_output)]) )

    ls_of_ls_train_input, ls_of_ls_test_input = [], []

    # At the same time, compute statistics and standarize orignal data by statistics from train split ... 
    means, stds, data_target = torch.zeros(num_outputs), torch.zeros(num_outputs), torch.zeros_like(video)

    # Make random seeds ready to be used
    data_random_seed = config['data_random_seed'] if 'data_random_seed' in config else 1
    np.random.seed(data_random_seed)
    list_expri_random_seeds = np.random.randn(num_outputs)

    print('Prepare dataset: splitting train/test')

    # NOTE we are free to choice how to split train/test: subsampling for each pixel (output) or subsampling for each frame (image).
    if train_test_split_mode == 'mode1':
    # for every output, sample certain number of data point (n_input_we_want_per_output) as training points
        masks4train = None
        for id_output in tqdm(range(num_outputs)):
            # NOTE every output, we have sampled a certain number of data points
            random.seed(list_expri_random_seeds[id_output])
            train_index = random.sample(range(n_inputs_per_output), config['n_input_we_want_per_output'])
            within_test_index = [x for x in range(n_inputs_per_output) if x not in train_index]
            ls_of_ls_train_input.append(train_index)
            ls_of_ls_test_input.append(within_test_index)

            # Compute statistics and normalize data ... 
            means[id_output] = video[id_output, ls_of_ls_train_input[-1]].mean()
            stds[id_output] = video[id_output, ls_of_ls_train_input[-1]].std()

            data_target[id_output, :] = (video[id_output, :] - means[id_output]) / (stds[id_output] + 1e-8)

    elif train_test_split_mode == 'mode2':
    # for every frame, sample certain number of data points (num_locs_per_frame) as training points
        masks4train, ls_of_ls_train_input, ls_of_ls_test_input, ls_num_training_samples = helper_sample_locs_per_frame(video, num_locs_per_frame=config['num_locs_per_frame'])
        np_num_training_samples = np.array(ls_num_training_samples)
        n_train_per_output_min, n_train_per_output_average, n_train_per_output_max = np_num_training_samples.min(), np_num_training_samples.mean(), np_num_training_samples.max()

        print(f'The min/average/max number of training point for one outputs are: {n_train_per_output_min}; {n_train_per_output_average}; {n_train_per_output_max}.\n')

        for id_output in tqdm(range(num_outputs)):
            # NOTE make sure at lease 2 points in each output ... 
            if 'donot_normalize_dataset' not in config or config['donot_normalize_dataset'] == False:
                # Compute statistics and normalize data ... 
                means[id_output] = video[id_output, ls_of_ls_train_input[id_output]].mean()
                stds[id_output] = video[id_output, ls_of_ls_train_input[id_output]].std()

            elif config['donot_normalize_dataset'] == True:
                means[id_output], stds[id_output] = 0, 1
            
            data_target[id_output, :] = (video[id_output, :] - means[id_output]) / (stds[id_output] + 1e-8)
    
    # for video dataset, we are also interested in coordinates to describe every output (pixel, pixel)
    all_coords = torch.cartesian_prod(torch.arange(n_rows), torch.arange(n_columns)).float()
    coords_means, coords_stds = all_coords.mean(dim=0), (all_coords.std(dim=0, unbiased=False) + 1e-10)
    normed_coords = (all_coords - coords_means.unsqueeze(0)) / coords_stds.unsqueeze(0)

    results_dict = {
        'data_target': data_target, 
        'data_inputs': data_inputs, 
        'ls_of_ls_train_input': ls_of_ls_train_input,   
        'ls_of_ls_test_input': ls_of_ls_test_input,     
        'means': means,             
        'stds': stds,               
        'normed_coords': normed_coords, 
        'coords_means': coords_means,   
        'coords_stds': coords_stds,
        'masks4train': masks4train
    }

    return results_dict

def prepare_rotate_expanded_mnist_data(config):
    """
    Expand original 28*28 MNIST image to size of 100*100 via bicubic interpolation. And rotate it to get a simple video dataset (n_frames, 100, 100).

    One need to specify:
        mnist_digit_id: id of digit we interested in mnist train dataset. 

            Some examples: 
                        id=34,51 (number 0); 
                        id=23,40 (number 1); 
                        id=25 (number 2);
                        id=7,50 (number 3); 
                        id=58,61 (number 4);
                        id=236, (number 5);
                        id=36, 62 (number 6); 
                        id=15 (number 7); 
                        id=41,55 (number 8);
                        id=43,45 (number 9);   

        start_angle, end_angle: angles of first frame and last frame for rotating the digit.
        n_frames: how many frames we want, note that the rotating angles are evenly picked from start_angle to end_angle.
    """
    from scipy.ndimage import rotate

    ### STEP1: generate and store rotating mnist dataset
    n_grids = config['n_grids']
    transform = transforms.Compose([
        transforms.Resize((n_grids, n_grids), interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.ToTensor()
    ])

    train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    # test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    image, label = train_dataset[config['mnist_digit_id']]
    image = image.numpy()

    # image is in form (C, H, W) we need to reform to (H, W, C). 
    # we further delete last dim, which is channel dim
    image = image.transpose((1, 2, 0)).squeeze()
    start_angle, end_angle = config['start_angle'], config['end_angle']
    num_frames = config['n_frames']

    folder_path = '/Users/jiangxiaoyu/Desktop/All_Projects/Scalable_LVMOGP_v3/data/rotate_expanded_mnist'
    video_path = f'{folder_path}/healing_mnist_id_{config["mnist_digit_id"]}_ngrids_{n_grids}_label_{label}_{start_angle}_{end_angle}_{num_frames}.pt'

    if not os.path.exists(video_path):
        rotated_images = []

        # NOTE the angles are evenly space from start_angle to end_angle
        angles = np.linspace(start_angle, end_angle, num_frames)

        for angle in angles:
            rotated = rotate(image, angle, reshape=False)
            rotated_images.append(rotated)

        rotated_images_np = np.stack(rotated_images)
        rotated_images_np_torch = torch.tensor(rotated_images_np).reshape(num_frames, -1).permute(1, 0)

        torch.save(rotated_images_np_torch, video_path)

    else: # dataset already exists
        pass

    ### STEP2: leverage (reuse) prepare_simple_video func
    config['video_path'] = video_path
    print('NOTE we do NOT normalize dataset!!!')
    config['donot_normalize_dataset'] = True
    results_dict = prepare_simple_video(config)

    return results_dict

def prepare_rotate_expanded_mnist_integer_data(config):
    """
    Similar to prepare_rotate_expanded_mnist_data, but keep all pixel values all 0-255 integers.
    """
    from scipy.ndimage import rotate
    from PIL import Image

    train_dataset = datasets.MNIST(root='./data', train=True, download=True)
    img, label = train_dataset.data[config['mnist_digit_id']], train_dataset.targets[config['mnist_digit_id']]

    n_grids = config['n_grids']
    pil_img = Image.fromarray(img.numpy()).resize((n_grids, n_grids), Image.NEAREST)
    image = np.array(pil_img)

    start_angle, end_angle = config['start_angle'], config['end_angle']
    num_frames = config['n_frames']

    folder_path = '/Users/jiangxiaoyu/Desktop/All_Projects/Scalable_LVMOGP_v3/data/rotate_expanded_mnist_integer'
    video_path = f'{folder_path}/healing_mnist_id_{config["mnist_digit_id"]}_ngrids_{n_grids}_label_{label}_{start_angle}_{end_angle}_{num_frames}.pt'

    if not os.path.exists(video_path):
        rotated_images = []
        angles = np.linspace(start_angle, end_angle, num_frames)

        for angle in angles:
            rotated = rotate(image, angle, reshape=False, order=1, mode='nearest')
            rotated_clipped = np.clip(rotated, 0, 255).astype(np.uint8)
            rotated_images.append(rotated_clipped)

        rotated_images_np = np.stack(rotated_images)
        rotated_images_np_torch = torch.tensor(rotated_images_np).reshape(num_frames, -1).permute(1, 0)

        torch.save(rotated_images_np_torch, video_path)

    else: # dataset already exists
        pass

    ### STEP2: leverage (reuse) prepare_simple_video func
    config['video_path'] = video_path
    config['donot_normalize_dataset'] = True    # This MUST be true! We need to keep integers.
    results_dict = prepare_simple_video(config)

    return results_dict

def prepare_USHCN_data(config):
    """
    Dirctly using post-processed data from the paper:
        GRU-ODE-Bayes: Continuous modeling of sporadically-observed time series 2019, NeurIPS.
    
    We treat every variables for any station as an output. For time < 150, training data. We test on first 3 observations after time=150 for every output.
    """
    data_pd = pd.read_csv(config['USHCN_data_path'])

    # Re-organize the dataframe: every row is an output, column index (Time) become inputs.
    value_columns = ['Value_0', 'Value_1', 'Value_2', 'Value_3', 'Value_4']
    mask_columns = ['Mask_0', 'Mask_1', 'Mask_2', 'Mask_3', 'Mask_4']

    for v, m in zip(value_columns, mask_columns):
        data_pd[v] = data_pd.apply(lambda x: x[v] if x[m] == 1 else np.nan, axis=1)

    melted_df = data_pd.melt(id_vars=['ID', 'Time'], value_vars=value_columns, var_name='Variable', value_name='Value')
    pivot_df = melted_df.pivot_table(index=['ID', 'Variable'], columns='Time', values='Value', aggfunc='first')

    data_inputs = torch.tensor(pivot_df.iloc[0, :].index.tolist())

    train_df = pivot_df.iloc[:, :1118] # Time<150 are training, first 3 element after Time=150 are for test
    other_df = pivot_df.iloc[:, 1118:]

    # print('Total number of training data points is: ', train_df.notna().to_numpy().sum(), '\n')
    # Reindex columns, so that the indices are integers 0,1,2... etc
    # NOTE 
    train_df.columns = range(train_df.shape[1]) 
    other_df.columns = range(1118, 1118+other_df.shape[1])

    # NOTE we decide to abandon some outputs with insufficient data points.
    # but len(ls_num_training_samples) should be equal to pivot_df.shape[0]
    ls_of_ls_train_input, ls_of_ls_test_input = [], []
    ls_num_training_samples = []

    num_train_data_abandon = 0
    for row_id in range(pivot_df.shape[0]):
        train_non_nan_columns = train_df.iloc[row_id, :].notna()
        train_indices = list(train_non_nan_columns[train_non_nan_columns].index)
        if len(train_indices) < 2:
            # There is not enough training data point (at least 2), thus we abandon this output ... 
            num_train_data_abandon += len(train_indices)
            ls_num_training_samples.append(0)
            pass

        else:
            ls_of_ls_train_input.append(train_indices)
            ls_num_training_samples.append(len(train_indices))

            test_non_nan_columns = other_df.iloc[row_id, :].notna()
            test_indices = list(test_non_nan_columns[test_non_nan_columns].index)[:3] # only first 3 observations are used for testing ...

            ls_of_ls_test_input.append(test_indices)

    print('Total number of training data points is: ', train_df.notna().to_numpy().sum() - num_train_data_abandon, '\n')
    print(f'The number of output is: ', len(ls_of_ls_train_input), '\n')
    assert len(ls_of_ls_train_input) == len(ls_of_ls_test_input) == config['n_outputs']
    assert len(ls_num_training_samples) == pivot_df.shape[0]
    assert config['n_outputs'] < len(ls_num_training_samples)

    # np_num_training_samples = np.array(ls_num_training_samples)
    # n_train_per_output_min, n_train_per_output_average, n_train_per_output_max = np_num_training_samples.min(), np_num_training_samples.mean(), np_num_training_samples.max()
    # print(f'The min/average/max number of training point for one outputs are: {n_train_per_output_min}; {n_train_per_output_average}; {n_train_per_output_max}.\n')
    
    # At the same time, compute statistics and standarize orignal data by statistics from train split ... 
    pivot_df = torch.tensor(pivot_df.to_numpy())
    means, stds, data_target = torch.zeros(config['n_outputs']), torch.zeros(config['n_outputs']), torch.zeros(config['n_outputs'], data_inputs.shape[0])

    valid_output_id = 0 # which valid output we are considering now ... 
    for id_output in tqdm(range(len(ls_num_training_samples))):
        # Compute statistics and normalize data ... 
        if ls_num_training_samples[id_output] == 0:
            pass

        else:
            means[valid_output_id] = pivot_df[id_output, ls_of_ls_train_input[valid_output_id]].mean()
            stds[valid_output_id] = pivot_df[id_output, ls_of_ls_train_input[valid_output_id]].std()
            data_target[valid_output_id, :] = (pivot_df[id_output, :] - means[valid_output_id]) / (stds[valid_output_id] + 1e-8)

            valid_output_id += 1
    
    assert valid_output_id == config['n_outputs']

    results_dict = {'data_inputs': data_inputs,
                    'data_target': data_target,
                    'ls_of_ls_train_input': ls_of_ls_train_input,
                    'ls_of_ls_test_input': ls_of_ls_test_input,
                    'means': means,
                    'stds': stds}
    
    return results_dict

'''
def prepare_ST_human_Prostate_Cancer_data(config):
    """
    Spatial Transcriptomics data. Every gene is regarded as an output, equiped with a latent variable.
    This is a count dataset (NOTE we donot normalize this dataset): input is the spatial coordinate (2D), target value is a count number (integer) for each output.
    """
    import scanpy as sc

    # read ST data
    adata = sc.read_visium(path = '/Users/jiangxiaoyu/Desktop/All_Projects/Scalable_LVMOGP_v3/data/ST_human_Prostate_Cancer/', 
                            count_file = 'Visium_FFPE_Human_Prostate_Cancer_filtered_feature_bc_matrix.h5', 
                            library_id = 'A1_spot',                        
                            load_images=True)
    
    adata.var_names_make_unique()
    adata.var['SYMBOL'] = adata.var_names

    data_target = torch.tensor(adata.X.A).T               # (num_of_genes, num_of_locs)
    gene_names = adata.var_names.to_numpy()
    locs_data = torch.tensor(adata.obsm['spatial'])       # this is the unnormalized (original) loc data

    # We may want to filter out spatially less variable genes, this is done by sorting variance of the count data for genes
    data_target_std = torch.std(data_target, dim=1)
    sorted_indices = torch.argsort(-data_target_std)
    top_variable_row_ids = sorted_indices[:config['num_top_variable_genes']]

    data_target = data_target[top_variable_row_ids]
    gene_names = gene_names[top_variable_row_ids]

    assert locs_data.shape[1] == config['input_dim']
    assert data_target.shape[0] == gene_names.shape[0] == config['n_outputs']
    assert data_target.shape[1] == locs_data.shape[0]

    # normalize the inputs locs (normalize them altogether, no matter train/test)
    locs_data = locs_data.to(torch.double)
    locs_data_means, locs_data_std = locs_data.mean(axis=0), locs_data.std(axis=0)
    normed_locs_data = (locs_data - locs_data_means) / locs_data_std
    data_inputs = normed_locs_data
    num_all_inputs = data_inputs.shape[0] 
    assert config['n_train_inputs'] < num_all_inputs

    input_ids_full_list = [i for i in range(num_all_inputs)]
    ls_of_ls_train_input, ls_of_ls_test_input = [], []

    for output_id in tqdm(range(config['n_outputs'])):
        train_indices = torch.randperm(len(input_ids_full_list))[:config['n_train_inputs']].tolist()
        test_indicies = [j for j in input_ids_full_list if j not in train_indices]
        ls_of_ls_train_input.append(train_indices)
        ls_of_ls_test_input.append(test_indicies)

    # NOTE we donot normalize target data
    means, stds = torch.zeros(config['n_outputs']), torch.ones(config['n_outputs'])

    results_dict = {'data_inputs': data_inputs,
                    'data_target': data_target,
                    'ls_of_ls_train_input': ls_of_ls_train_input,
                    'ls_of_ls_test_input': ls_of_ls_test_input,
                    'means': means,
                    'stds': stds,
                    'gene_names': gene_names,
                    'locs_means': locs_data_means,
                    'locs_stds': locs_data_std}
    
    return results_dict
'''

def prepare_ST_human_Prostate_Cancer_data(config):
    """
    Spatial Transcriptomics data. Every gene is regarded as an output, equiped with a latent variable.
    This is a count dataset (NOTE we donot normalize this dataset): input is the spatial coordinate (2D), target value is a count number (integer) for each output.
    """
    import scanpy as sc

    # read ST data
    adata = sc.read_visium(path = '/Users/jiangxiaoyu/Desktop/All_Projects/Scalable_LVMOGP_v3/data/ST_human_Prostate_Cancer/', 
                            count_file = 'Visium_FFPE_Human_Prostate_Cancer_filtered_feature_bc_matrix.h5', 
                            library_id = 'A1_spot',                        
                            load_images=True)
    
    adata.var_names_make_unique()
    adata.var['SYMBOL'] = adata.var_names

    sc.pp.normalize_total(adata, inplace=True)
    sc.pp.log1p(adata)
    sc.pp.highly_variable_genes(adata, flavor='seurat', n_top_genes=config['num_top_variable_genes'])

    # Pick only 5,000 highly variable genes
    adata_red = adata[:, adata.var['highly_variable']]

    # Stupid things TODO 
    adata_2 = sc.read_visium(path = '/Users/jiangxiaoyu/Desktop/All_Projects/Scalable_LVMOGP_v3/data/ST_human_Prostate_Cancer/', 
                       count_file='Visium_FFPE_Human_Prostate_Cancer_filtered_feature_bc_matrix.h5', 
                       library_id='A1_spot',                        
                       load_images=True)

    adata_2.var_names_make_unique()
    adata_2.var['SYMBOL'] = adata_2.var_names

    adata_2 = adata_2[:,adata_2.var_names.isin(adata_red.var_names)]
    sc.pp.normalize_total(adata_2, inplace=True)
    all_data_target =  torch.tensor(np.round(adata_2.X.A)).T    # (5000, num_of_locs)

    gene_names = adata_2.var_names.to_numpy()
    locs_data = torch.tensor(adata_2.obsm['spatial'])       # this is the unnormalized (original) loc data

    assert locs_data.shape[1] == config['input_dim']
    assert all_data_target.shape[0] == gene_names.shape[0] == config['n_outputs']
    assert all_data_target.shape[1] == locs_data.shape[0]

    # The following is stupid TODO!
    # normalize all the inputs locs (normalize them altogether, no matter train/test)
    locs_data = locs_data.to(torch.double)
    locs_data_means, locs_data_std = locs_data.mean(axis=0), locs_data.std(axis=0)
    normed_locs_data = (locs_data - locs_data_means) / locs_data_std

    data_inputs = normed_locs_data
    data_target = all_data_target

    '''
    # Optional: only focus on a certain region of the tissue
    small_region_input_locs = torch.tensor([])
    small_region_input_loc_ids = []
    for id, loc in enumerate(normed_locs_data):
        if loc[0] > 1 and loc[1] > 1:
            small_region_input_loc_ids.append(id)
            small_region_input_locs = torch.concatenate([small_region_input_locs, loc], dim=0)

    data_target = all_data_target[:, small_region_input_loc_ids]

    small_region_input_locs = small_region_input_locs.reshape(-1, 2)
    # Re-normalize the small region input locations
    small_region_locs_data_means, small_region_locs_data_std = small_region_input_locs.mean(axis=0), small_region_input_locs.std(axis=0)
    small_region_input_locs = (small_region_input_locs - small_region_locs_data_means) / small_region_locs_data_std

    print('Smaller region of tissue has shape: ', small_region_input_locs.shape)
    data_inputs = small_region_input_locs
    '''
    
    num_all_inputs = data_inputs.shape[0] 
    assert config['n_train_inputs'] <= num_all_inputs
    

    input_ids_full_list = [i for i in range(num_all_inputs)]
    ls_of_ls_train_input, ls_of_ls_test_input = [], []

    for output_id in tqdm(range(config['n_outputs'])):
        train_indices = torch.randperm(len(input_ids_full_list))[:config['n_train_inputs']].tolist()
        test_indicies = [j for j in input_ids_full_list if j not in train_indices]
        ls_of_ls_train_input.append(train_indices)
        ls_of_ls_test_input.append(test_indicies)

    # NOTE we donot normalize target data
    means, stds = torch.zeros(config['n_outputs']), torch.ones(config['n_outputs'])

    results_dict = {'data_inputs': data_inputs,
                    'data_target': data_target,
                    'ls_of_ls_train_input': ls_of_ls_train_input,
                    'ls_of_ls_test_input': ls_of_ls_test_input,
                    'means': means,
                    'stds': stds,
                    'gene_names': gene_names,
                    'locs_means': locs_data_means,
                    'locs_stds': locs_data_std}
    
    return results_dict

def prepare_NYC_Crime_Count_large(config):
    """
    NYC Crime Count dataset, from Spatio-Temporal Variational Gaussian Processes, 2021 NeurIPS.
    We follows exactly the same spilts of their paper: 5 fold cross-validation.

    This dataset have Ns=447 outputs, Nt=182 inputs, in total 81,354 data points. No missing values. This is count data, we donot perform normalization 
    """

    # Read entire dataset 
    data_df = pd.read_csv('/Users/jiangxiaoyu/Desktop/All_Projects/Scalable_LVMOGP_v3/data/nyc_crime_counts/processed_data/data_df.csv')

    # pivoting, re-organize the dataset
    pivot_Y = data_df.pivot_table(index=['x1', 'x2'], columns='Time', values='Y')

    # CV id, which cross-validation split we are going to use
    CV_id = config['CV_id']
    the_value = f'train_split{CV_id}'
    # test data are masked
    pivot_mask = data_df.pivot_table(index=['x1', 'x2'], columns='Time', values=the_value)

    _data_inputs = torch.tensor(pivot_Y.columns)
    _data_inputs = (_data_inputs - _data_inputs.min()) / (_data_inputs.max() - _data_inputs.min() + 1e-8) # transform to range [0, 1]
    data_inputs = _data_inputs * (config['max_input_bound'] - config['min_input_bound']) + config['min_input_bound']

    data_target = torch.tensor(pivot_Y.to_numpy())

    assert 447 == data_target.shape[0] == config['n_outputs']
    assert 182 == data_target.shape[1] == config['n_inputs'] == data_inputs.shape[0]

    ls_of_ls_train_input, ls_of_ls_test_input = [], []
    for _, row in pivot_mask.iterrows():
        true_indices = [i for i, value in enumerate(row) if value]
        false_indices = [i for i, value in enumerate(row) if value==False]

        ls_of_ls_train_input.append(true_indices)
        ls_of_ls_test_input.append(false_indices)
    
    # NOTE we donot normalize target data
    means, stds = torch.zeros(config['n_outputs']), torch.ones(config['n_outputs'])

    x1_x2_locs = pivot_Y.index.to_numpy() # 447 tuples in numpy array
    x1_x2_locs = np.array([list(item) for item in x1_x2_locs])
    x1_x2_locs = torch.tensor(x1_x2_locs)
    x1_x2_locs = (x1_x2_locs - x1_x2_locs.mean(axis=0)) / x1_x2_locs.std(axis=0)

    results_dict = {'data_inputs': data_inputs,
                    'data_target': data_target,
                    'ls_of_ls_train_input': ls_of_ls_train_input,
                    'ls_of_ls_test_input': ls_of_ls_test_input,
                    'means': means,
                    'stds': stds,
                    'x1_x2_locs': x1_x2_locs,
                    }
    
    return results_dict

####################  Sampling/Picking function used duing training  ####################

def identify_ids_of_nearest_elements(scores: Tensor, chosen_element_id: int, num_nearest_elements: int):
    """
    We has a batch of elements, each has a score. Given the id of an element, pick some elements (and their ids) that are the closest to the element with id chosen_element_id.

    :param scores: the scores for the elements
    :param chosen_element_id: the id of the chosen element, for which we are going to find nearst elements
    :param num_nearest_elements: how many elements we are searching for 

    Return:
        elements: List
        ids: List

    For instance 1:
        scores = torch.Tensor([1, 6, 3, 8, 9, 12])
        chosen_element_id = 3
        num_nearest_elements = 2

        we shoud return 
            elements = [9, 6]
            ids = [4, 1]

    Another instance:

        scores = torch.Tensor([7, 3, 0, 2])
        chosen_element_id = 0
        num_nearest_elements = 1

        we shoud return 
            elements = [3]
            ids = [1]

    """
    assert num_nearest_elements < scores.shape[0]

    # Get the score for the chosen element
    target_score = scores[chosen_element_id]
    
    # abs difference
    differences = torch.abs(scores - target_score)
    
    # 
    # Use torch.topk, largest=False to pick smaller elements
    values, indices = torch.topk(differences, num_nearest_elements + 1, largest=False, sorted=True)
    
    # !! exclude the chosen element
    nearest_ids = []
    nearest_scores = []
    for idx, value in zip(indices, values):
        if idx != chosen_element_id:
            nearest_ids.append(int(idx))
            nearest_scores.append(scores[idx])
        if len(nearest_ids) == num_nearest_elements:
            break
    
    return nearest_scores, nearest_ids

class ReservoirSampler():
    """
    This is an Sampler used for mini-batching.
    We group the outputs into clusters, in each mini-batch iteration, we sample a sample from each cluster without replacement. Resetting clusters until them becoming empty.
    """
    def __init__(self, ls_of_ls_inputs: list, batch_size_output: int, batch_size_input: int, scores: Tensor, **kwargs):
        """
        
        :param ls_of_ls_inputs: list of list of inputs, len(ls_of_ls_inputs)=#outputs, ls_of_ls_inputs[0]=#inputs for first output.
        """
        self.ls_of_ls_inputs = ls_of_ls_inputs
        self.batch_size_output = batch_size_output
        self.batch_size_input = batch_size_input
        self.scores = scores

        ## -------------------------------------------------------------
        self.generate_clusters_for_outputs()
        # Assign a np.array of ids for each group.
        self.output_groups_indices = [np.arange(len(lst)) for lst in self.output_groups]
        self.output_groups_indices_reset()

    def sample(self):
        """
        """
        outputs_ids = self.sample_output_from_every_cluster()
        assert len(outputs_ids) == self.batch_size_output

        output_id_list = [x for x in outputs_ids for _ in range(self.batch_size_input)]
        input_id_list = self.sample_inputs_for_each_output(outputs_ids=outputs_ids)

        assert len(output_id_list) == len(input_id_list) == self.batch_size_output * self.batch_size_input

        return output_id_list, input_id_list

    def generate_clusters_for_outputs(self):
        """
        Generate len(scores)/batch_size_output clusters, each time, sample one sample from each cluster.
        """
        sorted_data, sorted_indices = torch.sort(self.scores)

        cluster_size = int(len(self.scores) / self.batch_size_output)

        groups = []

        for i in range(self.batch_size_output):
            start_index = i * cluster_size
            
            if i == self.batch_size_output - 1:
                end_index = len(self.scores)
            else:
                end_index = start_index + cluster_size

            groups.append(sorted_indices[start_index:end_index].tolist())

        assert len(groups) == self.batch_size_output

        # list of list, totally batch_size_output lists.
        self.output_groups = groups

    def sample_output_from_every_cluster(self, n:int=1):
        """
        Sample (n=1) one sample from each cluster (without replacement). Totally batch_size_output samples.
        
        Return:
            list, of size batch_size_output
        """
        sampled_items = []
        for i, idx in enumerate(self.output_groups_indices):
            if len(idx) < n:
                self.output_groups_indices_reset(i)
                idx = self.output_groups_indices[i]

            choices = np.random.choice(idx, size=n, replace=False).tolist()
            sampled_items.append(int(self.output_groups[i][choices[0]])) # by default n=1
            # Remove the ids of the selected elements
            self.output_groups_indices[i] = np.setdiff1d(idx, choices)

        return sampled_items

    def sample_inputs_for_each_output(self, outputs_ids:list):
        """
        Sample batch_size_input samples for each output specified in outputs_ids (sampling with replacement)
        """
        assert len(outputs_ids) == self.batch_size_output

        # Use list comprehension for input_id_list
        input_id_list = [j for i in outputs_ids for j in random.choices(self.ls_of_ls_inputs[i], k=self.batch_size_input)]

        return input_id_list
    
    def output_groups_indices_reset(self, index=None):
        if index is None:
            self.output_groups_indices = [np.arange(len(lst)) for lst in self.output_groups]
        else:
            self.output_groups_indices[index] = np.arange(len(self.output_groups[index]))


def sample_ids_with_careful_grouping_outputs(ls_of_ls_inputs: list, batch_size_output: int, batch_size_input: int, scores: Tensor):
    """
    Put outputs with similar scores in one mini-batch.
    """
    first_output_id = random.choices(range(len(ls_of_ls_inputs)), k=1)
    _, output_ids = identify_ids_of_nearest_elements(scores=scores, chosen_element_id=first_output_id, num_nearest_elements=batch_size_output-1)
    output_ids.append(first_output_id[0])

    assert len(output_ids) == batch_size_output
    output_id_list = [x for x in output_ids for _ in range(batch_size_input)]

    # Use list comprehension for input_id_list
    input_id_list = [j for i in output_ids for j in random.choices(ls_of_ls_inputs[i], k=batch_size_input)]

    # Recall, in kronecker variational strategy, two list of 'inputs' jointly determine the corresponding target. 
    assert len(output_id_list) == len(input_id_list) == batch_size_output * batch_size_input

    return output_id_list, input_id_list

def sample_ids_of_output_and_input(ls_of_ls_inputs: list, batch_size_output:int, batch_size_input:int):
    """
    Given ls_of_ls_inputs (list of list) containing available inputs for each latent (output).
        * len(ls_of_ls_inputs) == the number of all outputs.
        * ls_of_ls_inputs[i] refers to the list of available indices for (i+1) th output.
    
    :param batch_size_output: the number of elements sampled from all outputs.
    :param batch_size_input: the number of elements sampled from all available inputs.

    Return:
        output_id_list (list): selected_output_ids
        input_id_list (list): selected_input_ids
    """

    output_ids = random.choices(range(len(ls_of_ls_inputs)), k=batch_size_output)
    output_id_list = [x for x in output_ids for _ in range(batch_size_input)]
    
    # Use list comprehension for input_id_list
    input_id_list = [j for i in output_ids for j in random.choices(ls_of_ls_inputs[i], k=batch_size_input)]

    # Recall, in kronecker variational strategy, two list of 'inputs' jointly determine the corresponding target. 
    assert len(output_id_list) == len(input_id_list) == batch_size_output * batch_size_input

    return output_id_list, input_id_list

def sample_single_output_and_batch_input(ls_of_ls_inputs: list, batch_size_input:int):
    """
    Randomly pick one output, and pick batch_size_input of inputs for that.
    """
    single_output_id = [random.choice(range(len(ls_of_ls_inputs)))]
    output_id_list = [x for x in single_output_id for _ in range(batch_size_input)]

    # Use list comprehension for input_id_list
    input_id_list = [j for i in single_output_id for j in random.choices(ls_of_ls_inputs[i], k=batch_size_input)]

    # Recall, in kronecker variational strategy, two list of 'inputs' jointly determine the corresponding target. 
    assert len(output_id_list) == len(input_id_list) == 1 * batch_size_input

    return output_id_list, input_id_list

def getY_from_ids_of_output_input(batch_index_output: list, batch_index_input: list, data_target: Tensor):
    """
    2 lists: batch_index_output and batch_index_input jointly determine a target Y. This function aims to pick them 
    from data_target, and return as a tensor.

    :param batch_index_output (list): a list of indices of outputs
    :param batch_index_input (list): a list of indices of inputs
    :data_target (2D tensor): of shape num_output * num_inputs

    Return: 
        batch_Y (tensor)
    """

    assert len(batch_index_output) == len(batch_index_input)
    batch_Y = torch.zeros(len(batch_index_output))
    for i, (id_output, id_input) in enumerate(zip(batch_index_output, batch_index_input)):
        batch_Y[i] = data_target[id_output, id_input]
    
    return batch_Y


####################   Functions used in prediction time    ####################
@torch.no_grad()
def conditional_pred_Y_test(mean_test: Tensor, mean_ref: Tensor, covar_test_ref: Tensor, covar_ref_ref: Tensor, covar_test_test: Tensor, Y_ref: Tensor):
        """
        p(Y_test | Y_ref) = Normal(K_test_ref K_ref_ref^{-1} Y_ref, K_test_test - K_test_ref K_ref_ref^{-1} K_ref_test)

        Using conditional gaussian formula, we obtain predictive distribution with mean and covar_matrix.

        :param covar_test_ref: covariance matrix between test and ref data.
        :param covar_ref_ref: covariance matrix between ref and ref data itself.
        :param covar_test_test: covariance matrix between test and test data itself.
        :param Y_ref: observed reference data.
        """
        inv_covar_ref_ref = torch.linalg.inv(covar_ref_ref)
        mean = mean_test + covar_test_ref @ inv_covar_ref_ref @ (Y_ref - mean_ref)

        covariance_matrix = covar_test_test - covar_test_ref @ inv_covar_ref_ref @ covar_test_ref.T

        return MultivariateNormal(mean=mean, covariance_matrix=covariance_matrix)

@torch.no_grad() # TODO improve this func
def prediction_on_test_outputs_with_reference_data(my_model, data_inputs: Tensor, ls_of_ls_ref_input: list, ls_of_ls_test_input: list, data_target: Tensor, **kwargs):
    """
    LVMOGP model predictions on certain outputs and certain inputs. (identified by ls_of_ls_test_input)
    This prediction is based on means insteads on samplings. Making predictions based on (well trained) model parameters and reference data.

    :param data_inputs: of shape (#output, #inputs)
    :param ls_of_ls_ref_input, ls_of_ls_test_input: list of list of ids, ids MUST for data_inputs.
    :param data_target: of shape (#outputs, #inputs)

    Return:
        NOTE Although the returns has #output rows, only test outputs contains meaningful values, others are just nan.
        for test_output, only test inputs have meaningful predictions, others are just nan.
        pred_mean:  of shape (#outputs, #inputs)
        pred_std: of shape (#outputs, #inputs)
    """

    my_model.eval()

    num_inputs = data_inputs.shape[0]
    Q, num_outputs, latent_dim = my_model.H.q_mu.shape
    latents = my_model.H.q_mu.data # ONLY ref_means are used !

    # Sanity checks --- all test outputs has the same number of test points
    whether_first = True
    common_length = -1
    for i in range(num_outputs):
        if len(ls_of_ls_test_input[i]) != 0:
            if whether_first:
                common_length = ls_of_ls_test_input[i]
                whether_first = False
            else:
                assert common_length == ls_of_ls_test_input[i]


    pred_mean = torch.ones(num_outputs, num_inputs) * torch.nan
    pred_std = torch.ones(num_outputs, num_inputs) * torch.nan

    for i in range(num_outputs): 

        if len(ls_of_ls_ref_input[i]) == 0:
            # NO test data for current output ... 
            assert len(ls_of_ls_test_input[i]) == 0
            continue

        curr_latent = latents[:, i:(i+1), :].expand(-1, num_inputs, -1)
        pred_f = my_model(latents=curr_latent, inputs=data_inputs)
        task_indices = torch.tensor([i for _ in range(num_inputs)])
        pred_y = my_model.likelihood(pred_f, task_indices=task_indices)
        complete_mean = pred_y.mean.detach()
        complete_covar = pred_y.lazy_covariance_matrix.detach()

        # Pick elements by indexing ... 
        ref_index = torch.tensor(ls_of_ls_ref_input[i])
        test_index = torch.tensor(ls_of_ls_test_input[i])

        # split mean
        mean_ref, mean_test = complete_mean[ref_index], complete_mean[test_index]

        # split covar: covar_test_ref, covar_ref_ref, covar_test_test
        index_row1, index_col1 = torch.meshgrid(test_index, ref_index, indexing='ij')
        covar_test_ref = complete_covar[index_row1, index_col1]

        index_row2, index_col2 = torch.meshgrid(ref_index, ref_index, indexing='ij')
        covar_ref_ref = complete_covar[index_row2, index_col2]

        index_row3, index_col3 = torch.meshgrid(test_index, test_index, indexing='ij')
        covar_test_test = complete_covar[index_row3, index_col3]

        Y_ref = data_target[i][ref_index]

        Y_test_pred = conditional_pred_Y_test(mean_test=mean_test, mean_ref=mean_ref, 
                                              covar_test_ref=covar_test_ref, covar_ref_ref=covar_ref_ref, covar_test_test=covar_test_test, Y_ref=Y_ref)

        # Store prediction results
        pred_mean[i][test_index] = Y_test_pred.mean.detach()
        pred_std[i][test_index] = Y_test_pred.stddev.detach()

    return pred_mean, pred_std

@torch.no_grad()
def single_output_prediction_with_reference_data(my_model, output_id: int, inputs: Tensor, reference_data: dict, **kwargs):
    """
    LVMOGP prediction (with mean approach) for inputs at a certain output (output_id) with reference dataset. 

    :param my_model: well trained LVMOGP model.
    :param output_id: the index of output we are interested in making predictions.
    :param inputs: for which we are interested in making predictions, of shape (#inputs,)
    :param reference_data: a reference dataset in dict format, keys are ref inputs, values are corresponding ref targets.
        # NOTE, keys and values MUST be float(int) numbers.

    Return:
        Y_test_pred.mean.detach(): of shape (#inputs,)
        Y_test_pred.stddev.detach(): of shape (#inputs,)
    """
    my_model.eval()

    latents = my_model.H.q_mu.data 
    # ref inputs are put at the end of the tensor.
    num_inputs = inputs.shape[0]
    num_ref = len(list(reference_data.keys()))
    all_inputs = torch.cat((inputs, torch.tensor(list(reference_data.keys()))), axis=0)
    num_all_inputs = all_inputs.shape[0]
    assert num_all_inputs == num_inputs + num_ref

    # ref target
    ref_targets = torch.tensor(list(reference_data.values()))

    curr_latent = latents[:, output_id:(output_id+1), :].expand(-1, num_all_inputs, -1)
    pred_f = my_model(latents=curr_latent, inputs=all_inputs)
    task_indices = torch.tensor([output_id for _ in range(num_all_inputs)])
    pred_y = my_model.likelihood(pred_f, task_indices=task_indices)
    complete_mean = pred_y.mean.detach()
    complete_covar = pred_y.lazy_covariance_matrix.to_dense().detach()
    
    # print(torch.isclose(pred_f.lazy_covariance_matrix.to_dense(), complete_covar))

    # Pick elements by indexing ... 
    inputs_index = torch.tensor([i for i in range(num_inputs)])
    ref_index = torch.tensor([j + num_inputs for j in range(num_ref)])

    # split mean
    mean_ref, mean_test = complete_mean[ref_index], complete_mean[inputs_index]

    # split covar: covar_test_ref, covar_ref_ref, covar_test_test
    index_row1, index_col1 = torch.meshgrid(inputs_index, ref_index, indexing='ij')
    covar_test_ref = complete_covar[index_row1, index_col1]

    index_row2, index_col2 = torch.meshgrid(ref_index, ref_index, indexing='ij')
    covar_ref_ref = complete_covar[index_row2, index_col2]

    index_row3, index_col3 = torch.meshgrid(inputs_index, inputs_index, indexing='ij')
    covar_test_test = complete_covar[index_row3, index_col3]

    Y_test_pred = conditional_pred_Y_test(mean_test, mean_ref, covar_test_ref, covar_ref_ref, covar_test_test, ref_targets)

    return Y_test_pred.mean.detach(), Y_test_pred.stddev.detach()

@torch.no_grad()
def prediction_with_means(my_model, data_inputs:Tensor, task_index_list:list = None, num_f_samples:int=10, **kwargs):
    """
    LVMOGP model predictions (with Gaussian Likelihood).

    Make predictions on (unseen) (input, output) pair, ONLY use ref_means of q(H).
    If task_index_list == None, we make predictions for all outputs, otherwise only make prediction for one output per input.

    :param my_model: well trained model
    :param data_inputs: (tensor) of shape (#inputs, #features)
    :param task_index_list: (list) of task ids, if not None, must has same length as data_inputs

    Return:
        pred_mean: tensor of shape (#output, #inputs)
        pred_std: tensor of shape (#output, #inputs)
        pred_f_samples: tensor of shape (#num_f_samples, #output, #inputs)
    """
    my_model.eval()
    # Sanity checks
    if task_index_list != None:
        assert len(task_index_list) == data_inputs.shape[0]
    
    assert my_model.H.q_mu.dim() == 3

    num_inputs = data_inputs.shape[0]
    Q, num_outputs, latent_dim = my_model.H.q_mu.shape
    latents = my_model.H.q_mu.data # ONLY ref_means are used !
    num_f_samples = num_f_samples # How many functions y(x) we sample 

    if task_index_list == None:
        # Prepare empty tensors for storing results
        pred_mean = torch.zeros(num_outputs, num_inputs)
        pred_std = torch.zeros(num_outputs, num_inputs)
        pred_f_samples = torch.zeros(num_f_samples, num_outputs, num_inputs)

        for i in tqdm(range(num_outputs)): 
            curr_latent = latents[:, i:(i+1), :].expand(-1, num_inputs, -1)
            pred_f = my_model(latents=curr_latent, inputs=data_inputs)
            task_indices = torch.tensor([i for _ in range(num_inputs)])
            pred_y = my_model.likelihood(pred_f, task_indices=task_indices)
            pred_mean[i, :] = pred_y.mean.detach()
            pred_std[i, :] = pred_y.stddev.detach()   
            for k in range(10):   
                # Generate 10 function samples for i-output
                pred_f_samples[k, i, :] = pred_f.sample().detach()
    
    else:
        raise NotImplementedError

    return pred_mean, pred_std, pred_f_samples

@torch.no_grad()
def prediction_to_get_f_distribution(my_model, data_inputs:Tensor, **kwargs):
    """
    We only want to get q(f) at test locations. This is independent of the choice of likelihood.

    Return:
        list_pred_f: list of q(f) (MultivariateNormal), of length #outputs. 
    """
    my_model.eval()

    num_inputs = data_inputs.shape[0]
    Q, num_outputs, latent_dim = my_model.H.q_mu.shape
    latents = my_model.H.q_mu.data # ONLY ref_means are used !
    
    assert 'log_mean_express_level' in kwargs
    list_pred_f = []

    for i in tqdm(range(num_outputs)): 
        curr_latent = latents[:, i:(i+1), :].expand(-1, num_inputs, -1)
        pred_f = my_model(latents=curr_latent, inputs=data_inputs)

        pred_f.loc += kwargs['log_mean_express_level'][i]
        
        list_pred_f.append(pred_f)
    
    return list_pred_f


@torch.no_grad()
def prediction_with_means_possion_likelihood(my_model, data_inputs:Tensor, num_f_samples:int=2, **kwargs):
    """
    NOTE all metrics implemented in this func are based on sampling. 

    Following almost same logic as function prediction_with_means, but now with Poisson Likelihood.

    Return:
        NOTE for possion likelihood, mode is different from mean, thus the returns are:
        pred_mean: tensor of shape (#output, #inputs)
        pred_mode: tensor of shape (#output, #inputs)
        pred_std: tensor of shape (#output, #inputs)
        pred_f_samples: tensor of shape (#num_f_samples, #output, #inputs)
    """
    my_model.eval()

    num_inputs = data_inputs.shape[0]
    Q, num_outputs, latent_dim = my_model.H.q_mu.shape
    latents = my_model.H.q_mu.data # ONLY ref_means are used !
    num_f_samples = num_f_samples # How many functions y(x) we sample 

    # Prepare empty tensors for storing results
    pred_mean = torch.zeros(num_outputs, num_inputs)
    pred_mode = torch.zeros(num_outputs, num_inputs)
    pred_std = torch.zeros(num_outputs, num_inputs)
    pred_f_samples = torch.zeros(num_f_samples, num_outputs, num_inputs)

    assert 'log_mean_express_level' in kwargs

    for i in tqdm(range(num_outputs)): 
        curr_latent = latents[:, i:(i+1), :].expand(-1, num_inputs, -1)
        pred_f = my_model(latents=curr_latent, inputs=data_inputs)

        pred_f.loc += kwargs['log_mean_express_level'][i]

        # NOTE PossionLikelihood here! marginal is computed by sampling, i.e. we get multiple poisson distributions (n=10) for each input 
        pred_y = my_model.likelihood(pred_f) # (10, #inputs)
        
        # PAY ATTENTION HOW TO COMPUTE THESE STATISTICS
        pred_mean[i, :] = pred_y.mean.mean(dim=0).detach()

        # not sure this mode statistics is correct or not, just an empirical computation ...
        pred_mode[i, :] = pred_y.mode.mode(dim=0)[0].detach()

        # formular of mixture model
        pred_std[i, :] = (pred_y.variance.mean(dim=0).detach() + (pred_y.mean.detach() - pred_mean[i, :]).square().mean(dim=0)).sqrt()
        # pred_std[i, :] = pred_y.variance.sqrt().mean(dim=0).detach()

        for k in range(num_f_samples):   
            # Generate 10 function samples for i-output
            pred_f_samples[k, i, :] = pred_f.sample().detach()
        
        # don't forget we are using exp() as inverse link function
        pred_f_samples = pred_f_samples.exp()

    return pred_mean, pred_mode, pred_std, pred_f_samples

@torch.no_grad()
def prediction_with_sampling(my_model, data_inputs, task_index_list=None, n_gaussian_mixture=5, **kwargs):
    """
    LVMOGP model predictions.

    Make predictions with the approach of monte carlo sampling, i.e. sample latents n_gaussian_mixture (default=5) times
    from q(H), each sample of latents is used to generate a q(y). This form a mixture of gaussian predictive distribution.

    formula:
        x ~ \Sum_i w_i N(x | \mu_i, \sigma_i), x scalar
        overall mean = \Sum_i w_i \mu_i
        overall var = \Sum_i w_i (\mu_i^2 + \sigma_i^2) - (\Sum_i w_i \mu_i)^2
    
    :param my_model: well trained model
    :data_inputs: (tensor) of shape (#inputs, #features)
    :task_index_list: (list) of task ids, if not None, must has same length as data_inputs

    Return:
        pred_mean_final: tensor of shape (#output, #inputs), averaged over n_gaussian_mixture
        pred_std_final: tensor of shape (#output, #inputs), averaged over n_gaussian_mixture
    """
    my_model.eval()
    # Sanity checks
    if task_index_list != None:
        assert len(task_index_list) == data_inputs.shape[0]
    
    assert my_model.H.q_mu.dim() == 3
    
    num_inputs = data_inputs.shape[0]
    Q, num_outputs, latent_dim = my_model.H.q_mu.shape
    latents = torch.zeros(n_gaussian_mixture, Q, num_outputs, latent_dim)
    for mixture_id in range(n_gaussian_mixture):
        latents[mixture_id, ...] = my_model.H().detach()
    
    if task_index_list == None:
        # Prepare empty tensors for storing results
        pred_mean = torch.zeros(n_gaussian_mixture, num_outputs, num_inputs)
        pred_std = torch.zeros(n_gaussian_mixture, num_outputs, num_inputs)

        for mixture_id in range(n_gaussian_mixture):
            for output_id in range(num_outputs): 
                curr_latent = latents[mixture_id, :, output_id:(output_id+1), :].expand(-1, num_inputs, -1)
                pred_f = my_model(latents=curr_latent, inputs=data_inputs)
                task_indices = torch.tensor([output_id for _ in range(num_inputs)])
                pred_y = my_model.likelihood(pred_f, task_indices=task_indices)
                pred_mean[mixture_id, output_id, :] = pred_y.mean.detach()
                pred_std[mixture_id, output_id, :] = pred_y.stddev.detach()

        # Using the properties of mixture of gaussian to compute overall mean and std ...
        pred_mean_final = pred_mean.mean(axis=0)
        pred_var_final = (pred_std.pow(2) + pred_mean.pow(2)).mean(axis=0) - pred_mean_final.pow(2)
        pred_std_final = pred_var_final.sqrt()
        
    else:
        raise NotImplementedError
    
    return pred_mean_final, pred_std_final

@torch.no_grad()
def extract_unseen_locations(config:dict, **kwargs):
    """
    Extract all unseen locations in a 2d tensor. Mask the seen parts, so remaining part is for the unseen locations.

    Return:
        unseen_data: of shape (#all_outputs, #times) not normalized
        unseen_locs: os shape (#all_outputs, 2), the data is normalized
    """

    all_data = torch.load(config['all_data_path'])
    # (#times, #lat, #lon)
    lat = torch.load(config['lat_path'])
    lon = torch.load(config['lon_path'])

    n_lon, n_lat, n_inputs = lon.shape[0], lat.shape[0], all_data.shape[0]
    subsampling_rate = config['rate']
    all_unseen_data = all_data.clone()
    all_unseen_lat = lat.clone()
    all_unseen_lon = lon.clone()

    # Mask the locations used in training ... 
    all_unseen_data[:, ::subsampling_rate, ::subsampling_rate] = torch.nan
    all_unseen_lat[::subsampling_rate] = torch.nan
    all_unseen_lon[::subsampling_rate] = torch.nan

    # first lon with all lats, second lon with all lats, ..., last lon with all lats.
    all_unseen_data_reshape = all_unseen_data.permute(2,1,0).reshape(n_lon*n_lat, n_inputs)

    all_unseen_lon_lat = torch.zeros(n_lon, n_lat, 2)
    for id_lon in range(n_lon):
        for id_lat in range(n_lat): 
            if torch.isnan(all_unseen_lon[id_lon]) and torch.isnan(all_unseen_lat[id_lat]):
                all_unseen_lon_lat[id_lon, id_lat, :] = torch.tensor([torch.nan, torch.nan])
            else:    
                all_unseen_lon_lat[id_lon, id_lat, :] = torch.tensor([lon[id_lon].item(), lat[id_lat].item()])
    
    all_unseen_lon_lat_reshape = all_unseen_lon_lat.reshape(-1, 2)
    final_unseen_lon_lat = (all_unseen_lon_lat_reshape - kwargs['lon_lat_means'].unsqueeze(0)) / kwargs['lon_lat_stds'].unsqueeze(0)

    unseen_dict = {
        'unseen_data': all_unseen_data_reshape, # notice: not normalized
        'unseen_locs': final_unseen_lon_lat     # notice: normalized
        }

    return unseen_dict

@torch.no_grad()
def prediction_on_single_given_loc(my_model, data_inputs: Tensor, loc_info: Tensor):
    """
    LVMOGP prediction q(f), NOTE this is not q(y), as we have no access to noise scale at unknown locs.
    # NOTE loc_info MUST be vector with 2 elements.
    Make predictions for given (possible unseen) loc.

    :param loc_info: of shape (2, )
    """
    my_model.eval()
    loc_info = loc_info.reshape(-1)
    num_inputs = data_inputs.shape[0]

    curr_latents = loc_info.reshape(1, 1, 2).expand(my_model.Q, num_inputs, -1)
    pred_f = my_model(latents=curr_latents, inputs=data_inputs)

    return pred_f

def prediction_on_all_locs(my_model, data_inputs: Tensor, all_locs: Tensor):
    """
    LVMOGP model prediction.
    Iteratively making predictions for all locations.

    # NOTE loc_info MUST be vector with 2 elements.

    :param all_locs: of shape (#locs, 2)
    """
    my_model.eval()
    assert all_locs.shape[-1] == 2

    num_locs = all_locs.shape[0]
    num_inputs = data_inputs.shape[0]

    pred_f_means = torch.zeros(num_locs, num_inputs)
    pred_f_stds = torch.zeros(num_locs, num_inputs)

    for id_loc, loc in enumerate(tqdm(all_locs)):
        if torch.isnan(loc).any().item():
            continue
        loc = loc.reshape(-1)
        curr_latents = loc.reshape(1, 1, 2).expand(my_model.Q, num_inputs, -1)
        pred_f = my_model(latents=curr_latents, inputs=data_inputs)
        pred_f_means[id_loc, :] = pred_f.mean.detach()
        pred_f_stds[id_loc, :] = pred_f.stddev.detach()
        
    return pred_f_means, pred_f_stds

@torch.no_grad()
def multi_indepSVGP_prediction(my_model, data_dict: dict, finer_data_inputs:Tensor=None, **kwargs):
    """
    Multi-IndepSVGP model prediction (by default GaussianLikelihood). If 'likelihood_type' == 'PoissonLikelihood'
    specified in kwargs, we switch to PoissonLikelihood.

    Make predictions using multiple independent SVGP models.
    Notice that some of the outputs need NO prediction, as no test points available for that.
    
    :param my_model: multiple SVGP models.
    :data_dict: a dict generated from 'helper_specify_dataset' func.
    :finer_data_inputs: data_inputs for visualization purpose. (Optional, if None, no need for finer data_inputs)
    """
    num_outputs = data_dict['data_target'].shape[0]
    num_inputs = data_dict['data_inputs'].shape[-1]
    pred_means, pred_stds = torch.zeros(num_outputs, num_inputs), torch.zeros(num_outputs, num_inputs)

    finer_pred_means, finer_pred_stds, finer_pred_f_samples = None, None, None 
    if finer_data_inputs != None:
        num_finer_inputs = finer_data_inputs.shape[-1]
        finer_pred_means, finer_pred_stds = torch.zeros(num_outputs, num_finer_inputs), torch.zeros(num_outputs, num_finer_inputs)
        finer_pred_f_samples = torch.zeros(10, num_outputs, num_finer_inputs)

    for id_output in range(num_outputs):
        if helper_whether_has_test_input(data_dict, id_output) == False:
        # if len(data_dict['ls_of_ls_test_input'][id_output]) < 1:
            continue
        curr_model = my_model.get_model(id_output)
        pred_f = curr_model(data_dict['data_inputs'])
        pred_y = curr_model.likelihood(pred_f)

        if 'likelihood_type' in kwargs and kwargs['likelihood_type'] == 'PoissonLikelihood':
             # PAY ATTENTION HOW TO COMPUTE THESE STATISTICS
            pred_means[id_output, :] = pred_y.mean.mean(dim=0).detach()

            # not sure this mode statistics is correct or not, just an empirical computation ...
            # pred_mode[i, :] = pred_y.mode.mode(dim=0)[0].detach()

            # formular of mixture model
            pred_stds[id_output, :] = (pred_y.variance.mean(dim=0).detach() + (pred_y.mean.detach() - pred_means[id_output, :]).square().mean(dim=0)).std()
        
        else:
            pred_means[id_output, :], pred_stds[id_output, :] = pred_y.mean.detach(), pred_y.stddev.detach()

        if finer_data_inputs != None:
            pred_f_finer = curr_model(finer_data_inputs)
            pred_y_finer = curr_model.likelihood(pred_f_finer)

            if 'likelihood_type' in kwargs and kwargs['likelihood_type'] == 'PoissonLikelihood':
                # PAY ATTENTION HOW TO COMPUTE THESE STATISTICS
                finer_pred_means[id_output, :] = pred_y_finer.mean.mean(dim=0).detach()
                # formular of mixture model
                finer_pred_stds[id_output, :] = (pred_y_finer.variance.mean(dim=0).detach() + (pred_y_finer.mean.detach() - finer_pred_means[id_output, :]).square().mean(dim=0)).std()
        
            else:
                finer_pred_means[id_output, :], finer_pred_stds[id_output, :] = pred_y_finer.mean.detach(), pred_y_finer.stddev.detach()

            for k in range(10):   
                # Generate 10 function samples for i-output
                finer_pred_f_samples[k, id_output, :] = pred_f_finer.sample().detach()
    
    results_dict = {
        'pred_mean_metric': pred_means,
        'pred_std_metric': pred_stds,
        'pred_mean_visual': finer_pred_means,
        'pred_std_visual': finer_pred_stds,
        'pred_f_samples_visual': finer_pred_f_samples,
    }

    return results_dict
    
####################   Helper functions   ####################

def helper_write_dict_results_to_file(dict: dict, file_path: str, extra_description: str, **kwargs):
    """
    Write the results in dict to a given file.
    """
    with open(file_path, 'a') as file:
        file.write(f'{extra_description}\n')
        for key, value in dict.items():
            file.write(f'{key}: {value}\n')
        file.write('\n')

def helper_specify_kernel_by_name(kernel_type:str, input_dim:int):
    """
    """
    if kernel_type == 'RBF':
        return RBFKernel(ard_num_dims=input_dim)
    
    elif kernel_type == 'Linear':
        return LinearKernel()
    
    elif kernel_type == "Matern12":
        return MaternKernel(nu=0.5)

    elif kernel_type == 'Matern32':
        return MaternKernel(nu=1.5)

    elif kernel_type == 'Matern52':
        return MaternKernel(nu=2.5)

    elif kernel_type == 'Scale_RBF':
        return ScaleKernel(RBFKernel(ard_num_dims=input_dim))

    elif kernel_type == 'Scale_RBF_share_lengthscale':
        return ScaleKernel(RBFKernel(ard_num_dims=1))
    
    elif kernel_type == 'Scale_Matern32':
        return ScaleKernel(MaternKernel(nu=1.5))
    
    elif kernel_type == 'Scale_Matern52':
        return ScaleKernel(MaternKernel(nu=2.5))

    elif kernel_type == 'Scale_Matern52_plus_Scale_PeriodicInputMatern52':
        return (ScaleKernel(MaternKernel(nu=2.5)) + ScaleKernel(PeriodicInputsMaternKernel(nu=2.5)))
    
    elif kernel_type == 'Matern52_plus_PeriodicInputMatern52':
        return (MaternKernel(nu=2.5) + PeriodicInputsMaternKernel(nu=2.5))
    
    elif kernel_type == 'Scaled(Matern52_plus_PeriodicInputMatern52)':
        return ScaleKernel(MaternKernel(nu=2.5) + PeriodicInputsMaternKernel(nu=2.5))

    elif kernel_type == 'PeriodicKernel':
        return PeriodicKernel()

    elif kernel_type == 'Scale_PeriodicKernel':
        return ScaleKernel(PeriodicKernel())
    
    else:
        raise NotImplementedError('Your kernel name has not been implemented')

def helper_generate_list_kernels(list_of_str: list, list_of_input_dim: list):
    """
    generate kernels according to str indication.
    :param list_of_input_dim: how many lengthscales are used
    """
    assert len(list_of_str) == len(list_of_input_dim)

    list_of_kernels = []

    for str, input_dim in zip(list_of_str, list_of_input_dim):
        
        list_of_kernels.append(helper_specify_kernel_by_name(str, input_dim))
    
    return list_of_kernels

@torch.no_grad()
def helper_data_transform(_mean: Tensor, _std: Tensor, ref_means: Tensor, ref_stds: Tensor, mode:str='norm2original'):
    """
    Tranform predictions/targets on normalized data scale to original data scale (i.e, times ref_stds plus ref_means)
    or the opposite: transform from original scale to normalized scale ... 

    :param _mean: of shape (#output, #inputs)
    :param _std: of shape (#output, #inputs)
    :param ref_means: of shape (#output,)
    :param ref_stds: of shape (#output,)
    :param mode: two choices: 'norm2original' or 'original2norm'
    """

    if mode == 'norm2original':
        original_mean = _mean.mul(ref_stds.unsqueeze(1)) + ref_means.unsqueeze(1)

        # only transform pred_mean 
        if _std == None: return original_mean

        original_std = _std.mul(ref_stds.unsqueeze(1) + 1e-8)

        return original_mean, original_std

    else:
        raise NotImplementedError

def helper_data_transform_for_single_output(_mean: Tensor, _std: Tensor, ref_means: Tensor, ref_stds: Tensor, mode:str='norm2original'):
    """
    transform (target) data from normalized scale to original scale where there is only one output.

    :param ref_means, ref_stds tensors containing ONLY 1 value.
    """
    if mode == 'norm2original':
        original_mean = _mean.mul(ref_stds) + ref_means

        # only transform pred_mean 
        if _std == None: return original_mean

        original_std = _std.mul(ref_stds)

        return original_mean, original_std

    else:
        raise NotImplementedError

def helper_synthetic_demo_plot(covar_H: Tensor, results_folder_path: str, descrip=None):
    """
    Visualize K(H, H) and display the values in each cell of the matrix.

    :param covar_H: The covariance matrix to visualize.
    :param results_folder_path: Path to save the resulting plot.
    :param descrip: Description of the data, e.g., 'Ground Truth' or 'Estimated'.
    """
    num_H = covar_H.shape[0]
    covar_H_array = covar_H.numpy()  # Convert tensor to numpy array for plotting

    # Create the figure and axes for the plot
    fig, ax = plt.subplots(figsize=(8, 8))

    # Display the tensor as an image with a colormap
    cax = ax.matshow(covar_H_array, cmap='viridis')

    # Add a colorbar to interpret the scale
    fig.colorbar(cax, label='Value')

    # Annotate each cell with the numeric value
    for (i, j), val in np.ndenumerate(covar_H_array):
        ax.text(j, i, f'{val:.2f}', ha='center', va='center', color='white')

    # Set the ticks to be in the middle of the cells and adjust them to the correct range
    ax.set_xticks(np.arange(-.5, int(num_H-1), 1), minor=True)
    ax.set_yticks(np.arange(-.5, int(num_H-1), 1), minor=True)

    # Display grid lines based on minor ticks
    ax.grid(which='minor', color='w', linestyle='-', linewidth=2)
    ax.tick_params(which='minor', size=0)

    # Hide the major tick grid lines
    ax.grid(which='major', color='w', linestyle='', linewidth=0)

    # Set the axis labels and title
    ax.set_xlabel('X-axis')
    ax.set_ylabel('Y-axis')
    if descrip is None:
        descrip = 'Ground Truth'
    ax.set_title(f"Visualization of {descrip} Covariance Matrix from Latent Variables")

    # Show the plot
    plt.tight_layout()

    # Save the figure
    plt.savefig(f'{results_folder_path}/{descrip}_covar_H.pdf')

def helper_plot(data_input: Tensor, 
                pred_mean: Tensor, # for visual
                pred_std: Tensor,  # for visual
                data_target: Tensor, 
                ls_train_input: list,
                ls_test_input: list,
                input_inducing_points: Tensor=None,
                pred_f_samples: Tensor=None,
                data_input4visual: Tensor=None,
                y_lower=None,
                y_upper=None,
                title=None,
                title_fontsize=20,
                **kwargs):
    
    """ 
    Make plots for the output we are interested.

    :param data_input: the input we have train/test data (missing data is possible)
    :param data_input4visual: the inputs we are used 4 visualization (sometime more dense than data_input 4 better visualization results)
    :param output_id: the index of the output we are interested in visualization
    :param pred_mean: pred mean of the output we are interested in 4 visual
    :param pred_std: .. std .. 4 visual
    :param data_target: target of the output we are interested in 
    :param ls_train_input: list of ids for input used for training
    :param ls_test_input: list of ids for input used for testing
    :param pred_f_samples: of shape (#samples, #data)
    """
    # Work on numpy data and make sure data is squeezed
    for var in [data_input, pred_mean, pred_std, data_target, input_inducing_points, data_input4visual, pred_f_samples]:
        if var == None:
            continue
        elif var.dim() > 1:
            var = var.squeeze().numpy()

    if data_input4visual == None:
        data_input4visual = data_input
    else:
        pass
    assert data_input4visual.shape == pred_mean.shape == pred_std.shape 

    
    data_train_input, data_test_input = data_input[ls_train_input], data_input[ls_test_input]
    data_train_target, data_test_target = data_target[ls_train_input], data_target[ls_test_input]
    
    plt.figure(figsize=(8, 6))
    plt.scatter(data_train_input, data_train_target, c='r', marker='x', label='Train Data', s=110)
    plt.scatter(data_test_input, data_test_target, c='k', marker='o', label='Test Data', alpha=0.5, s=90)
    
    if input_inducing_points != None:
        n_induc = input_inducing_points.shape[0]
        plt.scatter(input_inducing_points, [plt.gca().get_ylim()[0] - 1]*n_induc, color='black', marker='^', label='Inducing Locations')

    plt.plot(data_input4visual, pred_mean, 'b', lw=2.5, zorder=9)

    if pred_f_samples != None:
        # iterate over y(x), first id in pred_y_samples refers to #y(x)
        for f_sample in pred_f_samples:
            plt.plot(data_input4visual, f_sample, lw=0.6, alpha=0.5)

    def fill_between_layers(x, mean, std, color, n_layers=2):
        for i in range(1, n_layers + 1):
            alpha = (0.13 / n_layers) * (n_layers - i + 1)  # decrease alpha
            plt.fill_between(x, mean - i * std, mean + i * std, alpha=alpha, color=color)
    
    n_layers = 2
    fill_between_layers(data_input4visual, pred_mean, 2*pred_std / n_layers, 'blue', n_layers=n_layers)

    plt.ylim(y_lower, y_upper)
    plt.legend()

    title_ = title if title != None else "Train/Test Data and Fitted GP"
    plt.title(title_, fontsize=title_fontsize, fontweight='bold')
    plt.tight_layout()
    plt.show()

def visualize_single_frame(frame, points:int=[]):
    """
    Visualize every frame (by default: 100*100 matrix) and mark given points.
    
    Args:
    frame: 2D array representing the image.
    points: List of tuples, where each tuple represents the (x, y) coordinates of a point to be marked.
    """
    plt.figure(figsize=(5, 5))
    plt.imshow(frame)
    # plt.imshow(frame, cmap='gray', vmin=0, vmax=255)
    for point in points:
        # NOTE in matplotlib plot, first index refers to columns in picture ... 
        plt.scatter(point[1], point[0], c='black', s=10)  # s控制点的大小，c控制点的颜色
    plt.axis('off')
    plt.show()

def helper_special_checks_for_spatio_temp_data(config):
    """
    """
    assert config['dataset_type'] == 'spatio_temporal'
    if 'gplvm_init' in config:
        # We will use lon, lat info as initialization ...
        assert config['gplvm_init'] == False

def helper_special_checks_for_gusto_1_indiv_data(config):
    """
    """
    assert config['dataset_type'] == 'gusto_1_indiv'
    

def helper_init_kernel(kernel, kernel_init_dict: dict, kernel_type: str):
    """
    initialize a kernel by the given init parameters (stored in a dict). kernel_type is a str description of what type is the kernel.
    """
    if kernel_type == 'Scale_RBF' or kernel_type == 'Scale_RBF_share_lengthscale':
        with torch.no_grad():
            kernel.outputscale = kernel_init_dict['outputscale_init']
            kernel.base_kernel.lengthscale = kernel_init_dict['lengthscale_init']

    elif kernel_type == 'Linear':
        with torch.no_grad():
            kernel.variance = kernel_init_dict['variance_init']

    elif kernel_type == 'Matern12':
        with torch.no_grad():
            kernel.lengthscale = kernel_init_dict['lengthscale_init']
    
    elif kernel_type == 'Matern32':
        with torch.no_grad():
            kernel.lengthscale = kernel_init_dict['lengthscale_init']

    elif kernel_type == 'Matern52':
        with torch.no_grad():
            kernel.lengthscale = kernel_init_dict['lengthscale_init']
    
    elif kernel_type == 'Scale_Matern52_plus_Scale_PeriodicInputMatern52':
        with torch.no_grad():
            kernel.kernels[0].outputscale = kernel_init_dict['1stKernel_outputscale_init']
            kernel.kernels[0].base_kernel.lengthscale = kernel_init_dict['1stKernel_lengthscale_init']
            kernel.kernels[1].outputscale = kernel_init_dict['2ndKernel_outputscale_init']
            kernel.kernels[1].base_kernel.lengthscale = kernel_init_dict['2ndKernel_lengthscale_init']
            kernel.kernels[1].base_kernel.period_length = kernel_init_dict['2ndKernel_period_lengthscale_init']

    elif kernel_type == 'Matern52_plus_PeriodicInputMatern52':
        with torch.no_grad():
            kernel.kernels[0].lengthscale = kernel_init_dict['1stKernel_lengthscale_init']
            kernel.kernels[1].lengthscale = kernel_init_dict['2ndKernel_lengthscale_init']
            kernel.kernels[1].period_length = kernel_init_dict['2ndKernel_period_lengthscale_init']

    elif kernel_type == 'Scaled(Matern52_plus_PeriodicInputMatern52)':
        with torch.no_grad():
            kernel.outputscale = kernel_init_dict['overall_lengthscale_init']
            kernel.base_kernel.kernels[0].lengthscale = kernel_init_dict['1stKernel_lengthscale_init']
            kernel.base_kernel.kernels[1].lengthscale = kernel_init_dict['2ndKernel_lengthscale_init']
            kernel.base_kernel.kernels[1].period_length = kernel_init_dict['2ndKernel_period_lengthscale_init']

    elif kernel_type == 'RBF':
        with torch.no_grad():
            kernel.lengthscale = kernel_init_dict['lengthscale_init']

    elif kernel_type == 'Scale_Matern32':
        with torch.no_grad():
            kernel.outputscale = kernel_init_dict['outputscale_init']
    
    elif kernel_type == 'Scale_Matern52':
        with torch.no_grad():
            kernel.outputscale = kernel_init_dict['outputscale_init']

    elif kernel_type == 'PeriodicKernel':
        with torch.no_grad():
            kernel.period_length = kernel_init_dict['period_length_init']

    elif kernel_type == 'Scale_PeriodicKernel':
        with torch.no_grad():
            kernel.outputscale = kernel_init_dict['outputscale_init']
            kernel.period_length = kernel_init_dict['period_length_init']

    else:
        raise NotImplementedError(f'{kernel_type} is currently NOT implemented yet.')

def helper_whether_has_test_input(data_dict: dict, id_output: int):
    """
    check if cerrtain output has test data.
    The reason to have a seperate func to implement this functionality is to support old (ls_of_ls_test_input) and new versions (ls_of_ls_within_test_input, ls_of_ls_extrapolate_test_input). 

    Return:
        has_test_data: bool
    """

    if 'ls_of_ls_test_input' in data_dict:
        if len(data_dict['ls_of_ls_test_input'][id_output]) > 0:
            has_test_data = True

        else: 
            has_test_data = False

    else:
        if len(data_dict['ls_of_ls_within_test_input'][id_output]) > 0 or len(data_dict['ls_of_ls_extrapolate_test_input'][id_output]) > 0:
            has_test_data = True

        else: 
            has_test_data = False

    return has_test_data

def greedy_select_distant_points(data: Tensor, num_points:int=5):
    """
    Select a subgroup of points from given data so that they spread away from each other as much as possible.
    Args:
        :param data: of shape (num_points, n_dims)
        :param num_points: the number of points we want

    Return: 
        indices: 
        selected_points: 
    """

    # initialization
    indices = [torch.randint(0, data.size(0), (1,)).item()]  # randomly select the first point
    selected_points = [data[indices[0]]]

    for _ in range(1, num_points):
        # Compute min distance to selected points for every data point
        min_distances = None
        for sp in selected_points:
            distances = torch.norm(data - sp, dim=1, p=2)  # euclidean distance
            if min_distances is None:
                min_distances = distances
            else:
                min_distances = torch.minimum(min_distances, distances)

        # select farest point
        next_index = torch.argmax(min_distances).item()
        indices.append(next_index)
        selected_points.append(data[next_index])

    return indices, torch.stack(selected_points)

def helper_dimension_reduction(data_Y: Tensor, method: str='pca', final_dim: int=2, **kwarg):
    """
    Perform dim reduction for data_Y with different methods.

    Args:
        :param data_Y: of shape (#outputs, #inputs_per_output)
        :param method: str description of what method we want to use. choose from (gplvm, svd, t-SNE)
        :param final_dim: int, how many dims we want after reduction

    Return:
        reduced_data_Y: of shape (#output, final_dim)
    """
    
    if method == 'svd':
        # normalize data
        mean = torch.mean(data_Y, 0)
        std = torch.std(data_Y, 0)
        data_normalized = (data_Y - mean) / std

        # SVD
        U, S, V = torch.svd(data_normalized)

        # pick principle components
        principal_components = V[:, :final_dim]

        # projection
        projected_data = torch.mm(data_normalized, principal_components)

        return projected_data

    elif method == 'gplvm':
        from misc.gplvm_init import specify_gplvm, train_gplvm
        from gpytorch.likelihoods.gaussian_likelihood import GaussianLikelihoodWithMissingObs
        # run gplvm to initialize latent variables in our model
        gplvm_model = specify_gplvm(kwarg['config'])
        gplvm_likelihood = GaussianLikelihoodWithMissingObs()
        gplvm_model, gplvm_likelihood, losses = train_gplvm(gplvm_model, 
                                                            gplvm_likelihood,
                                                            data_Y=data_Y,
                                                            hyper_parameters={'training_steps': 10000,
                                                                                'batch_size': min(100, data_Y.shape[0]),
                                                                                'lr': 0.01})
        return gplvm_model.X().detach()

    elif method == 't-SNE':
        raise NotImplementedError

####################   Metric   ####################

@torch.no_grad()
def metric_prepare(pred_mean: Tensor, pred_std: Tensor, target: Tensor, indicator:list):
    # TODO: support outputs with varied number of data points ...
    """
    Prepare middle results used in following metric computation.
    NOTE if some outputs have no points for test (this info given by indicator), we will skip them ...

    :param pred_mean: predictions mean, of shape (#output_for_test, #all_inputs)
    :param pred_std: predictions stds, same shape as pred_mean
    :param target: targets, same shape as pred_mean
    :indicator: list of list, #outer lists = #output_for_test, len(indicator[i]) = #i_th_output_inputs_for_test. Normally picked from ls_of_ls_test_input/ls_of_ls_train_input.
    """
    assert pred_mean.shape == pred_std.shape == target.shape
    assert len(indicator) == pred_mean.shape[0]
    
    all_pred_means, all_pred_stds, all_targets, all_outputs_testpoints_count = [], [], [], []

    # Important Sanity Check: NOTE: this implementation does not support VARIED length in all_pred_means.
    first_time = True
    for id_output in range(len(indicator)):
        if len(indicator[id_output]) != 0:
            if first_time:
                common_length = len(indicator[id_output])
                first_time = False

            assert len(indicator[id_output]) == common_length

    for id_output in range(len(indicator)):
        curr_output_ids = indicator[id_output]
        if len(curr_output_ids) == 0:
            # No data to test in this output ... 
            continue
        all_pred_means.append(pred_mean[id_output][curr_output_ids].tolist())
        all_pred_stds.append(pred_std[id_output][curr_output_ids].tolist())
        all_targets.append(target[id_output][curr_output_ids].tolist())
        # How many test points for this output 
        all_outputs_testpoints_count.append(len(curr_output_ids))

    # batch of dists ... every batch refers to one output ... 
    # This is the main reason why VARIED length of datapoints (for different output) is not supported.
    pred_dist = MultivariateNormal(mean=torch.tensor(all_pred_means),
                                   covariance_matrix=torch.diag_embed(torch.tensor(all_pred_stds).pow(2)))
    test_y = torch.tensor(all_targets)

    return pred_dist, test_y, all_outputs_testpoints_count

@torch.no_grad()
def metric_mae(pred_mean: Tensor, pred_std: Tensor, target: Tensor, indicator:list):
    from gpytorch.metrics.metrics import mean_absolute_error
    """
    Compute Mean Absolute Error

    :param pred_mean: predictions mean, of shape (#output_for_test, #all_inputs)
    :param pred_std: predictions stds, same shape as pred_mean
    :param target: targets, same shape as pred_mean
    :indicator: list of list, #outer lists = #output_for_test, len(indicator[i]) = #i_th_output_inputs_for_test. Normally picked from ls_of_ls_test_input/ls_of_ls_train_input.

    Return: 
        maes: tensor containing multiple elements
        all_outputs_testpoints_count: how many test data points for every output
    """
    pred_dist, test_y, all_outputs_testpoints_count = metric_prepare(pred_mean, pred_std, target, indicator)

    maes = mean_absolute_error(pred_dist=pred_dist, test_y=test_y) # 1d tensor
    
    return maes, all_outputs_testpoints_count

@torch.no_grad()
def metric_mse(pred_mean: Tensor, pred_std: Tensor, target: Tensor, indicator:list):
    from gpytorch.metrics.metrics import mean_squared_error
    """
    Compute Mean Square Error

    :param pred_mean: predictions, of shape (#output_for_test, #all_inputs)
    :param target: targets, same shape as pred
    :indicator: list of list, #outer lists = #output_for_test, len(indicator[i]) = #i_th_output_inputs_for_test. Normally picked from ls_of_ls_test_input/ls_of_ls_train_input.
    """
    pred_dist, test_y, all_outputs_testpoints_count = metric_prepare(pred_mean, pred_std, target, indicator)

    mses = mean_squared_error(pred_dist=pred_dist, test_y=test_y, squared=True)

    return mses, all_outputs_testpoints_count

@torch.no_grad()
def metric_smse(pred_mean: Tensor, pred_std: Tensor, target: Tensor, indicator:list):
    from gpytorch.metrics.metrics import standardized_mean_squared_error
    """
    Compute Standardised Mean Square Error
        Standardizes the mean squared error by the variance of the test data (GpyTorch approach).
        NOTE If only have 1 test data ---> var of test data=0, this metric meaningless ... 

    :param pred_mean: predictions, of shape (#output_for_test, #all_inputs)
    :param target: targets, same shape as pred
    :indicator: list of list, #outer lists = #output_for_test, len(indicator[i]) = #i_th_output_inputs_for_test. Normally picked from ls_of_ls_test_input/ls_of_ls_train_input.
    """
    pred_dist, test_y, all_outputs_testpoints_count = metric_prepare(pred_mean, pred_std, target, indicator)

    # How many outputs
    num_outputs = pred_dist.batch_shape[0]
    smses = []
    for id_output in range(num_outputs):
        smses.append(standardized_mean_squared_error(pred_dist=pred_dist[id_output], test_y=test_y[id_output]).item())
    
    assert len(all_outputs_testpoints_count) == len(smses)

    return torch.tensor(smses), all_outputs_testpoints_count

@torch.no_grad()
def metric_smse_v2(pred_mean: Tensor, target: Tensor, train_indicator: Tensor, test_indicator: Tensor):
    """
    Use a different way to standarise mean square error.  Given by Nguyen and Bonilla, Collaborative MOGP (2014, UAI)

    :param pred_mean, target have shape (#outputs, #all_inputs)
    :param train_indicator: ls_of_ls_train_input 
    :param test_indicator: ls_of_ls_test_input
    """
    assert len(train_indicator) == len(test_indicator)

    smses, all_outputs_testpoints_count = [], []
    for id in range(len(test_indicator)):
        if len(test_indicator[id]) < 1:
            # NO testpoints for this output
            continue

        ypred = pred_mean[id][test_indicator[id]]
        ytrue = target[id][test_indicator[id]]
        trainmean = (target[id][train_indicator[id]]).mean()
        smse = (ypred-ytrue).square().mean() / (trainmean-ytrue).square().mean()

        smses.append(smse)
        all_outputs_testpoints_count.append(len(test_indicator[id]))

    return torch.tensor(smses), all_outputs_testpoints_count

@torch.no_grad()
def metric_smse_v3(pred_mean: Tensor, target: Tensor, train_indicator: Tensor, test_indicator: Tensor):
    """
    Compute Overall SMSE, i.e, collect prediction for all outputs first, and then compute smse (for overall data).
    """
    ypred_list = []
    ytrue_list = []
    trainmean_list = []

    for id in range(len(test_indicator)):
        if len(test_indicator[id]) < 1:
            # NO testpoints for this output
            continue

        ypred = pred_mean[id][test_indicator[id]]
        ytrue = target[id][test_indicator[id]]
        trainmean = (target[id][train_indicator[id]])

        ypred_list.append(ypred)
        ytrue_list.append(ytrue)
        trainmean_list.append(trainmean)

    ypred_tensor = torch.cat(ypred_list)
    ytrue_tensor = torch.cat(ytrue_list)
    trainmean_tensor = torch.cat(trainmean_list).mean()
    overall_smse = (ypred_tensor-ytrue_tensor).square().mean() / (trainmean_tensor-ytrue_tensor).square().mean()

    return overall_smse

@torch.no_grad()
def metric_nlpd(pred_mean: Tensor, pred_std: Tensor, target: Tensor, indicator:list):
    from gpytorch.metrics.metrics import negative_log_predictive_density
    """
    Compute Negative Log Predictive Density

    :param pred_mean: predictions, of shape (#output_for_test, #all_inputs)
    :param target: targets, same shape as pred
    :indicator: list of list, #outer lists = #output_for_test, len(indicator[i]) = #i_th_output_inputs_for_test. Normally picked from ls_of_ls_test_input/ls_of_ls_train_input.
    NOTE: in preditive distribution, cross-sample covariance is zero, as we assume test data independent during inference time. 
    """
    pred_dist, test_y, all_outputs_testpoints_count = metric_prepare(pred_mean, pred_std, target, indicator)

    # How many outputs
    num_outputs = pred_dist.batch_shape[0]
    nlpds = []
    for id_output in range(num_outputs):
        nlpds.append(negative_log_predictive_density(pred_dist=pred_dist[id_output], test_y=test_y[id_output]).item())
    
    assert len(all_outputs_testpoints_count) == len(nlpds)

    return torch.tensor(nlpds), all_outputs_testpoints_count

@torch.no_grad()
def metric_nlpd_v2(pred_mean: Tensor, pred_std: Tensor, target: Tensor, indicator:list):
    """
    Compute Negative Log Predictive Density by approach given by Nguyen and Bonilla, Collaborative MOGP (2014, UAI).

    :param pred_mean, pred_std, target all have same shape (#outputs, #inputs) 
    """
    assert pred_mean.shape[0] == pred_std.shape[0] == target.shape[0] == len(indicator)
    nlpds, all_outputs_testpoints_count = [], []
    for id in range(len(indicator)):
        if len(indicator[id]) < 1:
            # NO testpoints for this output
            continue

        ymu = pred_mean[id][indicator[id]]
        ytrue = target[id][indicator[id]]
        yvar = (pred_std[id][indicator[id]]).square()

        nlpd = 0.5 * ((ytrue - ymu).square() / yvar + torch.log(2 * torch.pi * yvar)).mean()

        nlpds.append(nlpd)
        all_outputs_testpoints_count.append(len(indicator))
    
    return torch.tensor(nlpds), all_outputs_testpoints_count

def helper_eval_for_all_metrics(pred_mean: Tensor, pred_std: Tensor, target: Tensor, indicator:Tensor, **kwargs):
    """
    A helper func which eval the predictions with all possible metrices.
    :param pred_mean, pred_std, target all have same shape (#outputs, #all_inputs)
    :param indicator: indicator for test points
    """
    # MAE
    maes, all_outputs_testpoints_count = metric_mae(pred_mean=pred_mean, pred_std=pred_std, target=target, indicator=indicator)
    tensor_counts = torch.tensor(all_outputs_testpoints_count)
    # Weighted average; weights are propotional to #testpoints_count
    average_mae = (maes * tensor_counts / tensor_counts.sum()).sum()

    # MSE 
    mses, _ = metric_mse(pred_mean=pred_mean, pred_std=pred_std, target=target, indicator=indicator)
    average_mse = (mses * tensor_counts / tensor_counts.sum()).sum()

    # SMSE
    smses, _ = metric_smse(pred_mean=pred_mean, pred_std=pred_std, target=target, indicator=indicator)
    average_smse = (smses * tensor_counts / tensor_counts.sum()).sum()

    # SMSE_v2
    train_indicator = kwargs['data_dict']['ls_of_ls_train_input']
    smses_v2, _ = metric_smse_v2(pred_mean=pred_mean, target=target, train_indicator=train_indicator, test_indicator=indicator)
    average_smse_v2 = (smses_v2 * tensor_counts / tensor_counts.sum()).sum()

    # NLPD
    nlpds, _ = metric_nlpd(pred_mean=pred_mean, pred_std=pred_std, target=target, indicator=indicator)
    average_nlpd = (nlpds * tensor_counts / tensor_counts.sum()).sum()

    # NLPD_v2
    nlpds_v2, _ = metric_nlpd_v2(pred_mean=pred_mean, pred_std=pred_std, target=target, indicator=indicator)
    average_nlpd_v2 = (nlpds_v2 * tensor_counts / tensor_counts.sum()).sum()

    metric_dict = {
        'average_mae': average_mae.item(),
        'average_mse': average_mse.item(),
        'average_smse': average_smse.item(),
        'average_smse_Nguyen': average_smse_v2.item(),
        'average_nlpd': average_nlpd.item(),
        'average_nlpd_Nguyen': average_nlpd_v2.item()
    }

    return metric_dict
    
