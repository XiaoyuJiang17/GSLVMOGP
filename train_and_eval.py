import time
from tqdm import trange

import random
import numpy as np

import torch
from torch.optim.lr_scheduler import StepLR, CosineAnnealingLR, CyclicLR

import gpytorch
from gpytorch.mlls.variational_elbo import VariationalELBO

from code_blocks.models.OurModels import BayesianGPLVM
from code_blocks.models.MultiIndepSVGP import Multi_IndepSVGP
from code_blocks.models.MultiIndepGP import Multi_IndepExactGP
from code_blocks.likelihoods.poisson_likelihood import PoissonLikelihood
from code_blocks.utils.param_tracker import SimpleTracker, ParamTracker, param_extractor
from utils import (sample_ids_of_output_and_input,
                   sample_ids_with_careful_grouping_outputs,
                   getY_from_ids_of_output_input,
                   sample_single_output_and_batch_input,
                   prediction_with_means,
                   prediction_with_means_possion_likelihood,
                   prediction_on_test_outputs_with_reference_data,
                   helper_data_transform,
                   multi_indepSVGP_prediction,
                   helper_eval_for_all_metrics,
                   helper_write_dict_results_to_file,
                   helper_synthetic_demo_plot,
                   helper_whether_has_test_input,
                   ReservoirSampler
                    )

def helper_specify_scheduler(optimizer, config):
        """
        Specify which scheduler to use ... 
        """
        if config['scheduler'] == 'CyclicLR':
            step_size_up = config['scheduler_param']['step_size_up']
            scheduler = CyclicLR(optimizer, base_lr=config['lr'], max_lr=10*config['lr'], step_size_up=step_size_up, mode='triangular', cycle_momentum=False)
        
        elif config['scheduler'] == 'StepLR':
            try:
                step_size = config['scheduler_param']['step_size']
            except:
                print('No step size is given! We use default value = 30')
                step_size = 30

            try: 
                gamma = config['scheduler_param']['gamma'] 
            except:
                print('No gamma is given! We use default value = 0.95')
                gamma = 0.95
            scheduler = StepLR(optimizer, step_size=step_size, gamma=gamma) 
        else:
            raise NotImplementedError
        
        return scheduler

class train_and_eval_lvmogp_model:
    """
    Gaussian Likelihood case (default case).
    """
    def __init__(self, my_model: BayesianGPLVM, data_dict: dict, config: dict, results_folder_path: str, args, **kwargs):
        
        self.my_model = my_model
        self.data_dict = data_dict
        # Unfold some of the most frequently used parameters in data_dict... 
        self.data_target = data_dict['data_target']
        self.data_inputs = data_dict['data_inputs']
        self.ls_of_ls_train_input = data_dict['ls_of_ls_train_input']
        # self.ls_of_ls_test_input = data_dict['ls_of_ls_test_input']

        self.config = config
        self.results_folder_path = results_folder_path

        # (For Poisson Likelihood) Define the sampler used for mini-batching
        # otherwise, we just only General Sampling method
        if isinstance(self.my_model.likelihood, PoissonLikelihood):
            
            print('We are in Poisson Likelihood! Use ReservoirSampler for mini-batching.')

            medians = torch.nanmedian(self.data_target, dim=1).values
            self.minibatch_sampler = ReservoirSampler(
                ls_of_ls_inputs=self.ls_of_ls_train_input,
                batch_size_output=self.config['batch_size_output'],
                batch_size_input=self.config['batch_size_input'],
                scores=medians
            )
            self.log_mean_express_level = (1e-5 + self.data_target.mean(axis=1)).log() # avoid taking log of zero
            assert self.log_mean_express_level.shape[0] == config['n_outputs']

        # In one preliminary experiment, we compare the time between mini-batch training and full-data training.
        # if full_data_training in config['']

        # Create results.txt under results_folder
        self.results_txt = f'{results_folder_path}/results.txt'
        data_random_seed = config['data_random_seed'] if 'data_random_seed' in config else 1
        with open(self.results_txt, 'w') as file:
            file.write(f'Random seed: {args.random_seed}\n')
            file.write(f'Data random seed: {data_random_seed}\n')

    def train(self):

        # Specify Optimizer
        optimizer = torch.optim.Adam([
            {'params': self.my_model.parameters()}
        ], lr=self.config['lr'])

        # Specify Schedulers
        scheduler = helper_specify_scheduler(optimizer=optimizer, config=self.config)
        
        # Trackers --- record parameter changes during training 
        simple_tracker = SimpleTracker()
        param_tracker = ParamTracker(param_extractor=param_extractor)
        iterator = trange(self.config['n_iterations'], leave=True)

        ##### Training #####

        self.my_model.train()
        start_time = time.time()
        min_loss_value = 1e+10
        correction_term = self.config['total_num_training_points'] / self.config['n_outputs'] # roughly means how many data-points per output

        # -----------------------------------------------------------------
        # carefully sampling outputs: put more 'similar' outputs in one mini-batch. We first need to define 'similarity' between outputs.
        # Here we use mediean value (among all Ys) for each output, and rank them.
        # medians = torch.nanmedian(self.data_target, dim=1).values
        # time_per_iter = []
        for i in iterator: 
            iter_start_time = time.time()    
            # Careful sampling: put 'similar' outputs in one mini-batch.
            '''
            batch_index_output, batch_index_input = sample_ids_with_careful_grouping_outputs(
                ls_of_ls_inputs=self.ls_of_ls_train_input, 
                batch_size_output=self.config['batch_size_output'], 
                batch_size_input=self.config['batch_size_input'], 
                scores=medians
            )
            '''

            if isinstance(self.my_model.likelihood, PoissonLikelihood):
                batch_index_output, batch_index_input = self.minibatch_sampler.sample()

            

            # General sampling (first version)
            else:
                batch_index_output, batch_index_input = sample_ids_of_output_and_input(
                                                                self.ls_of_ls_train_input, 
                                                                batch_size_output = self.config['batch_size_output'], 
                                                                batch_size_input = self.config['batch_size_input'])
            

            '''
            # each interation, only have 1 output
            batch_index_output, batch_index_input = sample_single_output_and_batch_input(
                self.ls_of_ls_train_input, 
                batch_size_input = self.config['batch_size_input']
            )
            '''

            # Optional: Freeze part of the parameters until certain condition satisfies
            '''
            if i < 500:
                with torch.no_grad():
                    self.my_model.H.q_log_sigma.requires_grad = False
            else:
                with torch.no_grad():
                    self.my_model.H.q_log_sigma.requires_grad = True
            '''

            ### NOTE for spatio-temp dataset with periodic kernel only!
            if self.config['dataset_type'] == 'spatio_temporal_v2':
                if "training_trick" in self.config and self.config['training_trick'] == True:
                    if i < self.config['n_iterations'] / 2:
                        with torch.no_grad():
                            self.my_model.covar_module_input[0].kernels[1].base_kernel.raw_period_length.requires_grad = False
                    else:
                        with torch.no_grad():
                            self.my_model.covar_module_input[0].kernels[1].base_kernel.raw_period_length.requires_grad = True
            

            optimizer.zero_grad()
            ### Computing Loss = negative variational elbo = - (log_likelihood - kl_divergence - added_loss)
            # NOTE we are using elbo per data, which is averaged over total number of datapoints
            loss, log_likelihood_term, latent_kl_term = 0.0, 0.0, 0.0

            for _ in range(self.config['num_latent_MC']):
                sample_batch_latent = self.my_model.sample_latent_variable(batch_idx=batch_index_output, **self.data_dict)
                sample_batch_input = self.data_inputs[batch_index_input]
                # q(f), determined jointly by latent variables and inputs
                output_batch = self.my_model(latents=sample_batch_latent, inputs=sample_batch_input)  
                batch_targets = getY_from_ids_of_output_input(batch_index_output, batch_index_input, self.data_target)

                ## log-likelihood term
                if isinstance(self.my_model.likelihood, PoissonLikelihood):
                    # here, we may want to corrupt the distribution q(f) a little bit to get more better training (?)
                    shift_factors = self.log_mean_express_level[batch_index_output] # shift_factors have the same shape as output_batch.loc and batch_index_output
                    output_batch.loc += shift_factors
                    
                    log_likelihood_batch = self.my_model.likelihood.expected_log_prob(
                        observations=batch_targets,
                        function_dist=output_batch
                    )

                else:
                    log_likelihood_batch = self.my_model.likelihood.expected_log_prob(
                        target=batch_targets,
                        input=output_batch,
                        task_indices=torch.tensor(batch_index_output)
                    )
                loss += -log_likelihood_batch.mean()
                log_likelihood_term += -log_likelihood_batch.mean().detach().item()

                ## KL terms of latent variables
                added_loss = torch.zeros_like(loss)
                for added_loss_term in self.my_model.added_loss_terms():
                    # ONLY one added loss here, which is KL in latent space (sum over Q, averaged over mini-batches)
                    added_loss.add_((1/correction_term) * self.config['alpha'] * added_loss_term.loss())
                loss += added_loss
                latent_kl_term += added_loss.detach().item()
        
            ## KL term of q(u) p(u)
            kl_divergence = self.my_model.variational_strategy.kl_divergence().div(self.config['total_num_training_points']) * self.config['beta']  
            loss = loss / self.config['num_latent_MC'] +  kl_divergence
            variational_kl_term = kl_divergence.detach().item()

            loss.backward()
            loss_value = loss.item()
        
            # store model every 100 iterations
            if i > 200 and i % 100 == 0 and loss_value < min_loss_value:
                print(f'A new model is stored, with current loss value {loss_value}.')
                torch.save(self.my_model.state_dict(), f'{self.results_folder_path}/min_model.pth')
                min_loss_value = loss_value

            loss_terms_dict = {'loss_value': loss_value, 
                           'log_likelihood_term': log_likelihood_term / self.config['num_latent_MC'], 
                           'latent_kl_term': latent_kl_term / self.config['num_latent_MC'], 
                           'variational_kl_term': variational_kl_term}
            
            simple_tracker.update(loss_terms_dict)
            param_tracker.update(model=self.my_model)

            iterator.set_description('Loss: ' + str(float(np.round(loss_value, 3))) + ", iter no: " + str(i))

            # Clip gradients
            torch.nn.utils.clip_grad_norm_(self.my_model.parameters(), self.config['max_grad_norm'])

            optimizer.step()
            scheduler.step()

            # iter_end_time = time.time()
            # this_iter_time = iter_end_time - iter_start_time
            # time_per_iter.append(this_iter_time)
        # -----------------------------------------------------------------    

        # TODO:
        ##
        # print(np.array(time_per_iter).mean())
        # print(np.array(time_per_iter).std())
        # print(stop)
        ##
        end_time = time.time()
        total_training_time = end_time - start_time
        with open(self.results_txt, 'a') as file:
            file.write(f'Training time: {total_training_time:.2f}\n')

        # Save model
        torch.save(self.my_model.state_dict(), f'{self.results_folder_path}/final_model.pth')
        print('Finish Training!')

        # Make Plots 
        import os
        plots_folder_path = f'{self.results_folder_path}/plots'
        os.makedirs(plots_folder_path, exist_ok=True)
        simple_tracker.plot(plots_folder_path)
        param_tracker.plot(plots_folder_path)

    def eval(self):
        # NOTE We only eval final model, min model is NOT eval!
        self.my_model.eval()
        # In normalized data scale ; q(y)
        pred_mean_metric, pred_std_metric, _ = prediction_with_means(my_model=self.my_model, data_inputs=self.data_dict['data_inputs'])

        # Transform back to original data scale
        original_data_target = helper_data_transform(_mean=self.data_dict['data_target'], _std=None, ref_means=self.data_dict['means'], ref_stds=self.data_dict['stds'], mode='norm2original')
        original_pred_mean_metric, original_pred_std_metric = helper_data_transform(_mean=pred_mean_metric, _std=pred_std_metric, 
                                                                                    ref_means=self.data_dict['means'], 
                                                                                    ref_stds=self.data_dict['stds'], 
                                                                                    mode='norm2original')
        
        if 'ls_of_ls_test_input' in self.data_dict:
        # In most cases, we only have one split for testing (eval)
            dict_metric_test = helper_eval_for_all_metrics(
                pred_mean=original_pred_mean_metric,
                pred_std=original_pred_std_metric,
                target=original_data_target,
                indicator=self.data_dict['ls_of_ls_test_input'],
                data_dict = self.data_dict
            )
            
            helper_write_dict_results_to_file(dict_metric_test, self.results_txt, 'Metric on Test split: ')

        elif 'ls_of_ls_within_test_input' in self.data_dict and 'ls_of_ls_extrapolate_test_input' in self.data_dict:
        # for spatio-temp dataset, the testing samples are grouped into 2 categories: within or extrapolate
            dict_metric_test = helper_eval_for_all_metrics(
                pred_mean=original_pred_mean_metric,
                pred_std=original_pred_std_metric,
                target=original_data_target,
                indicator=self.data_dict['ls_of_ls_within_test_input'],
                data_dict = self.data_dict
            )
            
            helper_write_dict_results_to_file(dict_metric_test, self.results_txt, 'Metric on Within Test split: ')

            # # # # --- --- --- --- # # # # --- --- --- --- # # # # --- --- --- --- # # # # --- --- --- ---

            dict_metric_test = helper_eval_for_all_metrics(
                pred_mean=original_pred_mean_metric,
                pred_std=original_pred_std_metric,
                target=original_data_target,
                indicator=self.data_dict['ls_of_ls_extrapolate_test_input'],
                data_dict = self.data_dict
            )
            
            helper_write_dict_results_to_file(dict_metric_test, self.results_txt, 'Metric on Extrapolate Test split: ')

        else:
            raise TypeError('You need to specify which data points are for testing! ')
            


    def eval_with_ref(self):
        """
        evaluation for predictions with reference data, compute q(y_test | y_ref)
        """

        # Sanity Check
        assert 'ls_of_ls_ref_input' in self.data_dict

        self.my_model.eval()

        # Of shape (#outputs, #inputs), if not testing point, value = nan
        pred_mean_metric, pred_std_metric = prediction_on_test_outputs_with_reference_data(
            my_model = self.my_model, 
            data_inputs = self.data_inputs, 
            ls_of_ls_ref_input = self.data_dict['ls_of_ls_ref_input'], 
            ls_of_ls_test_input = self.data_dict['ls_of_ls_test_input'], 
            data_target = self.data_dict['data_target']
        )

        # Transform back to original data scale
        original_data_target = helper_data_transform(_mean=self.data_dict['data_target'], _std=None, ref_means=self.data_dict['means'], ref_stds=self.data_dict['stds'], mode='norm2original')
        original_pred_mean_metric, original_pred_std_metric = helper_data_transform(_mean=pred_mean_metric, _std=pred_std_metric, 
                                                                                    ref_means=self.data_dict['means'], 
                                                                                    ref_stds=self.data_dict['stds'], 
                                                                                    mode='norm2original')

        dict_metric_test = helper_eval_for_all_metrics(
            pred_mean=original_pred_mean_metric,
            pred_std=original_pred_std_metric,
            target=original_data_target,
            indicator=self.data_dict['ls_of_ls_test_input'],
            data_dict = self.data_dict
        )

        helper_write_dict_results_to_file(dict_metric_test, self.results_txt, 'Metric on Test split: ')

    def train_and_eval(self):
        self.train()

        # in some cases, we don't need eval!
        if self.config['dataset_type'] == 'synthetic_regression':
            print('No eval is performed! ')
        else:
            self.eval()

    def button_grads(self, requires_grad=True):
        """
        Switch on/off gradient flows for kernel parameters and likelihood parameters.
        """
        self.my_model.likelihood.raw_task_noises.requires_grad = requires_grad

        for kernel in self.my_model.covar_module_latent:
            for name, param in kernel.named_parameters():
                param.requires_grad = requires_grad
        
        for kernel in self.my_model.covar_module_input:
            for name, param in kernel.named_parameters():
                param.requires_grad = requires_grad

    def button_grads_var_related_params(self, requires_grad=True):
        """
        Optional: Switch on/off gradients flow for variance related model params
        """
        self.my_model.likelihood.raw_task_noises.requires_grad = requires_grad
        
        for kernel in self.my_model.covar_module_latent:
            kernel.raw_outputscale.requires_grad = requires_grad
    
    def plot_covar_H(self):
        """
        Optional (expected to be used in synthetic demo experiments, Q=1):
            plot covariance matrix (parametrized by latent variables)
        # TODO support more general cases ...
        """
        assert self.my_model.Q == 1

        my_H = self.my_model.H.q_mu.detach()
        covar_H = self.my_model.covar_module_latent[0](my_H)[0, :, :]
        # covar_H should be 2d tensor
        helper_synthetic_demo_plot(covar_H, self.results_folder_path, descrip='Estimated')

class train_and_eval_lvmogp_model_poisson_likelihood(train_and_eval_lvmogp_model):
    """
    Poisson Likelihood case (for rotateMNIST_integer experiment).
    """
    def __init__(self, my_model: BayesianGPLVM, data_dict: dict, config: dict, results_folder_path: str, args, **kwargs):
        super().__init__(my_model, data_dict, config, results_folder_path, args, **kwargs)

    def eval(self):
        self.my_model.eval()
        pred_mean_metric, pred_mode_metric, pred_std_metric, _ = prediction_with_means_possion_likelihood(my_model=self.my_model, data_inputs=self.data_dict['data_inputs'])
        # WE don't need prediction transformation, as we are dealing with original data in the model

        '''
        ## Using Poisson Likelihood Mean
        dict_metric_test = helper_eval_for_all_metrics(
                pred_mean=pred_mean_metric,
                pred_std=pred_std_metric,
                target=self.data_dict['data_target'],
                indicator=self.data_dict['ls_of_ls_test_input'],
                data_dict = self.data_dict
            )
        helper_write_dict_results_to_file(dict_metric_test, self.results_txt, 'Metric on Test split with Poisson Likelihood Mean: ')

        ## Using Poisson Likelihood Mode
        dict_metric_test = helper_eval_for_all_metrics(
                pred_mean=pred_mode_metric,
                pred_std=pred_std_metric,
                target=self.data_dict['data_target'],
                indicator=self.data_dict['ls_of_ls_test_input'],
                data_dict = self.data_dict
            )
        helper_write_dict_results_to_file(dict_metric_test, self.results_txt, 'Metric on Test split with Poisson Likelihood Mode: ')
        '''

        entire_maes = (pred_mean_metric - self.data_dict['data_target']).abs() # (#outputs, #inputs)

        entire_nlpd = 0.5 * ((self.data_dict['data_target'] - pred_mean_metric).square() / pred_std_metric.pow(2) + torch.log(2 * torch.pi * pred_std_metric.pow(2))) # recall metric_nlpd_v2 in utils.py

        masks4train = self.data_dict['masks4train'] # (#outputs, #inputs)
        masks4test = ~masks4train

        average_test_mae = entire_maes[masks4test].mean().item()
        average_test_nlpd = entire_nlpd[masks4test].mean().item()

        print('average_test_mae: ', average_test_mae, '\n')
        print('average_test_nlpd: ', average_test_nlpd, '\n')

        with open(self.results_txt, 'a') as file:
            file.write(f'average_test_mae: {average_test_mae}\n')
            file.write(f'average_test_nlpd: {average_test_nlpd}\n')


class train_and_eval_IndepSVGP_model:
    def __init__(self, my_models: Multi_IndepSVGP, data_dict: dict, config: dict, results_folder_path: str, args, **kwargs):
        
        self.my_models = my_models
        self.data_dict = data_dict
        # Unfold some of the most frequently used parameters in data_dict... 
        self.data_target = data_dict['data_target']
        self.data_inputs = data_dict['data_inputs']
        self.ls_of_ls_train_input = data_dict['ls_of_ls_train_input']

        self.config = config
        self.results_folder_path = results_folder_path

        # Create results.txt under results_folder
        self.results_txt = f'{results_folder_path}/results.txt'
        data_random_seed = config['data_random_seed'] if 'data_random_seed' in config else 1
        with open(self.results_txt, 'w') as file:
            file.write(f'Random seed: {args.random_seed}\n')
            file.write(f'Data random seed: {data_random_seed}\n')
    
    def train(self):
        """
        Train SVGPs for all outputs one by one ... 
        If no test points available for certain output, we skip training that model ...  
        """
        # Record training time for different outputs
        # Record active outputs
        ls_training_time = []
        self.active_outputs_list = []

        for id_output in range(self.config['n_outputs']):
            # Check if there are test data for certain output
            has_test_data = helper_whether_has_test_input(self.data_dict, id_output)

            if has_test_data == False:
                # No test data! No need to train this model.
                print(f'Skip training {id_output}_th model! Because no test points for it !')
            
            else:
                self.train_single_SVGP(id_output, ls_training_time=ls_training_time)
                self.active_outputs_list.append(id_output)

        total_time = np.array(ls_training_time).sum()

        with open(self.results_txt, 'a') as file:
            file.write(f'Total time: {total_time}\n')
            file.write(f'Active Outputs: {self.active_outputs_list}\n')
        
        # Save the model we trained
        torch.save(self.my_models.state_dict(), f'{self.results_folder_path}/MultiSVGP.pth')

    def train_single_SVGP(self, id_output, **kwargs):
        """
        """
        def mini_batching_sampling_func(num_inputs, batch_size):
            idx_list = random.choices(range(num_inputs), k=batch_size)
            return idx_list

        # Identify the Model & Dataset for current output
        curr_model = self.my_models.get_model(id_output)
        train_ids = self.ls_of_ls_train_input[id_output]
        train_inputs = self.data_inputs[train_ids]
        train_targets = self.data_target[id_output, train_ids]

        assert train_inputs.shape == train_targets.shape
        n_train_inputs = train_inputs.shape[0]

        # Specify Optimizer
        optimizer = torch.optim.Adam([
            {'params': curr_model.parameters()}
        ], lr=self.config['lr'])

        # Specify Schedulers
        scheduler = helper_specify_scheduler(optimizer=optimizer, config=self.config)
        
        # Trackers --- record parameter changes during training 
        ls_train_loss = []
        iterator = trange(self.config['n_iterations'], leave=True)

        ##### Training #####
        curr_model.train()
        start_time = time.time()

        # Specify Loss
        mll = VariationalELBO(curr_model.likelihood, curr_model, num_data=n_train_inputs)

        for i in iterator:
            
            # NOTE Optional: ONLY for spatio-temp dataset!
            if self.config['dataset_type'] == 'spatio_temporal_v2':
                if i < 700:
                    with torch.no_grad():
                        curr_model.covar_module.kernels[1].base_kernel.raw_period_length.requires_grad = False
                else:
                    with torch.no_grad():
                        curr_model.covar_module.kernels[1].base_kernel.raw_period_length.requires_grad = True
            
            optimizer.zero_grad()
            mini_batch_idx = mini_batching_sampling_func(num_inputs=n_train_inputs, batch_size=self.config['batch_size_input'])
            output_pred = curr_model(train_inputs[mini_batch_idx])
            loss = -mll(output_pred, train_targets[mini_batch_idx])
            ls_train_loss.append(loss.item())
            iterator.set_description('Training ' + str(id_output) + 'th Model; ' + 'Loss: ' + str(float(np.round(loss.item(),3))) + ', iter no: ' + str(i))
            loss.backward()

            # clip gradients
            torch.nn.utils.clip_grad_norm_(curr_model.parameters(), self.config['max_grad_norm'])

            optimizer.step()
            scheduler.step()

        end_time = time.time()
        total_training_time = end_time - start_time
        kwargs['ls_training_time'].append(total_training_time)

    def eval(self):
        self.my_models.eval()
        # In normalized data scale
        pred_results_dict = multi_indepSVGP_prediction(self.my_models, self.data_dict)

        # Transform back to original data scale
        original_data_target = helper_data_transform(_mean=self.data_dict['data_target'], _std=None, ref_means=self.data_dict['means'], ref_stds=self.data_dict['stds'], mode='norm2original')
        original_pred_mean_metric, original_pred_std_metric = helper_data_transform(_mean=pred_results_dict['pred_mean_metric'], 
                                                                                    _std=pred_results_dict['pred_std_metric'], 
                                                                                    ref_means=self.data_dict['means'], 
                                                                                    ref_stds=self.data_dict['stds'], 
                                                                                    mode='norm2original')

        if 'ls_of_ls_test_input' in self.data_dict:
        # In most cases, we only have one split for testing (eval)
            dict_metric_test = helper_eval_for_all_metrics(
                pred_mean=original_pred_mean_metric,
                pred_std=original_pred_std_metric,
                target=original_data_target,
                indicator=self.data_dict['ls_of_ls_test_input'],
                data_dict = self.data_dict
            )
            
            helper_write_dict_results_to_file(dict_metric_test, self.results_txt, 'Metric on Test split: ')

        elif 'ls_of_ls_within_test_input' in self.data_dict and 'ls_of_ls_extrapolate_test_input' in self.data_dict:
        # for spatio-temp dataset, the testing samples are grouped into 2 categories: within or extrapolate
            dict_metric_test = helper_eval_for_all_metrics(
                pred_mean=original_pred_mean_metric,
                pred_std=original_pred_std_metric,
                target=original_data_target,
                indicator=self.data_dict['ls_of_ls_within_test_input'],
                data_dict = self.data_dict
            )
            
            helper_write_dict_results_to_file(dict_metric_test, self.results_txt, 'Metric on Within Test split: ')

            # # # # --- --- --- --- # # # # --- --- --- --- # # # # --- --- --- --- # # # # --- --- --- ---

            dict_metric_test = helper_eval_for_all_metrics(
                pred_mean=original_pred_mean_metric,
                pred_std=original_pred_std_metric,
                target=original_data_target,
                indicator=self.data_dict['ls_of_ls_extrapolate_test_input'],
                data_dict = self.data_dict
            )
            
            helper_write_dict_results_to_file(dict_metric_test, self.results_txt, 'Metric on Extrapolate Test split: ')
            
        else:
            raise TypeError('You need to specify which data points are for testing! ')
        
    def train_and_eval(self):
        self.train()
        # in some cases, we don't need eval!
        if self.config['dataset_type'] == 'synthetic_regression':
            print('No eval is performed! ')
        else:
            self.eval()
    
    def button_grads(self, requires_grad=True):
        """
        Switch on/off gradient flows for kernel parameters and likelihood parameters.
        Just like func defined in train_and_eval_lvmogp_model, but now we need to loop over all IndepSVGPs defined for every output.
        """

        # iterative over all IndepSVGPs
        for id_output in range(self.config['n_outputs']):
            curr_model = self.my_models.get_model(id_output)

            for name, param in curr_model.covar_module.named_parameters():
                param.requires_grad = requires_grad

            for name, param in curr_model.likelihood.named_parameters():
                param.requires_grad = requires_grad

class train_and_eval_IndepSVGP_model_poisson_likelihood(train_and_eval_IndepSVGP_model):
    """
    Poisson Likelihood case (for rotateMNIST_integer experiment).
    """
    def __init__(self, my_models: Multi_IndepSVGP, data_dict: dict, config: dict, results_folder_path: str, args, **kwargs):
        super().__init__(my_models, data_dict, config, results_folder_path, args, **kwargs)
    
    def eval(self):
        """"""
        self.my_models.eval()
        # In normalized data scale
        pred_results_dict = multi_indepSVGP_prediction(self.my_models, self.data_dict, likelihood_type='PoissonLikelihood')
        # NO transformation is needed!
        dict_metric_test = helper_eval_for_all_metrics(
                pred_mean=pred_results_dict['pred_mean_metric'],
                pred_std=pred_results_dict['pred_std_metric'],
                target=self.data_dict['data_target'],
                indicator=self.data_dict['ls_of_ls_test_input'],
                data_dict = self.data_dict
        )
            
        helper_write_dict_results_to_file(dict_metric_test, self.results_txt, 'Metric on Test split: ')

class train_and_eval_IndepExactGP_model:
    """
    """
    def __init__(self, my_models: Multi_IndepExactGP, data_dict: dict, config: dict, results_folder_path: str, args, **kwargs):
        
        self.my_models = my_models
        self.data_dict = data_dict
        # Unfold some of the most frequently used parameters in data_dict... 
        self.data_target = data_dict['data_target']
        self.data_inputs = data_dict['data_inputs']
        self.ls_of_ls_train_input = data_dict['ls_of_ls_train_input']
        # self.ls_of_ls_test_input = data_dict['ls_of_ls_test_input']

        self.config = config
        self.results_folder_path = results_folder_path

        # Create results.txt under results_folder
        self.results_txt = f'{results_folder_path}/results.txt'
        data_random_seed = config['data_random_seed'] if 'data_random_seed' in config else 1
        with open(self.results_txt, 'w') as file:
            file.write(f'Random seed: {args.random_seed}\n')
            file.write(f'Data random seed: {data_random_seed}\n')

    def train(self):
        """
        We train all ExactGPs together!
        NOTE This might be in-efficient if many outputs do not have test data (We actually not necessary to train the model).
        """
        model = gpytorch.models.IndependentModelList(*self.my_models.model_list)
        from gpytorch.mlls import SumMarginalLogLikelihood

        model.train()
        start_time = time.time()

        # Specify Loss
        mll = SumMarginalLogLikelihood(model.likelihood, model)
        
        # Specify Optimizer
        optimizer = torch.optim.Adam([
            {'params': model.parameters()}
        ], lr=self.config['lr'])
       
        # Specify Schedulers
        scheduler = helper_specify_scheduler(optimizer=optimizer, config=self.config)
        
        iterator = trange((self.config['n_epochs']), leave=True)
        for i in iterator:
            optimizer.zero_grad()

            # NOTE Optional: ONLY for spatio-temp dataset!
            if self.config['dataset_type'] == 'spatio_temporal_v2':

                if 'training_trick' in self.config and self.config['training_trick'] == True:

                    if i < self.config['n_epochs'] / 2:
                        with torch.no_grad():
                            # fix certain parameters in each of the output model
                            for curr_model_id, curr_model in enumerate(self.my_models.model_list):
                                curr_model.covar_module.kernels[1].base_kernel.raw_period_length.requires_grad = False
                    else:
                        with torch.no_grad():
                            # unfix certain parameters in each of the output model
                            for curr_model_id, curr_model in enumerate(self.my_models.model_list):
                                curr_model.covar_module.kernels[1].base_kernel.raw_period_length.requires_grad = True
            
            output = model(*model.train_inputs)
            loss = -mll(output, model.train_targets)
            loss.backward()
            iterator.set_description('Training Loss: ' + str(float(np.round(loss.item(), 4))) + ', iter no: ' + str(i + 1))
            # print('Iter %d/%d - Loss: %.3f' % (i + 1, self.config['n_epochs'], loss.item()))
            
            # clip gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), self.config['max_grad_norm'])

            optimizer.step()
            scheduler.step()

        end_time = time.time()
        total_training_time = end_time - start_time
        print('Finish training! total training time: ', total_training_time)

        # Save the model we trained
        torch.save(self.my_models.state_dict(), f'{self.results_folder_path}/IndepExactGP.pth')

    def eval(self, ls_test_inputs: list=None, **kwargs):
        """
        param ls_test_inputs: list of tensors, each tensor corresponds to one output. 
        If none, we will use the test input specified in data_dict.
        """
        model = gpytorch.models.IndependentModelList(*self.my_models.model_list)
        model.eval()

        if ls_test_inputs != None:
            assert len(ls_test_inputs) == len(self.my_models)
            assert torch.is_tensor(ls_test_inputs[0])

        else:
            # We specify test inputs and targets for each output.
            ls_test_inputs = [self.data_inputs for _ in range(self.config['n_outputs'])]
        
        # Make Predictions
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            predictions = model.likelihood(*model(*ls_test_inputs))
        
        pred_mean_metric, pred_std_metric = torch.zeros_like(self.data_dict['data_target']), torch.zeros_like(self.data_dict['data_target'])
        
        for output_id, pred in enumerate(predictions):
            pred_mean_metric[output_id, :] = pred.mean.detach()
            pred_std_metric[output_id, :] = pred.stddev.detach()

        # Transform back to original data scale
        original_data_target = helper_data_transform(_mean=self.data_dict['data_target'], _std=None, ref_means=self.data_dict['means'], ref_stds=self.data_dict['stds'], mode='norm2original')
        original_pred_mean_metric, original_pred_std_metric = helper_data_transform(_mean=pred_mean_metric, 
                                                                                    _std=pred_std_metric, 
                                                                                    ref_means=self.data_dict['means'], 
                                                                                    ref_stds=self.data_dict['stds'], 
                                                                                    mode='norm2original')
        
        if 'ls_of_ls_test_input' in self.data_dict:
        # In most cases, we only have one split for testing (eval)
            dict_metric_test = helper_eval_for_all_metrics(
                pred_mean=original_pred_mean_metric,
                pred_std=original_pred_std_metric,
                target=original_data_target,
                indicator=self.data_dict['ls_of_ls_test_input'],
                data_dict = self.data_dict
            )
            
            helper_write_dict_results_to_file(dict_metric_test, self.results_txt, 'Metric on Test split: ')

        elif 'ls_of_ls_within_test_input' in self.data_dict and 'ls_of_ls_extrapolate_test_input' in self.data_dict:
        # for spatio-temp dataset, the testing samples are grouped into 2 categories: within or extrapolate
            dict_metric_test = helper_eval_for_all_metrics(
                pred_mean=original_pred_mean_metric,
                pred_std=original_pred_std_metric,
                target=original_data_target,
                indicator=self.data_dict['ls_of_ls_within_test_input'],
                data_dict = self.data_dict
            )
            
            helper_write_dict_results_to_file(dict_metric_test, self.results_txt, 'Metric on Within Test split: ')

            # # # # --- --- --- --- # # # # --- --- --- --- # # # # --- --- --- --- # # # # --- --- --- ---

            dict_metric_test = helper_eval_for_all_metrics(
                pred_mean=original_pred_mean_metric,
                pred_std=original_pred_std_metric,
                target=original_data_target,
                indicator=self.data_dict['ls_of_ls_extrapolate_test_input'],
                data_dict = self.data_dict
            )
            
            helper_write_dict_results_to_file(dict_metric_test, self.results_txt, 'Metric on Extrapolate Test split: ')
            
        else:
            raise TypeError('You need to specify which data points are for testing! ')
        
    def train_and_eval(self):
        raise NotImplementedError
