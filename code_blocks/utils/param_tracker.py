import matplotlib.pyplot as plt

class SimpleTracker:
    '''
    track several variables during training. For instance, track how different components of loss changes at training time.
    '''
    def __init__(self):

        self.values_history_dict = {} # dict of list
        self.initialized = False
    
    def update(self, dict):

        # dict stores variable names and values
        if self.initialized == False:
            for key in dict:
                self.values_history_dict[f'{key}'] = []

            self.initialized = True
        
        for key in dict:
            self.values_history_dict[key].append(dict[key])

    def plot(self, folder_path):

        for key, values in self.values_history_dict.items():
            plt.figure()
            plt.plot(values)
            plt.title(f"Plot of value {key}")
            plt.xlabel("iter")
            plt.ylabel("Value")
            plt.savefig(f"{folder_path}/value_key_{key}.png") 
            plt.close()  


class ParamTracker:
    '''
    A util class used in training phase, record the values of parameters we are interested for tracking.
    '''
    def __init__(self, param_extractor):
        '''
        :param param_extractor: a function takes model config as input, output the dict of parameters we are interested for tracking.
        '''
        self.param_extractor = param_extractor
        self.param_history = [] # list of dictonaries
        self.initialized = False
        
    def update(self, model):
        '''
        used in training process, record parameter values at certain iteration.
        '''
        self.param_history.append(self.param_extractor(model))

    def plot(self, folder_path):
        '''
        '''
        data = self.param_history # list of dicts
        values_dict = {key: [] for key in data[0]} 

        for item in data:
            for key in item:
                values_dict[key].append(item[key])

        for key, values in values_dict.items():
            plt.figure()
            plt.plot(values)
            plt.title(f"Plot of param {key}")
            plt.xlabel("iter")
            plt.ylabel("Value")
            plt.savefig(f"{folder_path}/key_{key}.png") 
            plt.close()  


def param_extractor(model):
    '''
    Used for lvmogp model.

    Args:
        model: 

    '''
    param_dict = {}

    # covar_module_latent
    for i, kernel in enumerate(model.covar_module_latent):
        for name, param in kernel.named_parameters():
            try:
                param_dict[f'latent_kernel_{i}_{name}'] = param.detach().item()
            except RuntimeError:
                param = param.detach().reshape(-1)
                for j, ele in enumerate(param):
                    param_dict[f'latent_kernel_{i}_{name}_{j}th_element'] = ele.detach().item()

    # covar_module_input
    for i, kernel in enumerate(model.covar_module_input):
        for name, param in kernel.named_parameters():
            try:
                param_dict[f'input_kernel_{i}_{name}'] = param.detach().item()
            except RuntimeError:
                param = param.detach().reshape(-1)
                for j, ele in enumerate(param):
                    param_dict[f'input_kernel_{i}_{name}_{j}th_element'] = ele.detach().item()
    
    # likelihood


    return param_dict


