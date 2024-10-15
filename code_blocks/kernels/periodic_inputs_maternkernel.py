from typing import Optional

import math
import torch

from gpytorch.kernels import Kernel
from gpytorch.constraints import Interval, Positive

class PeriodicInputsMaternKernel(Kernel):

    has_lengthscale = True

    def __init__(self, 
                 nu: Optional[float] = 2.5, 
                 period_length_constraint: Optional[Interval] = None,
                 **kwargs):
        if nu not in {0.5, 1.5, 2.5}:
            raise RuntimeError("nu expected to be 0.5, 1.5, or 2.5")
        super(PeriodicInputsMaternKernel, self).__init__(**kwargs)
        self.nu = nu

        if period_length_constraint is None:
            period_length_constraint = Positive()
        
        ard_num_dims = kwargs.get("ard_num_dims", 1)

        self.register_parameter(
            name="raw_period_length", parameter=torch.nn.Parameter(torch.zeros(*self.batch_shape, 1, ard_num_dims))
        )
        self.register_constraint("raw_period_length", period_length_constraint)

    @property
    def period_length(self):
        return self.raw_period_length_constraint.transform(self.raw_period_length)

    @period_length.setter
    def period_length(self, value):
        self._set_period_length(value)
    
    def _set_period_length(self, value):
        if not torch.is_tensor(value):
            value = torch.as_tensor(value).to(self.raw_period_length)
        self.initialize(raw_period_length=self.raw_period_length_constraint.inverse_transform(value))

    def forward(self, x1, x2, diag=False, **params):

        _x1 = 2 * x1 * torch.pi / self.period_length
        _x2 = 2 * x2 * torch.pi / self.period_length

        x1 = torch.cat((torch.sin(_x1).reshape(-1, 1), torch.cos(_x1).reshape(-1, 1)), axis=1)
        x2 = torch.cat((torch.sin(_x2).reshape(-1, 1), torch.cos(_x2).reshape(-1, 1)), axis=1)

        mean = x1.reshape(-1, x1.size(-1)).mean(0)[(None,) * (x1.dim() - 1)]

        x1_ = (x1 - mean).div(self.lengthscale)
        x2_ = (x2 - mean).div(self.lengthscale)
        distance = self.covar_dist(x1_, x2_, diag=diag, **params)
        exp_component = torch.exp(-math.sqrt(self.nu * 2) * distance)

        if self.nu == 0.5:
            constant_component = 1
        elif self.nu == 1.5:
            constant_component = (math.sqrt(3) * distance).add(1)
        elif self.nu == 2.5:
            constant_component = (math.sqrt(5) * distance).add(1).add(5.0 / 3.0 * distance**2)
        
        return constant_component * exp_component
        