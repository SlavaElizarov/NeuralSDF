from enum import Enum
import math
from turtle import forward
from typing import List, Optional
import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
import numpy as np

from layers.siren import SirenInitScheme

class CESLayer(nn.Module):
    def __init__(self, input_dim: int, 
                 output_dim: int, 
                 bias: bool = True, 
                 omega_0: float = 30,
                init_scheme: SirenInitScheme = SirenInitScheme.SIREN_UNIFORM,
                ):
        """
        https://arxiv.org/pdf/2210.14476.pdf

        Args:
            input_dim (int): _description_
            output_dim (int): _description_
            bias (bool, optional): _description_. Defaults to True.
            omega_0 (int, optional): _description_. Defaults to 30.
        """
        
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.omega_0  = omega_0
        self.is_first = True
        
        weight = torch.Tensor(output_dim, input_dim)
        weight = self._init_siren_lognormal(weight) * omega_0
        
        self.complex_weight = Parameter(torch.complex(torch.cos(weight), torch.sin(weight)), 
                                        requires_grad=True)
                                        
        
        if bias:
            self.bias = Parameter(torch.Tensor(output_dim))
            # fan_in, _ = torch.nn.init._calculate_fan_in_and_fan_out(weight)
            # bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            # torch.nn.init.uniform_(self.bias, -bound, bound)
            torch.nn.init.uniform_(self.bias, -torch.pi, torch.pi)
        else:
            self.register_parameter('bias', None)
            
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        magnitude = self.complex_weight.abs()[None].pow(x[:, None, :]).prod(-1)
                
        y = F.linear(x, self.complex_weight.angle(), self.bias)
        
        return torch.cos(y) * magnitude
    
    
    def _init_siren_lognormal(self, weight):
        # TODO: make std configurable, or calculate the optimal one
        with torch.no_grad():
            out, _ = weight.shape
            weight = weight.log_normal_(std=2.2)
            weight[:out//2] *= -1
            weight /= self.omega_0
        return weight
        
            
    def _init_siren_uniform(self, weight):
        if self.is_first:            
            nn.init.uniform_(
                weight, -1 / self.input_dim, 1 / self.input_dim
            )
        else:
            nn.init.uniform_(
                weight,
                -np.sqrt(6 / self.input_dim) / self.omega_0,
                np.sqrt(6 / self.input_dim) / self.omega_0,
            )
        return weight