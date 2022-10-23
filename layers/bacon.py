from enum import Enum
from typing import List, Optional
import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
import numpy as np


class BaconFourierLayer(nn.Module):
    def __init__(
        self, input_dim: int, output_dim: int, add_bias: bool = True, quantization: int = 8, max_freq: float = 1024
    ) -> None:
        """_summary_

        Autors implementation of the Bacon Fourier Layer:
        https://github.com/computational-imaging/bacon/blob/main/modules.py#L136

        Args:
            input_dim (int): _description_
            output_dim (int): _description_
            add_bias (bool, optional): _description_. Defaults to True.
            quantization (int, optional): _description_. Defaults to 8.
            max_freq (float, optional): _description_. Defaults to 1024.
        """
        super().__init__()

        frequency_quant = max_freq / quantization
        quant_indeces = torch.randint(-quantization, quantization, (output_dim, input_dim)) 
        weight = quant_indeces * frequency_quant
        self.weight = Parameter(weight, requires_grad=False)
        self.bias = None

        if add_bias:
            self.bias = Parameter(torch.empty(output_dim), requires_grad=True)
            torch.nn.init.uniform_(self.bias, -np.pi, np.pi)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.linear(x, self.weight, self.bias)
        return torch.sin(x)
