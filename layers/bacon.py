import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
import numpy as np


class BaconFourierLayer(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        add_bias: bool = True,
        quantization: int = 8,
        max_freq: float = 1024,
    ) -> None:
        """
        Bacon Fourier layer

        Autors implementation of the Bacon Fourier Layer:
        https://github.com/computational-imaging/bacon/blob/main/modules.py#L136

        Args:
            in_features (int): Number of input features.
            out_features (int): Number of output features.
            add_bias (bool, optional): Add bias. Defaults to True.
            quantization (int, optional): Number of uniform quants in the band. Defaults to 8.
            max_freq (float, optional): Max frequency of the band. Defaults to 1024.
        """
        super().__init__()

        frequency_quant = max_freq / quantization
        quant_indeces = torch.randint(
            -quantization, quantization, (out_features, in_features)
        )
        weight = quant_indeces * frequency_quant
        self.weight = Parameter(weight, requires_grad=False)
        self.bias = None

        if add_bias:
            self.bias = Parameter(torch.empty(out_features), requires_grad=True)
            torch.nn.init.uniform_(self.bias, -np.pi, np.pi)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.linear(x, self.weight, self.bias)
        return torch.sin(x)
