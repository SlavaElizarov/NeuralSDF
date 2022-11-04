import math
import torch
from torch import nn
from torch.nn.parameter import Parameter
import numpy as np


class CESLayer(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        add_bias: bool = True,
        is_first: bool = True,
        omega_0: float = 30,
    ):
        """
        Siren-compatible layer with complex weights inspired by https://arxiv.org/abs/2210.14476

        Args:
            input_dim (int):  Number of input features.
            output_dim (int): Number of output features.
            bias (bool, optional): Add bias. Defaults to True.
            omega_0 (int, optional): Kind of a frequency. Defaults to 30.
        """

        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.omega_0 = omega_0
        self.is_first = is_first

        frequency = torch.Tensor(output_dim, input_dim)

        frequency = self._init_siren_uniform(frequency)
        self.complex_weight = Parameter(torch.exp(1j * frequency), requires_grad=True)

        if add_bias:
            self.bias = torch.Tensor(output_dim)
            fan_in, _ = torch.nn.init._calculate_fan_in_and_fan_out(frequency)
            torch.nn.init.uniform_(
                self.bias, -torch.pi / math.sqrt(fan_in), torch.pi / math.sqrt(fan_in)
            )
            self.bias = Parameter(
                torch.exp(torch.complex(torch.zeros_like(self.bias), self.bias)),
                requires_grad=True,
            )

        else:
            self.register_parameter("bias", None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = (x[:, None, :] + 1) * self.omega_0
        y = self.complex_weight[None] ** x
        y = torch.prod(y, dim=-1)
        y = y * self.bias

        return y.real

    def _init_siren_uniform(self, weight):
        if self.is_first:
            nn.init.uniform_(
                weight, -torch.pi / self.input_dim, torch.pi / self.input_dim
            )
        else:
            nn.init.uniform_(
                weight,
                -np.sqrt(6 / self.input_dim) / self.omega_0,
                np.sqrt(6 / self.input_dim) / self.omega_0,
            )
        return weight
