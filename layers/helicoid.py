import math
import torch
from torch import nn
from torch.nn import Parameter

from layers.initializers import SirenInitializer, SirenUniformInitializer


class HelicoidLayer(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        add_bias: bool = True,
        init_scheme: SirenInitializer = SirenUniformInitializer(),
    ):
        """
        Siren-compatible layer with complex weights inspired by https://arxiv.org/abs/2210.14476

        Args:
            in_features (int):  Number of input features.
            out_features (int): Number of output features.
            bias (bool, optional): Add bias. Defaults to True.
        """

        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.omega = init_scheme.omega

        frequency = torch.Tensor(out_features, in_features)
        frequency = init_scheme(frequency) * self.omega
        self.complex_weight = Parameter(1j * frequency, requires_grad=True)

        if add_bias:
            self.bias = Parameter(torch.Tensor(out_features), requires_grad=True)
            # initialization copied from nn.Linear
            # TODO: investigate different initialization schemes
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(frequency)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)
        else:
            self.register_parameter("bias", None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + 1.0  # TODO: remove this hack

        # helicoid parametrized by complex number, where imag part is frequency
        # this parametrization is crucial since it allows network to control frequency
        # indirectrly nudging z.abs()
        # TODO: rigourous proof is needed
        z = torch.exp(self.complex_weight)

        y = torch.cos(
            torch.nn.functional.linear(x, self.complex_weight.imag, self.bias)
            # * self.omega
        )

        # log-exp trick to reduce memory footprint
        radius = z.abs()
        powers = torch.log(radius)
        powers = torch.nn.functional.linear(x, powers)
        radius = torch.exp(powers)

        return y * radius
