import math
import torch
from torch import nn
from torch.nn import Parameter

from layers.initializers import (
    SirenInitializer,
    SirenUniformInitializer,
)


class ComplexExpLayer(nn.Module):
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
        bias = self.bias * 1j if self.bias is not None else None
        z = torch.nn.functional.linear(
            torch.complex(x, torch.zeros_like(x)), self.complex_weight, bias
        )
        return torch.exp(z)


class WireLayer(nn.Linear):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        add_bias: bool = True,
        gaussian_windows: int = 1,
        s_0: float = 1.0,
        return_real: bool = False,
        init_scheme: SirenInitializer = SirenUniformInitializer(),
    ):
        """
        WIRE layer with complex weights. Based on https://arxiv.org/abs/2301.05187
        WIRE is a SIREN-compatible layer with complex weights and Gaussian windows.

        Args:
            in_features (int): input features
            out_features (int): output features
            add_bias (bool, optional): use bias. Defaults to True.
            gaussian_windows (int, optional): Number of gaussian windows. For details see https://arxiv.org/abs/2301.05187, paragraph 3.5. Defaults to 1.
            s_0 (float, optional): Scaling factor for the Gaussian windows. Defaults to 1.0.
            return_real (bool, optional): Return real part of the output. Defaults to False.
            init_scheme (SirenInitializer, optional): Initialization scheme. Defaults to SirenUniformInitializer().
        """
        super().__init__(
            in_features,
            out_features * gaussian_windows,
            add_bias,
            dtype=torch.complex64,
        )

        self.gaussian_windows = gaussian_windows
        self.s_0 = s_0
        self.omega = init_scheme.omega
        self.return_real = return_real

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dtype == torch.float32 or x.dtype == torch.float16:
            x = torch.complex(x, torch.zeros_like(x))

        batch_size, features = x.shape
        assert (
            features == self.in_features
        ), f"Input features {features} do not match layer in_features {self.in_features}"

        y = super().forward(x)
        y = y.view(batch_size, self.gaussian_windows, -1)
        y = torch.exp(
            self.omega * y[:, 0, :] - torch.sum(torch.abs(self.s_0 * y) ** 2, dim=1)
        )

        return y.real if self.return_real else y
