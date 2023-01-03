import math
from typing import Optional
import torch
from torch import nn
from torch.nn import Parameter
import torch.nn.functional as F

from layers.initializers import (
    Initializer,
    SirenInitializer,
    SirenUniformInitializer,
)


class SirenLayer(nn.Linear):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        add_bias: bool = True,
        init_scheme: SirenInitializer = SirenUniformInitializer(),
        disable_activation: bool = False,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        """Siren layer.

            Dense layer with sin activation.
            Described in paper: https://arxiv.org/abs/2006.09661

            y = sin(omega_0 * (Wx + b))

            omega_0 is regulating frequency of the output signal.
            For details see paragraph 3.2 in the paper.

            If you want to use this layer to represent results from the paper
            keep defaults for parameters folowing omega_0.

            Siren layer is modulated by
            FILM (Feature-wise Linear Modulation) proposed in https://arxiv.org/abs/1709.07871
                This method shown to be effective for neural fileds in https://arxiv.org/abs/2201.12904

                h = Wx + b
                y = FILM(scale, shift) = scale * h + shift
                y = sin(omega_0 * y)

        Args:
            in_features (int): Number of input features.
            out_features (int): Number of output features.
            add_bias (bool, optional): Use bias? Defaults to True.
            init_scheme (Initializer, optional): See 3.2 of the paper. Defaults to SirenUniformInitializer.
            disable_activation (bool, optional): Make layer linear, this option is specific for some architectures. Defaults to False.
        """
        super().__init__(
            in_features=in_features,
            out_features=out_features,
            bias=add_bias,
            device=device,
            dtype=dtype,
        )
        self.in_features = in_features
        self.out_features = out_features
        self.add_bias = add_bias
        self.disable_activation = disable_activation
        self.omega = init_scheme.omega

        init_scheme(self.weight)

    def forward(
        self,
        x: torch.Tensor,
        scale: Optional[torch.Tensor] = None,
        shift: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Apply Siren layer to input.
        Scale and shift are optional parameters which are used to modulate the output of the layer.

        y = sin(omega_0 * ((Wx + b) * scale + shift))

        Args:
            x (torch.Tensor): input tensor
            scale (Optional[torch.Tensor], optional): Scale. Defaults to None.
            shift (Optional[torch.Tensor], optional): Shift. Defaults to None.

        Returns:
            torch.Tensor: output tensor
        """
        y = F.linear(x, self.weight, self.bias)

        if scale is not None:
            y = y * scale

        if shift is not None:
            y = y + shift

        if self.disable_activation:
            return y

        return torch.sin(y * self.omega)


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
        return torch.exp(z).real
