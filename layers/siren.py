from audioop import add
from enum import Enum
from typing import Optional, Set
import torch
from torch import nn
import numpy as np


class SirenInitScheme(Enum):
    SIREN_UNIFORM = 1
    SIREN_NORMAL = 2
    HE_UNIFORM = 3


class SirenBiasInitScheme(Enum):
    NORMAL = 1
    ZEROS = 2


class SirenModulationType(Enum):
    FILM = 1
    PFA = 2


class ModulateArg(Enum):
    Amplitude = (0,)
    Frequency = (1,)
    Phase = 2


class SirenLayer(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        add_bias: bool = True,
        is_first: bool = False,
        omega_0: float = 30,
        init_scheme: SirenInitScheme = SirenInitScheme.SIREN_UNIFORM,
        bias_init_scheme: SirenBiasInitScheme = SirenBiasInitScheme.ZEROS,
        modulation_type: SirenModulationType = SirenModulationType.FILM,
        disable_activation: bool = False,
        self_modulate: Set[ModulateArg] = set(),
    ):
        """Siren layer.

            Dense layer with sin activation.
            Described in paper: https://arxiv.org/abs/2006.09661

            y = sin(omega_0 * (Wx + b))

            omega_0 is regulating frequency of the output signal.
            For details see paragraph 3.2 in the paper.

            If you want to use this layer to represent results from the paper
            keep defaults for parameters folowing omega_0.

            There are two ways to modulate output of the layer:
            1) FILM (Feature-wise Linear Modulation) proposed in https://arxiv.org/abs/1709.07871
                This method shown to be effective for neural fileds in https://arxiv.org/abs/2201.12904

                h = Wx + b
                y = FILM(scale, shift) = scale * h + shift
                y = sin(omega_0 * y)

            2) PFA (Phase-Frequency-Amplitude)
                Idea of PFA is to modulate frequency and amplitude of the output signal explicitly.

                h = Wx * omega_0
                y = PFA(scale, shift) = scale * h + b + shift
                y = sin(y)

                Which is equivalent to:
                y = sin((Omega_0 * scale) * Wx + (b + shift))

                Where (Omega_0 * scale) term represents frequency, and (b + shift) is phase.

                To modulate amplitude multiply output of the layer by coefficient.
                y = SirenLayer(x, frequency, phase) * amplitude


        Args:
            input_dim (int): Number of input features.
            output_dim (int): Number of output features.
            add_bias (bool, optional): Use bias? Defaults to True.
            is_first (bool, optional): Is first? Initialzation depends on this parameter.
                                       See 3.2 of the paper. Defaults to False.
            omega_0 (float, optional): omega_0 is a frequency factor which simply multiplies
                                        the features before the nonlinearity.
                                        Different signals may require different omega_0 in the first layer.
                                        Defaults to 30.
            initScheme (SirenInitScheme, optional): See 3.2 of the paper. Defaults to SirenInitScheme.SIREN_UNIFORM.
            biasInitScheme (SirenBiasInitScheme, optional): For PFA preferably to use Normal init. Defaults to SirenBiasInitScheme.ZEROS.
            modulationType (SirenModulationType, optional): FILM or PFA, see description for details. Defaults to SirenModulationType.FILM.
            disable_activation (bool, optional): Make layer linear, this option is specific for some architectures. Defaults to False.
        """
        super().__init__()
        self.omega_0 = omega_0
        self.is_first = is_first
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.add_bias = add_bias
        self.initScheme = init_scheme
        self.biasInitScheme = bias_init_scheme
        self.modulationType = modulation_type
        self.disable_activation = disable_activation
        self.self_modulate = self_modulate

        if (
            ModulateArg.Phase in self_modulate
            and add_bias
            and modulation_type == modulation_type.PFA
        ):
            ValueError("PFA is already self-modulated by bias")
        assert not (
            ModulateArg.Amplitude in self_modulate and disable_activation
        ), "Can't self-modulate amplitude without activation"

        self.linear = nn.Linear(input_dim, output_dim, bias=False)
        if add_bias:
            self.bias = nn.Parameter(torch.zeros(output_dim))

        self.scale = None
        self.amplitude = None
        self.shift = None
        if ModulateArg.Frequency in self_modulate:
            self.scale = nn.Parameter(torch.randn((1, output_dim), dtype=torch.float32))
        if ModulateArg.Amplitude in self_modulate:
            self.amplitude = nn.Parameter(
                torch.randn((1, output_dim), dtype=torch.float32)
            )
        if ModulateArg.Phase in self_modulate:
            self.shift = nn.Parameter(torch.randn((1, output_dim), dtype=torch.float32))

        self._init_weights()

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

        So, scale and shift should modulate frequency and amplitude of the output signal.

        But final formula looks like this:
        Or y = sin(omega_0 * (Wx * scale + b * scale + shift))  strange, isn't?
        # TODO: Investigate modulation more thorougly.

        Args:
            x (torch.Tensor): _description_
            scale (Optional[torch.Tensor], optional): Scale. Defaults to None.
            shift (Optional[torch.Tensor], optional): Shift. Defaults to None.

        Returns:
            torch.Tensor: _description_
        """

        if scale is None:
            scale = self.scale
        elif self.scale is not None:
            scale = scale * self.scale

        if shift is None:
            shift = self.shift
        elif self.shift is not None:
            shift = shift + self.shift

        if self.modulationType == SirenModulationType.FILM:
            y = self._forward_film(x, scale, shift)
        elif self.modulationType == SirenModulationType.PFA:
            y = self._forward_pfa(x, scale, shift)
        else:
            raise NotImplementedError("Unknown modulationType")

        return y if self.amplitude is None else y * self.amplitude

    def _init_siren_uniform(self):
        if self.is_first:
            nn.init.uniform_(
                self.linear.weight, -1 / self.input_dim, 1 / self.input_dim
            )
        else:
            nn.init.uniform_(
                self.linear.weight,
                -np.sqrt(6 / self.input_dim) / self.omega_0,
                np.sqrt(6 / self.input_dim) / self.omega_0,
            )

    def _init_siren_normal(self):
        if self.is_first:
            raise ValueError(
                "SirenNormal can't be used for first layer"
            )  # TODO: can be?
        else:
            nn.init.normal_(
                self.linear.weight, 0, np.sqrt(2 / self.input_dim) / self.omega_0
            )

    def _init_weights(self):
        if self.initScheme == SirenInitScheme.SIREN_UNIFORM:
            self._init_siren_uniform()
        elif self.initScheme == SirenInitScheme.SIREN_NORMAL:
            self._init_siren_normal()
        else:
            raise NotImplementedError("Unknown initScheme")
        if self.add_bias:
            if self.biasInitScheme == SirenBiasInitScheme.NORMAL:
                torch.nn.init.normal_(self.bias, mean=0.0, std=torch.pi / 2)
            elif self.biasInitScheme != SirenBiasInitScheme.ZEROS:
                raise NotImplementedError("Unknown biasInitScheme")

    def _forward_film(
        self,
        x: torch.Tensor,
        scale: Optional[torch.Tensor] = None,
        shift: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        y = self.linear(x)

        if self.add_bias:
            y = y + self.bias

        if scale is not None:
            y = y * scale

        if shift is not None:
            y = y + shift

        if self.disable_activation:
            return y

        return torch.sin(self.omega_0 * y)

    def _forward_pfa(
        self,
        x: torch.Tensor,
        scale: Optional[torch.Tensor] = None,
        shift: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        y = self.linear(x) * self.omega_0

        if scale is not None:
            y = y * scale

        if self.add_bias:
            y = y + self.bias

        if shift is not None:
            y = y + shift

        if self.disable_activation:
            raise ValueError("PFA can't be used with disable_activation")

        return torch.sin(y)
