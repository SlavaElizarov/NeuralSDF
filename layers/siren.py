from enum import Enum
from typing import List, Optional
import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
import numpy as np


class SirenInitScheme(Enum):
    SIREN_UNIFORM = 1
    SIREN_NORMAL = 2
    HE_UNIFORM = 3
    SIREN_LOGNORMAL = 4


class SirenBiasInitScheme(Enum):
    NORMAL = 1
    ZEROS = 2
    HE_UNIFORM = 3


class SirenModulationType(Enum):
    FILM = 1
    PFA = 2


class ModulateArg(Enum):
    Amplitude = 0
    Frequency = 1
    Phase = 2


class SirenLayer(nn.Linear):
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
        self_modulate: List[ModulateArg] = [],
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
            self_modulate (Set[ModulateArg], optional): Add learnable vectors as modulation. Defaults to {}.
        """
        super().__init__(
            in_features=input_dim,
            out_features=output_dim,
            bias=add_bias,
            device=device,
            dtype=dtype,
        )
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
            ValueError("If PFA is on, phase is already self-modulated by bias")
            
        assert not (
            ModulateArg.Amplitude in self_modulate 
            and disable_activation
        ), "Can't self-modulate amplitude without activation"
        
        # Self-modulation parameters
        self.scale = None
        self.amplitude = None
        self.shift = None
        if ModulateArg.Frequency in self_modulate:
            self.scale = Parameter(torch.randn((1, output_dim), dtype=torch.float32), requires_grad=True)
        if ModulateArg.Amplitude in self_modulate:
            self.amplitude = Parameter(torch.randn((1, output_dim), dtype=torch.float32), requires_grad=True)
        if ModulateArg.Phase in self_modulate:
            phase_init = torch.randn((1, output_dim), dtype=torch.float32) * torch.pi / 3
            self.shift = Parameter(phase_init, requires_grad=True)
            
        # Choosing forward implementation depending on modulation type
        if self.modulationType == SirenModulationType.FILM:
            self._forward = self._forward_film
        elif self.modulationType == SirenModulationType.PFA:
            self._forward = self._forward_pfa
        else:
            raise NotImplementedError("Unknown modulationType")
        
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

        1) If FILM is used: 
            y = sin(omega_0 * ((Wx + b) * scale + shift))
        2) If PFA is used:
            y = sin((omega_0 * scale) * Wx + (b + shift))
            
        If self_modulate is enabled for frequency, amplitude or phase,
        First, the layer is modulated by learned vectors and then external modulation is applied.
        
        Args:
            x (torch.Tensor): input tensor
            scale (Optional[torch.Tensor], optional): Scale. Defaults to None.
            shift (Optional[torch.Tensor], optional): Shift. Defaults to None.

        Returns:
            torch.Tensor: output tensor
        """

        if scale is None:
            scale = self.scale
        elif self.scale is not None:
            scale = scale * self.scale

        if shift is None:
            shift = self.shift
        elif self.shift is not None:
            shift = shift + self.shift

        y = self._forward(x, scale, shift)

        return y if self.amplitude is None else y * self.amplitude
    
    def _init_siren_uniform(self):
        if self.is_first:            
            nn.init.uniform_(
                self.weight, -1 / self.input_dim, 1 / self.input_dim
            )
        else:
            nn.init.uniform_(
                self.weight,
                -np.sqrt(6 / self.input_dim) / self.omega_0,
                np.sqrt(6 / self.input_dim) / self.omega_0,
            )

    def _init_siren_normal(self):
        if self.is_first:
            nn.init.normal_(self.weight, 0, 1 / np.sqrt(3) / self.input_dim)
        else:
            nn.init.normal_(
                self.weight, 0, np.sqrt(2 / self.input_dim) / self.omega_0
            )

    def _init_siren_lognormal(self):
        if self.is_first:
            # TODO: make std configurable, or calculate the optimal one
            with torch.no_grad():
                out, _ = self.weight.shape
                self.weight = self.weight.log_normal_(std=2.2)
                self.weight[:out//2] *= -1
                self.weight /= self.omega_0
        else:
            # TODO: I decided to use uniform init for the rest of the layers, not sure if it's correct
            nn.init.uniform_(
                self.weight,
                -np.sqrt(6 / self.input_dim) / self.omega_0,
                np.sqrt(6 / self.input_dim) / self.omega_0,
            )

    def _init_weights(self):
        if self.initScheme == SirenInitScheme.SIREN_UNIFORM:
            self._init_siren_uniform()
        elif self.initScheme == SirenInitScheme.SIREN_NORMAL:
            self._init_siren_normal()
        elif self.initScheme == SirenInitScheme.SIREN_LOGNORMAL:
            self._init_siren_lognormal()
        elif self.initScheme == SirenInitScheme.HE_UNIFORM:
            pass # Linear layer is already initialized with He, see super().__init__
        else:
            raise NotImplementedError("Unknown initScheme")
        if self.add_bias:
            if self.biasInitScheme == SirenBiasInitScheme.NORMAL:
                torch.nn.init.normal_(self.bias, mean=0.0, std=torch.pi / 3)
            elif self.biasInitScheme == SirenBiasInitScheme.ZEROS:
                torch.nn.init.zeros_(self.bias)
            elif self.biasInitScheme == SirenBiasInitScheme.HE_UNIFORM:
                pass # Bias of linear layer is already initialized with He, see super().__init__
            else:
                raise NotImplementedError("Unknown biasInitScheme")

    def _forward_film(
        self,
        x: torch.Tensor,
        scale: Optional[torch.Tensor] = None,
        shift: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        y = F.linear(x, self.weight, self.bias)
        
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
        y = F.linear(x, self.weight) * self.omega_0

        if scale is not None:
            y = y * scale

        if self.add_bias:
            y = y + self.bias

        if shift is not None:
            y = y + shift

        if self.disable_activation:
            raise ValueError("PFA can't be used with disable_activation")

        return torch.sin(y)
