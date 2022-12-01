from enum import Enum
from typing import List
import torch
from torch import nn
from torch.nn.parameter import Parameter
from layers import CrossAttentionLayer, SubtractionCrossAttentionLayer
from layers import (
    SirenLayer,
)
from layers.helicoid import HelicoidLayer
from layers.initializers import SirenInitializer, SirenUniformInitializer

from models.sdf import SDF


class AttentionType(Enum):
    DOT = 1
    SUBTRACTION = 2


class ModulateArg(Enum):
    Amplitude = 0
    Frequency = 1
    Phase = 2


class Siren(nn.Sequential, SDF):
    def __init__(
        self,
        in_features: int,
        hidden_dim: int,
        hidden_layers: int,
        out_features: int,
        outermost_linear: bool = False,
        first_layer_init: SirenInitializer = SirenUniformInitializer(
            omega=30.0, is_first=True
        ),
        hidden_layer_init: SirenInitializer = SirenUniformInitializer(
            omega=30.0, is_first=False
        ),
    ):
        """
            Siren model described in paper: https://arxiv.org/abs/2006.09661

        Args:
            in_features (int): Number of input features.
            hidden_dim (int): Number of hidden features.
            hidden_layers (int): Number of hidden layers.
            out_features (int): Number of output features.
            outermost_linear (bool, optional): Is final layer linear?. Defaults to False.
            init_scheme (SirenInitializer, optional): See 3.2 of the paper. Defaults to SirenUniformInitializer.
        """
        super().__init__()
        layers = []

        for i in range(hidden_layers):
            is_first = i == 0
            layers.append(
                SirenLayer(
                    in_features if is_first else hidden_dim,
                    hidden_dim,
                    init_scheme=first_layer_init if is_first else hidden_layer_init,
                )
            )
        l = SirenLayer(
            hidden_dim, out_features, add_bias=True, disable_activation=outermost_linear
        )
        nn.init.zeros_(l.bias)
        layers.append(l)

        super().__init__(*layers)


class TransposedAttentionSiren(Siren):
    def __init__(
        self,
        in_features: int,
        hidden_dim: int,
        hidden_layers: int,
        out_features: int,
        latent_seq_len: int = 64,
        use_dropout: bool = True,
        attention_dim: int = 64,
        attention_type: AttentionType = AttentionType.DOT,
        attention_modulate: List[ModulateArg] = [ModulateArg.Amplitude],
        first_layer_init: SirenInitializer = SirenUniformInitializer(
            omega=30.0, is_first=True
        ),
        hidden_layer_init: SirenInitializer = SirenUniformInitializer(
            omega=30.0, is_first=False
        ),
    ):
        super().__init__(
            in_features,
            hidden_dim,
            hidden_layers,
            out_features,
            True,
            first_layer_init,
            hidden_layer_init,
        )

        assert len(attention_modulate) > 0, "Should modulate at least one parameter"

        self.hidden_layers = hidden_layers
        self.modulate = attention_modulate

        number_of_heads = (hidden_layers - 1) * len(attention_modulate)
        if ModulateArg.Amplitude in attention_modulate:
            number_of_heads += 1

        if attention_type == AttentionType.DOT:
            attention_layer = CrossAttentionLayer
        elif attention_type == AttentionType.SUBTRACTION:
            attention_layer = SubtractionCrossAttentionLayer
        else:
            raise NotImplementedError("Attention type not implemented")

        self.attention = nn.ModuleDict()
        for modulate in attention_modulate:
            number_of_heads = hidden_layers - 1
            if modulate == ModulateArg.Amplitude:
                number_of_heads += 1

            self.attention[modulate.name] = attention_layer(
                first_in_features=hidden_dim,
                second_in_features=hidden_dim,
                attention_dim=attention_dim,
                number_of_heads=number_of_heads,
                value_dim=hidden_dim,
                use_dropout=use_dropout,
            )

        self.latent = Parameter(
            torch.randn((1, latent_seq_len, hidden_dim), dtype=torch.float32),
            requires_grad=True,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self[0](x)

        modulation_dict = {
            key: layer(x, self.latent) for key, layer in self.attention.items()
        }

        if ModulateArg.Amplitude in self.modulate:
            x = x * modulation_dict[ModulateArg.Amplitude.name][:, 0, 0, :]

        for i in range(1, self.hidden_layers):
            layer = self[i]
            frequency_mod = None
            if ModulateArg.Frequency.name in modulation_dict:
                frequency_mod = modulation_dict[ModulateArg.Frequency.name][
                    :, 0, i - 1, :
                ]

            phase_mod = None
            if ModulateArg.Phase.name in modulation_dict:
                phase_mod = modulation_dict[ModulateArg.Phase.name][:, 0, i - 1, :]

            x = layer(x, scale=frequency_mod, shift=phase_mod)

            if ModulateArg.Amplitude.name in modulation_dict:
                x = x * modulation_dict[ModulateArg.Amplitude.name][:, 0, i, :]

        return self[self.hidden_layers](x)


class ComplexSiren(Siren):
    def __init__(
        self,
        in_features: int,
        hidden_dim: int,
        hidden_layers: int,
        out_features: int,
        outermost_linear: bool = False,
        first_layer_init: SirenInitializer = SirenUniformInitializer(
            omega=30.0, is_first=True
        ),
        hidden_layer_init: SirenInitializer = SirenUniformInitializer(
            omega=30.0, is_first=False
        ),
    ):
        super().__init__(
            in_features,
            hidden_dim,
            hidden_layers,
            out_features,
            outermost_linear,
            first_layer_init,
            hidden_layer_init,
        )

        first_layer = HelicoidLayer(
            in_features=in_features,
            out_features=hidden_dim,
            init_scheme=first_layer_init,
        )
        self[0] = first_layer
