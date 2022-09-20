from enum import Enum

from torch import nn
import numpy as np
from models.attention import ImplicitAttetionLayer, ImplicitAttetionLayerLite

from models.sdf import SDF
from models.siren import SinActivation, SirenLayer


class InitializationType(Enum):
    SIREN_UNIFORM = 1
    SIREN_NORMAL = 2
    HE_UNIFORM = 3

class AttentionType(Enum):
    FULL = 1
    LITE = 2

class SirenLinear(SirenLayer):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        is_first: bool = False,
        omega_0: float = 30,
        initializationType:InitializationType=InitializationType.SIREN_UNIFORM,
    ):
        self.initializationType = initializationType
        super().__init__(in_features, out_features, bias, is_first, omega_0)

        # main difference from SirenLayer is that we don't use activation
        self.activation = nn.Identity()

    def init_weights(self):
        if self.initializationType == InitializationType.SIREN_UNIFORM:
            super().init_weights()
        elif self.initializationType == InitializationType.SIREN_NORMAL:
            if self.is_first:
                raise ValueError("SirenNormal can't be used with first layer")
            else:
                nn.init.normal_(self.linear.weight, 0, np.sqrt(2 / self.in_features) / self.omega_0)



class TransSiren(nn.Sequential, SDF):
    def __init__(
        self,
        in_features: int,
        hidden_features: int,
        hidden_layers: int,
        out_features: int,
        outermost_linear=False,
        first_omega_0=30,
        hidden_omega_0=30.0,
        initializationType:InitializationType=InitializationType.SIREN_UNIFORM,
        attention_type: AttentionType= AttentionType.FULL,
    ):
        super().__init__()
        
        if isinstance(initializationType, str):
            initializationType = InitializationType[initializationType]
        if isinstance(attention_type, str):
            attention_type = AttentionType[attention_type]

        layers = []
        layers.append(SirenLayer(in_features, hidden_features, is_first=True, omega_0=first_omega_0))
        
        attention_layer = ImplicitAttetionLayer if attention_type == AttentionType.FULL else ImplicitAttetionLayerLite

        for i in range(hidden_layers):
            layers.append(
                attention_layer(
                    n_heads=(i + 1) * 2, #2**(i+1),#(i + 1) * 2
                    input_dim=hidden_features,
                    output_dim=hidden_features,
                    values_projection_factory=lambda _in, _out, _: SirenLinear(
                        _in, _out, 
                        is_first=False, 
                        omega_0=hidden_omega_0, 
                        initializationType=initializationType
                    ),
                )
            )
            layers.append(SinActivation(omega_0=hidden_omega_0))

        if outermost_linear:
            final_linear = nn.Linear(hidden_features, out_features)
            layers.append(final_linear)
        else:
            layers.append(
                SirenLayer(
                    hidden_features,
                    out_features,
                    is_first=False,
                    omega_0=hidden_omega_0,
                )
            )

        super().__init__(*layers)
