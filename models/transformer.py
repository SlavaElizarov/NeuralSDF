from typing import List
from enum import Enum

import torch
from torch import nn
import numpy as np
from models.attention import ImplicitAttetionLayer

from models.sdf import SDF
from models.siren import SinActivation, SirenLayer


class InitializationType(Enum):
    SIREN_UNIFORM = 1
    SIREN_NORMAL = 2
    HE_UNIFORM = 3


class SirenLinear(SirenLayer):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        is_first: bool = False,
        omega_0: float = 30,
        initializationType=InitializationType.SIREN_UNIFORM,
    ):
        super().__init__(in_features, out_features, bias, is_first, omega_0)
        self.initializationType = initializationType

        # main difference from SirenLayer is that we don't use activation
        self.activation = lambda x: x

    def init_weights(self):
        if self.initializationType == InitializationType.SIREN_UNIFORM:
            super().init_weights()
        elif self.initializationType == InitializationType.SIREN_NORMAL:
            if self.is_first:
                nn.init.normal_(self.linear.weight, -1 / self.in_features, 1 / self.in_features)
            else:
                nn.init.normal_(
                    self.linear.weight,
                    -np.sqrt(6 / self.in_features) / self.omega_0,
                    np.sqrt(6 / self.in_features) / self.omega_0,
                )


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
        initializationType=InitializationType.SIREN_UNIFORM,
    ):
        super().__init__()

        layers = []
        layers.append(SirenLayer(in_features, hidden_features, is_first=True, omega_0=first_omega_0))

        for i in range(hidden_layers):
            layers.append(
                ImplicitAttetionLayer(
                    n_heads=(i + 1) * 2,
                    input_dim=hidden_features,
                    output_dim=hidden_features,
                    values_projection_factory=lambda _in, _out, _: SirenLinear(
                        _in, _out, is_first=False, omega_0=hidden_omega_0, initializationType=initializationType
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


class BandTransSiren(nn.Sequential, SDF):
    def __init__(
        self,
        in_features: int,
        hidden_features: int,
        hidden_layers: int,
        out_features: int,
        outermost_linear=True,
    ):
        super().__init__()

        layers = []
        # layers.append(SirenLayer(in_features, hidden_features, is_first=True, omega_0=50))
        layers.append(
            ImplicitBandAttetionLayer(
                input_dim=in_features, hidden_dim=hidden_features, omega_0=[5, 15, 30, 60, 90], is_first=True
            )
        )

        for i in range(hidden_layers):
            layers.append(
                ImplicitBandAttetionLayer(input_dim=hidden_features, hidden_dim=hidden_features, is_first=False)
            )

        if outermost_linear:
            final_linear = nn.Linear(hidden_features, out_features)

            nn.init.uniform_(
                final_linear.weight,
                -np.sqrt(6 / hidden_features) / 50,
                np.sqrt(6 / hidden_features) / 50,
            )

            layers.append(final_linear)

        super().__init__(*layers)
