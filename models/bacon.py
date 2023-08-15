from typing import List
import numpy as np
import torch
from torch import nn
from layers import BaconFourierLayer
from layers.initializers import SirenUniformInitializer
from layers.siren import ComplexExpLayer

from models.sdf import SDF

def mfn_weights_init(m):
    with torch.no_grad():
        if hasattr(m, 'weight'):
            num_input = m.weight.size(-1)
            m.weight.uniform_(-np.sqrt(6/num_input), np.sqrt(6/num_input))

class Bacon(SDF):
    def __init__(
        self,
        in_features: int,
        hidden_dim: int,
        hidden_layers: int,
        out_features: int,
        omega_max: float = 1024,
        quantization: int = 8,
    ) -> None:
        super().__init__()

        self.in_features = in_features
        self.hidden_dim = hidden_dim
        self.hidden_layers = hidden_layers
        self.out_features = out_features
        self.omega_max = omega_max

        omega_per_layer = omega_max / self.hidden_layers

        fourier_layers = []
        linear_layers = []
        projection_layers = []

        for i in range(hidden_layers):
            fourier_layers.append(
                BaconFourierLayer(
                    in_features=in_features,
                    out_features=hidden_dim,
                    add_bias=True,
                    quantization=quantization,
                    max_freq=omega_per_layer,
                )
            )
            if i != 0:
                linear_layers.append(nn.Linear(hidden_dim, hidden_dim))
                projection_layers.append(nn.Linear(hidden_dim, out_features))

        self.fourier_layers = nn.ModuleList(fourier_layers)
        self.linear_layers = nn.ModuleList(linear_layers)
        self.projection_layers = nn.ModuleList(projection_layers)

        # self.linear_layers.apply(mfn_weights_init)
        # self.projection_layers.apply(mfn_weights_init)
        # self.final_layer = nn.Linear(hidden_dim, out_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        values: List[torch.Tensor] = []
        z = self.fourier_layers[0](x)
        for i in range(1, self.hidden_layers):
            l = self.linear_layers[i - 1](z)
            g = self.fourier_layers[i](x)
            z = l * g
            val = self.projection_layers[i - 1](z)
            values.append(val)
        # index = np.random.randint(1, len(values))
        return values[-1]


class ComplexBacon(SDF):
    def __init__(
        self,
        in_features: int,
        hidden_dim: int,
        hidden_layers: int,
        out_features: int,
        omega_max: float = 1024,
    ) -> None:
        super().__init__()

        self.in_features = in_features
        self.hidden_dim = hidden_dim
        self.hidden_layers = hidden_layers
        self.out_features = out_features
        self.omega_max = omega_max

        omega_per_layer = omega_max / self.hidden_layers

        fourier_layers = []
        linear_layers = []
        for i in range(hidden_layers):
            fourier_layers.append(
                ComplexExpLayer(
                    in_features=in_features,
                    out_features=hidden_dim,
                    add_bias=True,
                    init_scheme=SirenUniformInitializer(
                        omega=omega_per_layer, is_first=True
                    ),
                )
            )
            if i != 0:
                linear_layers.append(nn.Linear(hidden_dim, hidden_dim))

        self.fourier_layers = nn.ModuleList(fourier_layers)
        self.linear_layers = nn.ModuleList(linear_layers)
        self.final_layer = nn.Linear(hidden_dim, out_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + 1.0

        z = self.fourier_layers[0](x)
        for i in range(1, self.hidden_layers):
            l = self.linear_layers[i - 1](z)
            g = self.fourier_layers[i](x)
            z = l * g

        return self.final_layer(z)
