import torch
from torch import nn
from layers import BaconFourierLayer

from models.sdf import SDF


class Bacon(SDF):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        hidden_layers: int,
        output_dim: int,
        omega_max: float = 1024,
        quantization: int = 8
    ) -> None:
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.hidden_layers = hidden_layers
        self.output_dim = output_dim
        self.omega_max = omega_max
        
        omega_per_layer = omega_max / self.hidden_layers
        
        fourier_layers = []
        linear_layers = []
        for i in range(hidden_layers):
            fourier_layers.append(
                BaconFourierLayer(
                    input_dim=input_dim,
                    output_dim=hidden_dim,
                    add_bias=True,
                    quantization=quantization,
                    max_freq=omega_per_layer,
                )
            )
            if i != 0:
                linear_layers.append(nn.Linear(hidden_dim, hidden_dim))
            
        
        self.fourier_layers = nn.ModuleList(fourier_layers)
        self.linear_layers = nn.ModuleList(linear_layers)
        self.final_layer = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.fourier_layers[0](x)
        for i in range(1, self.hidden_layers):
            l = self.linear_layers[i - 1](z)
            g = self.fourier_layers[i](x)
            z = l * g
            
        return self.final_layer(z)
