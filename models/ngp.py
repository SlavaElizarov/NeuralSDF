import math
from typing import Optional
import tinycudann as tcnn
import torch
from layers.encodings import GridEmbedding

from models.sdf import SDF, GradComputationType, GradientParameters


class NgpSdf(SDF):
    def __init__(self, 
        hidden_dim: int,
        hidden_layers: int,
        out_features: int,
        encoding: GridEmbedding,
        grad_parameters: Optional[GradientParameters] = None,
        ) -> None:
        if grad_parameters is None:
            # use numerical gradients by default since analytical laplacian is not defined
            grad_parameters = GradientParameters(computation_type=GradComputationType.NUMERICAL)

        super().__init__(grad_parameters)
        self.hidden_dim = hidden_dim
        self.hidden_layers = hidden_layers
        self.out_features = out_features
        self.encoding = encoding
        
        assert encoding is not None
        self.encoding = encoding

        network_config={
            "otype": "FullyFusedMLP",
            "activation": "ReLU",
            "output_activation": "None",
            "n_neurons": self.hidden_dim,
            "n_hidden_layers": self.hidden_layers,
        }
        
        network = tcnn.Network(encoding.out_features, self.out_features, network_config)
        self.ngp = torch.nn.Sequential(encoding, network)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.ngp(x)