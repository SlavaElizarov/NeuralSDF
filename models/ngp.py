import math
import tinycudann as tcnn
import torch
from layers.encodings import GridEmbedding

from models.sdf import SDF


class NgpSdf(SDF):
    def __init__(self, 
        hidden_dim: int,
        hidden_layers: int,
        out_features: int,
        encoding: GridEmbedding,
        ) -> None:
        super().__init__(analytical_gradient_available = False)
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