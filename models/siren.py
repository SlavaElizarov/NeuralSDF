import numpy as np
import torch
from torch import nn
import tinycudann as tcnn

from layers import SirenLayer, ComplexExpLayer
from layers.initializers import SirenInitializer, SirenUniformInitializer
from models.sdf import SDF


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
        self.hidden_dim = hidden_dim
        self.hidden_layers = hidden_layers

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
        final_layer = SirenLayer(
            hidden_dim, out_features, add_bias=True, disable_activation=outermost_linear
        )
        nn.init.zeros_(final_layer.bias)
        layers.append(final_layer)

        super().__init__(*layers)


class ModulatedSiren(Siren):
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
        embedding_resolution: int = 32,
        embedding_features: int = 64,
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

        self.embedding_resolution = embedding_resolution
        self.embedding_features = embedding_features
        # self.embedding = nn.Parameter(torch.zeros(1, embedding_features, embedding_resolution, embedding_resolution, embedding_resolution), 
        #                               requires_grad=True)
        # nn.init.normal_(self.embedding, 0, 1e-2)
        encoding_config={
                    "otype": "HashGrid" ,
                    "n_levels": 16,
                    "n_features_per_level": 8,
                    "log2_hashmap_size": 19,
                    "base_resolution": 16,
                    "per_level_scale": 2.0,
                    "interpolation": "Smoothstep",
                }
        self.encoding = tcnn.Encoding(in_features, encoding_config, dtype=torch.float32)


        projection_layers =[]
        for i in range(self.hidden_layers):
            projection_layer = nn.Linear(embedding_features, hidden_dim, bias=True)
            nn.init.zeros_(projection_layer.bias)
            projection_layers.append(projection_layer)
        self.projection_layers = nn.ModuleList(projection_layers)
        # self.projection_layer = nn.Linear(embedding_features, hidden_dim, bias=True)
        # nn.init.zeros_(self.projection_layer.weight)
        # nn.init.ones_(self.projection_layer.bias)



    def forward(self, points: torch.Tensor) -> torch.Tensor:
        batch_size = points.shape[0]
        # Get the features at the points
        x = points
        # points = points.view(1 , batch_size, 1, 1, 3).detach() # (B, 1, 1, 3)
        # features = nn.functional.grid_sample(self.embedding, points, align_corners=False, padding_mode="border") 
        features = self.encoding(points.detach())

        # Reshape the features
        features = features.view(batch_size, self.embedding_features)
        # modulation = self.projection_layer(features)
        # points = points.view(batch_size, 3)

        for i in range(self.hidden_layers):
            projection_layer = self.projection_layers[i]
            layer = self[i]
            # assert isinstance(layer, SirenLayer)
            x = layer.forward(x, shift=projection_layer(features))

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

        first_layer = ComplexExpLayer(
            in_features=in_features,
            out_features=hidden_dim,
            init_scheme=first_layer_init,
        )
        self[0] = first_layer

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + 1.0  # TODO: Remove this hack
        return super().forward(x)
