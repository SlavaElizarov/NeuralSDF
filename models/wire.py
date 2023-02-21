import math
from torch import nn

from layers.initializers import (
    SirenInitializer,
    SirenUniformInitializer,
)
from layers.wire import WireLayer
from models.sdf import SDF


class Wire(nn.Sequential, SDF):
    def __init__(
        self,
        in_features: int,
        hidden_dim: int,
        hidden_layers: int,
        out_features: int,
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
            is_last = i == hidden_layers - 1
            layers.append(
                WireLayer(
                    in_features if is_first else hidden_dim,
                    hidden_dim,
                    gaussian_windows=3 if is_first else 1,
                    s_0=1,
                    return_real=is_last,
                    init_scheme=first_layer_init if is_first else hidden_layer_init,
                )
            )
        final_layer = nn.Linear(hidden_dim, out_features, bias=True)
        nn.init.zeros_(final_layer.bias)
        layers.append(final_layer)

        super().__init__(*layers)
