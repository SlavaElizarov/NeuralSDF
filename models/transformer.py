from torch import nn
import torch
from layers import CommutatorAttetionLayer, SirenLayer
from layers.initializers import SirenInitializer, SirenUniformInitializer

from models.sdf import SDF


class TransSiren(nn.Sequential, SDF):
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
        super().__init__()

        self.embedding_layer = SirenLayer(
            in_features, hidden_dim, True, first_layer_init
        )

        attention_layers = []
        for i in range(hidden_layers):
            attention_layers.append(
                CommutatorAttetionLayer(
                    n_heads=(i + 1) * 2,  # 2**(i+1),#(i + 1) * 2
                    in_features=hidden_dim,
                    out_features=hidden_dim,
                    values_projection_factory=lambda _in, _out, _: SirenLayer(
                        _in,
                        _out,
                        init_scheme=hidden_layer_init,
                        disable_activation=True,
                    ),
                )
            )
        self.attention_layers = nn.ModuleList(attention_layers)

        self.final_layer = SirenLayer(
            hidden_dim, out_features, add_bias=True, disable_activation=True
        )
        nn.init.zeros_(self.final_layer.bias)

    def forward(self, x):
        x = self.embedding_layer(x)
        for attention_layer in self.attention_layers:
            x, _ = attention_layer(x)
            x = torch.sin(x)  # TODO: should it be torch.sin(x * omega)?
        x = self.final_layer(x)
        return x
