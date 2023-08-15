import torch
from torch import nn

from layers import SirenLayer
from layers.initializers import SirenInitializer, SirenUniformInitializer
from models.sdf import SDF

class LatentTransformer(SDF):
    def __init__(
        self,
        in_features: int,
        hidden_dim: int,
        hidden_layers: int,
        out_features: int,
        outermost_linear: bool = True,
        first_layer_init: SirenInitializer = SirenUniformInitializer(
            omega=30.0, is_first=True
        ),
        hidden_layer_init: SirenInitializer = SirenUniformInitializer(
            omega=30.0, is_first=False
        ),
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        first_layers = [SirenLayer(in_features, hidden_dim, init_scheme=SirenUniformInitializer(omega=i, is_first=True)) for i in [16, 32, 64, 128]]
        self.first_layers = nn.ModuleList(first_layers)
        self.transformer_block = TransformerBlock(
            hidden_dim,
            num_heads = 4,
            out_features = hidden_dim,
            init_scheme= hidden_layer_init,
        )
        layers = []

        
        for i in range(hidden_layers -1):
            # is_first = i == 0
            layers.append(
                SirenLayer(
                    hidden_dim,
                    hidden_dim,
                    init_scheme=hidden_layer_init,
                )
            )
        final_layer = SirenLayer(
            hidden_dim, out_features, add_bias=True, disable_activation=outermost_linear
        )
        nn.init.zeros_(final_layer.bias)
        layers.append(final_layer)

        self.layers = nn.Sequential(*layers)
        self.embeddings = nn.Embedding(4, hidden_dim)
        # hidden_layer_init(self.embeddings.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.shape[0]
        # print(x.shape)
        y = [layer(x) for layer in self.first_layers]
        x = torch.stack(y, dim=0)
        # print(x.shape)
        # latent = self.embeddings.weight.view(-1 ,1, self.hidden_dim).expand(-1, batch_size, -1)
        # print(latent.shape)
        x = self.transformer_block(x, x)
        x = torch.mean(x, dim=0)
        x = self.layers(x)
        return x


class Transformer(SDF):
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
        super().__init__()
        self.first_layer = SirenLayer(in_features, hidden_dim, init_scheme=first_layer_init)

        layers = []

        
        for _ in range(hidden_layers -1):
            layers.append(
                TransformerBlock(
                    hidden_dim,
                    num_heads = 1,
                    out_features = hidden_dim,
                    init_scheme= hidden_layer_init,
                )
            )
        final_layer = SirenLayer(
            hidden_dim, out_features, add_bias=True, disable_activation=outermost_linear
        )
        nn.init.zeros_(final_layer.bias)
        self.final_layer = final_layer

        self.layers = nn.ModuleList(layers)
        self.embeddings = nn.Embedding(8, hidden_dim)
        # hidden_layer_init(self.embeddings.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.first_layer(x)
        for layer in self.layers:
            x = layer(x, self.embeddings.weight)
        x = self.final_layer(x)
        return x



class TransformerBlock(nn.Module):
    def __init__(self, 
                 hidden_dim: int, 
                 num_heads: int,
                 out_features: int,
                 init_scheme: SirenInitializer = SirenUniformInitializer(
            omega=30.0, is_first=False
        )):
        super(TransformerBlock, self).__init__()
        self.attention = torch.nn.MultiheadAttention(hidden_dim, num_heads, 
                                                     add_bias_kv=True, 
                                                     add_zero_attn=True)
        # self.siren1 = SirenLayer(in_features=hidden_dim, out_features=out_features, init_scheme=SirenUniformInitializer(
            # omega=60.0, is_first=True))
        self.siren = SirenLayer(in_features=hidden_dim, out_features=out_features, init_scheme=init_scheme)

    def forward(self, x: torch.Tensor, latent: torch.Tensor) -> torch.Tensor:
        res = x
        # latent = self.siren1(latent)
        x, _ = self.attention(x, latent, latent)
        # x = x + res

        # res = x
        x = self.siren(x)
        # x = x + res
        return x