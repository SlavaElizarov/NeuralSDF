from typing import List
from torch import nn
import torch
from layers import CommutatorAttetionLayer, SirenLayer
from layers.siren import ModulateArg, SirenBiasInitScheme, SirenInitScheme, SirenModulationType

from models.sdf import SDF


class TransSiren(nn.Sequential, SDF):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        hidden_layers: int,
        output_dim: int,
        first_omega_0: float = 30,
        hidden_omega_0: float = 30.0,
        init_scheme: SirenInitScheme = SirenInitScheme.SIREN_UNIFORM,
        modulation_type: SirenModulationType = SirenModulationType.FILM,
        bias_init_scheme: SirenBiasInitScheme = SirenBiasInitScheme.HE_UNIFORM,
        self_modulate: List[ModulateArg] = [],
    ):
        super().__init__()

        self.embedding_layer = SirenLayer(
            input_dim,
            hidden_dim,
            is_first=True,
            omega_0=first_omega_0,
            init_scheme=SirenInitScheme.SIREN_UNIFORM,
            modulation_type=modulation_type,
            bias_init_scheme=bias_init_scheme,
            self_modulate=self_modulate
        )

        attention_layers = []
        for i in range(hidden_layers):
            attention_layers.append(
                CommutatorAttetionLayer(
                    n_heads=(i + 1) * 2,  # 2**(i+1),#(i + 1) * 2
                    input_dim=hidden_dim,
                    output_dim=hidden_dim,
                    values_projection_factory=lambda _in, _out, _: SirenLayer(
                        _in,
                        _out,
                        is_first=False,
                        omega_0=hidden_omega_0,
                        init_scheme=init_scheme,
                        modulation_type=modulation_type,
                        bias_init_scheme=bias_init_scheme,
                        disable_activation=True,
                        self_modulate=self_modulate
                    ),
                )
            )
        self.attention_layers = nn.ModuleList(attention_layers)

        self.final_linear = nn.Linear(hidden_dim, output_dim)
        # final_linear.weight.data.uniform_(-np.sqrt(6 / hidden_dim), np.sqrt(6 / hidden_dim))

    def forward(self, x):
        x = self.embedding_layer(x)
        for attention_layer in self.attention_layers:
            x, _ = attention_layer(x)
            x = torch.sin(x)
        x = self.final_linear(x)
        return x
