from enum import Enum
from typing import Optional, Set
import torch
from torch import nn
import numpy as np
from layers import CrossAttentionLayer, SubtractionCrossAttentionLayer
from layers import ModulateArg, SirenBiasInitScheme, SirenLayer, SirenModulationType

from models.sdf import SDF

class AttentionType(Enum):
    DOT = 1
    SUBTRACTION = 2


class SirenGeometricHead(nn.Module):
    def __init__(self) -> None:
        """
        Geometric head for Siren model.
        Used to apply a geometric intialization from https://arxiv.org/abs/2106.10811
        
        Empirically it does not work well :(
        """
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        sdf_output = torch.sign(x) * torch.sqrt(x.abs() + 1e-8)
        sdf_output -= 1.6
        return sdf_output

class Siren(nn.Sequential, SDF):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        hidden_layers: int,
        out_features: int,
        outermost_linear: bool =False,
        first_omega_0:float = 30.0,
        hidden_omega_0:float = 30.0,
        use_geometric_initialization = False,
        modulation_type: SirenModulationType = SirenModulationType.FILM,
        bias_init_scheme: SirenBiasInitScheme = SirenBiasInitScheme.ZEROS,
        self_modulate: Set[ModulateArg] = set(),        
    ):
        """
            Siren model described in paper: https://arxiv.org/abs/2006.09661

        Args:
            input_dim (int): Number of input features.
            hidden_dim (int): Number of hidden features.
            hidden_layers (int): Number of hidden layers.
            out_features (int): Number of output features.
            outermost_linear (bool, optional): Is final layer linear?. Defaults to False.
            first_omega_0 (float, optional): Omega for first layer. Defaults to 30.
            hidden_omega_0 (float, optional): omega_0 is a frequency factor which simply multiplies
                                            the activations before the nonlinearity. Defaults to 30.
            use_geometric_initialization (bool, optional): Use geometric initialization.
                    The main idea is to start from SDF of a sphere.
                    For details check https://arxiv.org/abs/2106.10811
                    Defaults to False.
        """
        super().__init__()

        layers = []

        for i in range(hidden_layers):
            is_first = i == 0
            layers.append(
                SirenLayer(
                    input_dim if is_first else hidden_dim,
                    hidden_dim,
                    is_first = is_first,
                    omega_0 = first_omega_0 if is_first else hidden_omega_0,
                    modulation_type=modulation_type,
                    bias_init_scheme=bias_init_scheme,
                    self_modulate = self_modulate
                )
            )

        layers.append(
            SirenLayer(
                hidden_dim,
                out_features,
                is_first=False,
                omega_0=hidden_omega_0,
                disable_activation = outermost_linear,
            )
        )

        if use_geometric_initialization:
            layers.append(SirenGeometricHead())

        super().__init__(*layers)

        if use_geometric_initialization:
            self.geometric_init()

    def geometric_init(self):
        assert len(self) >= 5, "Geometric initialization is only applicable for a network with at least 5 layers"
        # shamelessly copied from https://github.com/Chumbyte/DiGS/blob/main/models/DiGS.py
        # TODO: refactor it, God bless a soul who will do it
        # TODO: Consider deleting this method, it does not work well anyway 
        def geom_sine_init(m):
            with torch.no_grad():
                if hasattr(m, 'weight'):
                    num_output = m.weight.size(0)
                    m.weight.uniform_(-np.sqrt(3 / num_output), np.sqrt(3 / num_output))
                    m.bias.uniform_(-1 / (num_output * 1000), 1 / (num_output * 1000))
                    m.weight.data /= 30
                    m.bias.data /= 30

        def second_last_layer_geom_sine_init(m):
            with torch.no_grad():
                if hasattr(m, 'weight'):
                    num_output = m.weight.size(0)
                    assert m.weight.shape == (num_output, num_output)
                    m.weight.data = 0.5 * np.pi * torch.eye(num_output) + 0.001 * torch.randn(num_output, num_output)
                    m.bias.data = 0.5 * np.pi * torch.ones(
                        num_output,
                    ) + 0.001 * torch.randn(num_output)
                    m.weight.data /= 30
                    m.bias.data /= 30

        def last_layer_geom_sine_init(m):
            with torch.no_grad():
                if hasattr(m, 'weight'):
                    num_input = m.weight.size(-1)
                    assert m.weight.shape == (1, num_input)
                    assert m.bias.shape == (1,)
                    # m.weight.data = -1 * torch.ones(1, num_input) + 0.001 * torch.randn(num_input)
                    m.weight.data = -1 * torch.ones(1, num_input) + 0.00001 * torch.randn(num_input)
                    m.bias.data = torch.zeros(1) + num_input

        # ################################# multi frequency geometric initialization ###################################
        # periods = [1, 30] # Number of periods of sine the values of each section of the output vector should hit
        # # periods = [1, 60] # Number of periods of sine the values of each section of the output vector should hit
        # portion_per_period = np.array([0.25, 0.75]) # Portion of values per section/period

        def first_layer_mfgi_init(m):
            periods = [1, 30]
            portion_per_period = np.array([0.25, 0.75])
            with torch.no_grad():
                if hasattr(m, 'weight'):
                    num_input = m.weight.size(-1)
                    num_output = m.weight.size(0)
                    num_per_period = (portion_per_period * num_output).astype(
                        int
                    )  # Number of values per section/period
                    assert len(periods) == len(num_per_period)
                    assert sum(num_per_period) == num_output
                    weights = []
                    for i in range(0, len(periods)):
                        period = periods[i]
                        num = num_per_period[i]
                        scale = 30 / period
                        weights.append(
                            torch.zeros(num, num_input).uniform_(
                                -np.sqrt(3 / num_input) / scale, np.sqrt(3 / num_input) / scale
                            )
                        )
                    W0_new = torch.cat(weights, axis=0)
                    m.weight.data = W0_new

        def second_layer_mfgi_init(m):
            portion_per_period = np.array([0.25, 0.75])
            with torch.no_grad():
                if hasattr(m, 'weight'):
                    num_input = m.weight.size(-1)
                    assert m.weight.shape == (num_input, num_input)
                    num_per_period = (portion_per_period * num_input).astype(int)  # Number of values per section/period
                    k = num_per_period[0]  # the portion that only hits the first period
                    # W1_new = torch.zeros(num_input, num_input).uniform_(-np.sqrt(3 / num_input), np.sqrt(3 / num_input) / 30) * 0.00001
                    W1_new = (
                        torch.zeros(num_input, num_input).uniform_(-np.sqrt(3 / num_input), np.sqrt(3 / num_input) / 30)
                        * 0.0005
                    )
                    W1_new_1 = torch.zeros(k, k).uniform_(-np.sqrt(3 / num_input) / 30, np.sqrt(3 / num_input) / 30)
                    W1_new[:k, :k] = W1_new_1
                    m.weight.data = W1_new


        self.apply(geom_sine_init)
        self[0].linear.apply(first_layer_mfgi_init)
        self[1].linear.apply(second_layer_mfgi_init)
        self[-3].linear.apply(second_last_layer_geom_sine_init)
        self[-2].apply(last_layer_geom_sine_init)
        
    
class TransposedAttentionSiren(Siren):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        hidden_layers: int,
        output_dim: int,
        first_omega_0:float=30.,
        hidden_omega_0:float=30.0,
        latent_seq_len:int = 64,
        use_dropout:bool = True,
        attention_dim: int = 64, 
        attention_type: AttentionType = AttentionType.DOT,
        modulation_type: SirenModulationType = SirenModulationType.FILM,
        bias_init_scheme: SirenBiasInitScheme = SirenBiasInitScheme.ZEROS,
        self_modulate: Set[ModulateArg] = set(),        
        attention_modulate: Set[ModulateArg] = {ModulateArg.Amplitude, ModulateArg.Phase, ModulateArg.Frequency},
    ): 
        super().__init__(input_dim, 
                         hidden_dim, 
                         hidden_layers, 
                         output_dim, 
                         True,
                         first_omega_0, 
                         hidden_omega_0,
                         False,
                         modulation_type,
                         bias_init_scheme,
                         self_modulate
                         )
        
        assert len(attention_modulate) > 0, "Must modulate at least one parameter"
        assert ModulateArg.Amplitude in attention_modulate, "this assert is temporary"
        
        self.hidden_layers = hidden_layers
        self.modulate = attention_modulate
        
        number_of_heads = (hidden_layers -1) * len(attention_modulate)
        if ModulateArg.Amplitude in attention_modulate:
            number_of_heads += 1
        
        attention_layer = CrossAttentionLayer if attention_type == AttentionType.DOT else SubtractionCrossAttentionLayer
        
        self.attention = nn.ModuleDict()
        for modulate in attention_modulate:
            number_of_heads = hidden_layers if modulate == ModulateArg.Amplitude else hidden_layers -1
            self.attention[modulate.name] = attention_layer(first_input_dim=hidden_dim, 
                                            second_input_dim=hidden_dim,
                                            attention_dim=attention_dim,
                                            number_of_heads=number_of_heads,
                                            value_dim=hidden_dim,
                                            use_dropout=use_dropout)
        
        self.latent = nn.Parameter(torch.randn((1, latent_seq_len, hidden_dim), dtype=torch.float32))
        
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self[0](x)
                
        modulation_dict = {key: layer(x) for key, layer in self.attention.items()}
        
        if ModulateArg.Amplitude in self.modulate:
            x = x * modulation_dict[ModulateArg.Amplitude.name][:, 0, 0, :]

        for i, layer in enumerate(self[1:]):
            frequency_mod = None
            if ModulateArg.Frequency in modulation_dict:
                frequency_mod = modulation_dict[ModulateArg.Frequency][:, 0, i, :]
                
            phase_mod = None
            if ModulateArg.Phase in modulation_dict:
                phase_mod = modulation_dict[ModulateArg.Phase][:, 0, i, :]
                
            x = layer(x, scale=frequency_mod, shift=phase_mod)
            
            if ModulateArg.Amplitude in modulation_dict:
                x = x * modulation_dict[ModulateArg.Amplitude][:, 0, i + 1, :]

        return self[-1](x)
    
        