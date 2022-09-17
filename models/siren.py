from abc import ABC, abstractmethod, ABCMeta
from typing import List
from turtle import forward
from typing import Optional
import torch
from torch import nn
import numpy as np

class SDF(nn.Module, ABC):
    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pass

class SinActivation(nn.Module):
    def __init__(self, omega_0: float = 1.0):
        """
            Sin activation with scaling.
            Paper: https://arxiv.org/abs/2006.09661
        Args:
            omega_0 (float, optional):  Omega_0 parameter from SIREN paper. Defaults to 1.0.
        """
        super().__init__()
        assert omega_0 > 0
        self.omega_0: float = omega_0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sin(self.omega_0 * x)


class SirenLayer(nn.Module):
    # See paper sec. 3.2, final paragraph, and supplement Sec. 1.5 for discussion of omega_0.

    # If is_first=True, omega_0 is a frequency factor which simply multiplies the activations before the
    # nonlinearity. Different signals may require different omega_0 in the first layer - this is a
    # hyperparameter.

    # If is_first=False, then the weights will be divided by omega_0 so as to keep the magnitude of
    # activations constant, but boost gradients to the weight matrix (see supplement Sec. 1.5)

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        is_first: bool = False,
        omega_0: float = 30,
    ):
        """
            Dense layer with sin activation.
            Described in paper: https://arxiv.org/abs/2006.09661

        Args:
            in_features (int): Number of input features.
            out_features (int): Number of output features.
            bias (bool, optional): Use bias? Defaults to True.
            is_first (bool, optional): Is first? Initialzation depends on this parameter.
                                       See 3.2 of the paper. Defaults to False.
            omega_0 (float, optional): omega_0 is a frequency factor which simply multiplies
                                        the activations before the nonlinearity.
                                        Different signals may require different omega_0 in the first layer.
                                        Defaults to 30.
        """
        super().__init__()
        self.omega_0 = omega_0
        self.is_first = is_first

        self.in_features = in_features
        self.linear = nn.Linear(in_features, out_features, bias=bias)

        self.init_weights()
        self.activation = SinActivation(omega_0)

    def init_weights(self):
        if self.is_first:
            nn.init.uniform_(self.linear.weight, -1 / self.in_features, 1 / self.in_features)
        else:
            nn.init.uniform_(
                self.linear.weight,
                -np.sqrt(6 / self.in_features) / self.omega_0,
                np.sqrt(6 / self.in_features) / self.omega_0,
            )

    def forward(
        self,
        input: torch.Tensor,
        scale: Optional[torch.Tensor] = None,
        shift: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        y = self.linear(input)

        if scale is not None:
            y = y * scale

        if shift is not None:
            y = y + shift

        return self.activation(y)


class ModulatedSirenLayer(SirenLayer):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        latent_size: int,
        bias=True,
        is_first=False,
        omega_0: float = 30,
        shift_only: bool = True,
    ):
        """
            Modulated Siren layer described in paper: https://arxiv.org/abs/2201.12904

        Args:
            in_features (int): Number of input features.
            out_features (int): Number of output features.
            latent_size (int): Size of latent vector.
            bias (bool, optional): Use bias? Defaults to True.
            is_first (bool, optional): Is firts? Initialzation depends on this parameter.
                                        See 3.2 of the paper. Defaults to False.
            omega_0 (float, optional): omega_0 is a frequency factor which simply multiplies
                                        the activations before the nonlinearity.
                                        Different signals may require different omega_0 in the first layer.
                                        Defaults to 30.
            shift_only (bool, optional): Use shifts only for modulation.
                                        It proved to be sufficient in https://arxiv.org/abs/2201.12904.
                                        Defaults to True.
        """
        super().__init__(in_features, out_features, bias, is_first, omega_0)

        self._shift_only = shift_only

        if not self._shift_only:
            self._latent2scale = nn.Linear(latent_size, out_features)

        self._latent2shift = nn.Linear(latent_size, out_features)
        nn.init.zeros_(self._latent2shift.bias)

    def forward(self, x: torch.Tensor, latent: torch.Tensor) -> torch.Tensor:
        shift = self._latent2shift(latent)
        scale = None

        if not self._shift_only:
            scale = self._latent2scale(latent)

        return super().forward(x, scale, shift)


class SirenGeometricHead(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        sdf_output = torch.sign(x) * torch.sqrt(x.abs() + 1e-8)
        sdf_output -= 1.6
        return sdf_output


class Siren(nn.Sequential, SDF):
    def __init__(
        self,
        in_features: int,
        hidden_features: int,
        hidden_layers: int,
        out_features: int,
        outermost_linear=False,
        first_omega_0=30,
        hidden_omega_0=30.0,
        use_geometric_initialization=False,
    ):
        """
            Siren model described in paper: https://arxiv.org/abs/2006.09661

        Args:
            in_features (int): Number of input features.
            hidden_features (int): Number of hidden features.
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
        layers.append(SirenLayer(in_features, hidden_features, is_first=True, omega_0=first_omega_0))

        for _ in range(hidden_layers):
            layers.append(
                SirenLayer(
                    hidden_features,
                    hidden_features,
                    is_first=False,
                    omega_0=hidden_omega_0,
                )
            )

        if outermost_linear:
            final_linear = nn.Linear(hidden_features, out_features)

            nn.init.uniform_(
                final_linear.weight,
                -np.sqrt(6 / hidden_features) / hidden_omega_0,
                np.sqrt(6 / hidden_features) / hidden_omega_0,
            )

            layers.append(final_linear)
        else:
            layers.append(
                SirenLayer(
                    hidden_features,
                    out_features,
                    is_first=False,
                    omega_0=hidden_omega_0,
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
        # TODO: refactor it 
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

class ImplicitAttetionLayer(nn.Module):
    def __init__(self, 
                 n_heads: int = 1, 
                 input_dim: int = 256, 
                 attention_dim:int=64, 
                 hidden_dim: int = 256, 
                 omega_0:float = 30,
                 is_first:bool = False,
                 linear_map_output:bool = False):
        super().__init__()
        self.n_heads = n_heads
        self.input_dim = input_dim
        self.omega_0 = omega_0
        self.keys_transfors = torch.nn.ModuleList([nn.Linear(input_dim, attention_dim) for _ in range(n_heads)])
        
        value_transforms = []
        for _ in range(n_heads):
            l = nn.Linear(input_dim, hidden_dim) 
            
            if not linear_map_output:
                self.init_siren(l, is_first)
                
            value_transforms.append(l)
        
        self.values_transforms = torch.nn.ModuleList(value_transforms)
        
        self.query_transform = torch.nn.Linear(input_dim, attention_dim)
        
        self.linear_map_output = linear_map_output
        
        if linear_map_output:
            self.output_transform = torch.nn.Linear(hidden_dim, input_dim)
            self.init_siren(self.output_transform, is_first)
        
        self.activation = SinActivation(omega_0)

        
    def init_siren(self, linear: nn.Linear, is_first: bool = False):
        with torch.no_grad():
            if is_first:
                nn.init.uniform_(linear.weight, -1 / self.input_dim, 1 / self.input_dim)
            else:
                nn.init.normal_(linear.weight, 0, np.sqrt(2 / self.input_dim) / self.omega_0)
                # nn.init.uniform_(
                #     linear.weight,
                #     -np.sqrt(6 / self.input_dim) / self.omega_0,
                #     np.sqrt(6 / self.input_dim) / self.omega_0,
                # )
        

    def forward(self, x):
        query = self.query_transform(x)
        attention = []
        for i in range(self.n_heads):
            head_weight = torch.sum(self.keys_transfors[i](x) * query, dim=-1)

            attention.append(head_weight)
        attention = torch.softmax(torch.stack(attention, dim=-1), dim=-1)
        
        result = torch.cat([self.values_transforms[i](x)[:, None, :]  for i in range(self.n_heads)], dim=1)
        result = torch.sum(result * attention[:, :, None], dim=1)
        
        if self.linear_map_output:
            result = self.output_transform(result)
        
        return self.activation(result)
    
class ImplicitBandAttetionLayer(nn.Module):
    def __init__(self, 
                 input_dim: int = 256, 
                 attention_dim:int=64, 
                 hidden_dim: int = 256, 
                 omega_0: List[float] = [5, 15, 30, 60, 90],
                 is_first:bool = False,):
        super().__init__()
        self.n_heads = len(omega_0)
        self.input_dim = input_dim
        self.omega_0 = omega_0
        self.keys_transfors = torch.nn.ModuleList([nn.Linear(input_dim, attention_dim) for _ in range(self.n_heads)])
        
        value_transforms = []
        for omega in omega_0:
            l = SirenLayer(input_dim, hidden_dim, is_first=is_first, omega_0=omega) 
            value_transforms.append(l)
        
        self.values_transforms = torch.nn.ModuleList(value_transforms)
        
        self.query_transform = torch.nn.Linear(input_dim, attention_dim)
 

    def forward(self, x):
        query = self.query_transform(x)
        attention = []
        for i in range(self.n_heads):
            head_weight = torch.sum(self.keys_transfors[i](x) * query, dim=-1)

            attention.append(head_weight)
        attention = torch.softmax(torch.stack(attention, dim=-1), dim=-1)
        
        print(attention[:100].argmax(dim=-1))
        
        result = torch.cat([self.values_transforms[i](x)[:, None, :]  for i in range(self.n_heads)], dim=1)
        result = torch.sum(result * attention[:, :, None], dim=1)
        
        return result
            
       
class TransSiren(nn.Sequential, SDF):
    def __init__(
        self,
        in_features: int,
        hidden_features: int,
        hidden_layers: int,
        out_features: int,
        outermost_linear=False,
        first_omega_0=30,
        hidden_omega_0=30.0,
        linear_map_output: bool = False,
    ):
        super().__init__()

        layers = []
        layers.append(SirenLayer(in_features, hidden_features, is_first=True, omega_0=first_omega_0))

        for i in range(hidden_layers):
            layers.append(
                ImplicitAttetionLayer(n_heads=(i+1)*2, input_dim=hidden_features, hidden_dim=hidden_features, omega_0=hidden_omega_0, linear_map_output=linear_map_output)
            )

        if outermost_linear:
            final_linear = nn.Linear(hidden_features, out_features)

            # nn.init.uniform_(
            #     final_linear.weight,
            #     -np.sqrt(6 / hidden_features) / hidden_omega_0,
            #     np.sqrt(6 / hidden_features) / hidden_omega_0,
            # )

            layers.append(final_linear)
        else:
            layers.append(
                SirenLayer(
                    hidden_features,
                    out_features,
                    is_first=False,
                    omega_0=hidden_omega_0,
                )
            )


        super().__init__(*layers)
        
        
class BandTransSiren(nn.Sequential, SDF):
    def __init__(
        self,
        in_features: int,
        hidden_features: int,
        hidden_layers: int,
        out_features: int,
        outermost_linear=True,
    ):
        super().__init__()

        layers = []
        # layers.append(SirenLayer(in_features, hidden_features, is_first=True, omega_0=50))
        layers.append(ImplicitBandAttetionLayer(input_dim=in_features, hidden_dim=hidden_features, omega_0=[5, 15, 30, 60, 90], is_first=True))

        for i in range(hidden_layers):
            layers.append(
                ImplicitBandAttetionLayer(input_dim=hidden_features, hidden_dim=hidden_features, is_first=False)
            )

        if outermost_linear:
            final_linear = nn.Linear(hidden_features, out_features)

            nn.init.uniform_(
                final_linear.weight,
                -np.sqrt(6 / hidden_features) / 50,
                np.sqrt(6 / hidden_features) / 50,
            )

            layers.append(final_linear)
       

        super().__init__(*layers)
        
        
# class GaussianActivation(nn.Module):
#     def __init__(self, a=1., trainable=True):
#         super().__init__()
#         self.register_parameter('a', nn.parameter.Parameter(a*torch.ones(1), trainable))

#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         return torch.exp(-x**2/(2*self.a**2))


