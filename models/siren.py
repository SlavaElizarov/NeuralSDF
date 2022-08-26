from typing import Optional
import torch
from torch import nn
import numpy as np  

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


class Siren(nn.Sequential):
    def __init__(
        self,
        in_features: int,
        hidden_features: int,
        hidden_layers: int,
        out_features: int,
        outermost_linear=False,
        first_omega_0=30,
        hidden_omega_0=30.0,
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

        super().__init__(*layers)
