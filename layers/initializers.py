from abc import ABC
import torch
from torch import nn
import numpy as np


class Initializer(ABC):
    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        return self.initialize(tensor)

    def initialize(self, tensor: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


class SirenInitializer(Initializer, ABC):
    def __init__(self, omega: float = 30.0) -> None:
        assert omega > 0
        self.omega = omega


class SirenUniformInitializer(SirenInitializer):
    def __init__(self, omega: float = 30.0, is_first: bool = False):
        """
        Fill the weights with values from a uniform distribution
        Described in paper: https://arxiv.org/abs/2006.09661

        Args:
            omega (float): omega is a frequency factor which simply multiplies
                                        the features before the nonlinearity.
                                        Different signals may require different omega_0 in the first layer.
            is_first (bool): is this the first layer in the network
        """
        super().__init__(omega)
        self.is_first = is_first

    def initialize(self, tensor: torch.Tensor) -> torch.Tensor:
        assert tensor.ndim == 2
        _, input_dim = tensor.shape
        if self.is_first:
            return nn.init.uniform_(tensor, -1 / input_dim, 1 / input_dim)

        return nn.init.uniform_(
            tensor,
            -np.sqrt(6 / input_dim) / self.omega,
            np.sqrt(6 / input_dim) / self.omega,
        )


class SirenNormalInitializer(SirenInitializer):
    def __init__(self, omega: float = 30.0, is_first: bool = False):
        """
        Fill the weights with values from a normal distribution
        Modification of the method described in paper: https://arxiv.org/abs/2006.09661

        Args:
            omega (float): omega is a frequency factor which simply multiplies
                                        the features before the nonlinearity.
                                        Different signals may require different omega_0 in the first layer.
            is_first (bool): is this the first layer in the network
        """
        super().__init__(omega)
        self.is_first = is_first

    def initialize(self, tensor: torch.Tensor) -> torch.Tensor:
        assert tensor.ndim == 2
        _, input_dim = tensor.shape
        if self.is_first:
            return nn.init.normal_(tensor, 0, 1 / np.sqrt(3) / input_dim)
        return nn.init.normal_(tensor, 0, np.sqrt(2 / input_dim) / self.omega)


class SirenLogNormalInitializer(SirenInitializer):
    def __init__(self, omega: float = 30.0, std: float = 2.2):
        """
        Fill the weights with values from a log normal distribution
        Modification of the method described in paper: https://arxiv.org/abs/2006.09661

        Args:
            omega (float): omega is a frequency factor which simply multiplies
                                        the features before the nonlinearity.
                                        Different signals may require different omega in the first layer.

        """
        super().__init__(omega)
        self.std = std

    def initialize(self, tensor: torch.Tensor) -> torch.Tensor:
        assert tensor.ndim == 2
        with torch.no_grad():
            out, _ = tensor.shape
            tensor = tensor.log_normal_(std=self.std)
            tensor[: out // 2] *= -1
            # tensor /= self.omega
            return tensor
