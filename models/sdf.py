from abc import ABC, abstractmethod
import torch
from torch import nn


class SDF(nn.Module, ABC):
    def __init__(self, analytical_gradient_available: bool = True) -> None:
        super().__init__()
        self.analytical_gradient_available = analytical_gradient_available

    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pass

    def get_gradient(self, x: torch.Tensor, force_numerical: bool = False, delta: float = 1e-3) -> torch.Tensor:
        if self.analytical_gradient_available and not force_numerical:
            return self._analytical_gradient(x)
        
        return self._numerical_gradient(x, delta)

    def _analytical_gradient(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute the autograd gradient of the SDF at the given points
        Args:
            x (torch.Tensor): points at which the gradient is computed
        Returns:
            torch.Tensor: gradient of the SDF at the given points
        """
        
        input_requires_grad = x.requires_grad
        x.requires_grad_(True)
        sdf = self.forward(x)
        (gradient,) = torch.autograd.grad(
            outputs=sdf.sum(),
            inputs=x,
            retain_graph=True,
            create_graph=self.training,
        )
        if not input_requires_grad:
            x.requires_grad_(False)
        return gradient
    
    def _numerical_gradient(self, x: torch.Tensor, delta: float = 1e-3) -> torch.Tensor:
        """
        Compute the numerical gradient of the SDF at the given points
        Args:
            x (torch.Tensor): points at which the gradient is computed
            delta (float, optional): delta for the finite difference. Defaults to 1e-3.
        Returns:
            torch.Tensor: gradient of the SDF at the given points
        """
        assert x.ndim == 2
        assert x.shape[1] == 3
        
        offsets = torch.as_tensor(
                            [
                                [delta, 0.0, 0.0],
                                [-delta, 0.0, 0.0],
                                [0.0, delta, 0.0],
                                [0.0, -delta, 0.0],
                                [0.0, 0.0, delta],
                                [0.0, 0.0, -delta],
                            ],
                            dtype=x.dtype,
                        ).to(x)

        x = x.unsqueeze(-2) + offsets.unsqueeze(0)

        distances = self.forward(x.view(-1, 3))
        distances = distances.view(-1, 6)
        gradients = torch.stack(
            [
                0.5 * (distances[:, 0] - distances[:, 1]) / delta,
                0.5 * (distances[:, 2] - distances[:, 3]) / delta,
                0.5 * (distances[:, 4] - distances[:, 5]) / delta,
            ],
            dim=-1,
        )

        return gradients
