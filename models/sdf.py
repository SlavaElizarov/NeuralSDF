from abc import ABC, abstractmethod
from typing import Optional, Tuple, Union
import tinycudann as tcnn
import torch
from torch import nn
from enum import Enum


class GradComputationType(Enum):
    ANALYTICAL = 1
    NUMERICAL = 2


class GradientParameters:
    def __init__(self,
                 computation_type: GradComputationType = GradComputationType.ANALYTICAL,
                 delta: float = 1e-3):
        self.computation_type = computation_type
        self.delta = delta


class SDF(nn.Module, ABC):
    grad_parameters: GradientParameters

    def __init__(self, grad_parameters) -> None:
        super().__init__()
        if grad_parameters is None:
            grad_parameters = GradientParameters()
            
        self.grad_parameters = grad_parameters

    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pass

    def forward_with_grad(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """ Forward pass with gradient w.r.t. x computation
        Args:
            x (torch.Tensor): Input tensor

        Raises:
            ValueError: Unknown grad computation type

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Tuple of values and gradients
        """
        if self.grad_parameters.computation_type == GradComputationType.ANALYTICAL:
            return self._forward_analytical(x, False)
        if self.grad_parameters.computation_type == GradComputationType.NUMERICAL:
            return self._forward_numerical(x, False)

        raise ValueError("Unknown grad computation type")

    def forward_with_grad_and_laplacian(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """ Forward pass with gradient w.r.t. x and laplacian computation
        Args:
            x (torch.Tensor): Input tensor

        Raises:
            ValueError: Unknown grad computation type

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: Tuple of values, gradients and laplacians
        """

        if self.grad_parameters.computation_type == GradComputationType.ANALYTICAL:
            return self._forward_analytical(x, True)
        if self.grad_parameters.computation_type == GradComputationType.NUMERICAL:
            return self._forward_numerical(x, True)

        raise ValueError("Unknown grad computation type")

    def _forward_analytical(self, x: torch.Tensor, return_laplacian: bool) -> Union[Tuple[torch.Tensor, torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor]]:
        with torch.enable_grad():
            input_requires_grad = x.requires_grad
            x.requires_grad_(True)
            values = self.forward(x)
            (gradient,) = torch.autograd.grad(
                outputs=values.sum(),
                inputs=x,
                retain_graph=True,
                create_graph=self.training,
            )

            if return_laplacian:
                laplacian = self._divergence(gradient, x)

            if not input_requires_grad:
                x.requires_grad_(False)

            if return_laplacian:
                return values, gradient, laplacian

            return values, gradient

    def _forward_numerical(self, x: torch.Tensor, return_laplacian: bool) -> Union[Tuple[torch.Tensor, torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor]]:
        assert x.ndim == 2
        assert x.shape[1] == 3
        delta = self.grad_parameters.delta

        offsets = torch.as_tensor(
            [
                [delta, 0.0, 0.0],
                [-delta, 0.0, 0.0],
                [0.0, delta, 0.0],
                [0.0, -delta, 0.0],
                [0.0, 0.0, delta],
                [0.0, 0.0, -delta],
                [0.0, 0.0, 0.0],
            ],
            dtype=x.dtype,
        ).to(x)

        x = x.unsqueeze(-2) + offsets.unsqueeze(0)

        distances = self.forward(x.view(-1, 3))
        distances = distances.view(-1, 7)
        gradients = torch.stack(
            [
                0.5 * (distances[:, 0] - distances[:, 1]) / delta,
                0.5 * (distances[:, 2] - distances[:, 3]) / delta,
                0.5 * (distances[:, 4] - distances[:, 5]) / delta,
            ],
            dim=-1,
        )
        values = distances[:, 6, None]

        if return_laplacian:
            # calculate laplacian with finite differences
            laplacian = (
                distances[:, 0]
                + distances[:, 1]
                + distances[:, 2]
                + distances[:, 3]
                + distances[:, 4]
                + distances[:, 5]
                - 6 * distances[:, 6]
            ) / delta ** 2

            return values, gradients, laplacian

        return values, gradients

    def _divergence(self, y: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        (dx,) = torch.autograd.grad(y[:, 0].sum(),
                                    x, create_graph=True, retain_graph=True)
        (dy,) = torch.autograd.grad(y[:, 1].sum(),
                                    x, create_graph=True, retain_graph=True)
        (dz,) = torch.autograd.grad(y[:, 2].sum(),
                                    x, create_graph=True, retain_graph=True)

        div = dx[:, 0] + dy[:, 1] + dz[:, 2]
        return div
