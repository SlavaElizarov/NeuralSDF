from abc import ABC, abstractmethod
from typing import Callable, Optional
import torch
from torch.nn import functional as F

from models.sdf import SDF


class LossBase(ABC):
    def __init__(self, weight: float = 1.0, name: Optional[str] = None):
        self.weight = weight
        if name is None:
            self.name = self.__class__.__name__
        else:
            self.name = name

        self._logger = lambda name, loss: None

    def _log(self, name: str, loss: torch.Tensor):
        self._logger(name, loss)

    @abstractmethod
    def _loss(self, *args) -> torch.Tensor:
        pass

    def __call__(self, *args) -> torch.Tensor:
        loss_value = self.weight * self._loss(*args)
        self._log(self.name, loss_value)
        return loss_value

    def set_logger(self, logger: Callable[[str, torch.Tensor], None]):
        self._logger = logger


class SurfaceLoss(LossBase):
    def __init__(self, weight: float = 1.0):
        super().__init__(weight=weight, name=f"zero_set")

    def _loss(self, distances: torch.Tensor) -> torch.Tensor:
        return torch.abs(distances).mean()

    def __call__(self, distances: torch.Tensor) -> torch.Tensor:
        return super().__call__(distances)


class OffSurfaceLoss(LossBase, ABC):
    @abstractmethod
    def _loss(
        self, distances: torch.Tensor, gradients: Optional[torch.Tensor]
    ) -> torch.Tensor:
        pass

    def __call__(
        self, distances: torch.Tensor, gradients: torch.Tensor
    ) -> torch.Tensor:
        return super().__call__(distances, gradients)

class OffSurfaceGTLoss(LossBase):
    def __init__(self, weight: float = 1.0):
        super().__init__(weight=weight, name=f"offsurface_gt")

    def _loss(self, distances: torch.Tensor, distances_gt: torch.Tensor) -> torch.Tensor:
        return  torch.nn.functional.l1_loss(distances, distances_gt[:, None])

    def __call__(self, distances: torch.Tensor, distances_gt: torch.Tensor) -> torch.Tensor:
        return super().__call__(distances, distances_gt)

class LaplacianLoss(LossBase, ABC):
    def __init__(self, weight: float = 1.0):
        super().__init__(weight=weight, name=f"laplacian_loss")

    def _loss(self, laplacian: torch.Tensor, gradients: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return torch.abs(laplacian).mean()

    def __call__(
        self, laplacian: torch.Tensor, gradients: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return super().__call__(laplacian, gradients, y)


class GradientDirectionLoss(LossBase):
    def __init__(self, weight: float = 1.0, type: str = "cos"):
        super().__init__(weight=weight, name=f"∇_direction_{type}")

        if type == "cos":
            self._loss = self._loss_cos
        elif type == "l1":
            self._loss = self._loss_l1
        elif type == "monosdf":
            self._loss = self._loss_monosdf
        else:
            raise NotImplementedError(f"Loss {type} is not implemented")

    def _loss(self, gradient: torch.Tensor, normals: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError("This method should be implemented in __init__")
    
    def _loss_monosdf(self, gradient: torch.Tensor, normals: torch.Tensor) -> torch.Tensor:
        return self._loss_cos(gradient, normals) + self._loss_l1(gradient, normals)

    def _loss_cos(self, gradient: torch.Tensor, normals: torch.Tensor) -> torch.Tensor:
        return (1 - F.cosine_similarity(gradient, normals, dim=-1)).mean()

    def _loss_l1(self, gradient: torch.Tensor, normals: torch.Tensor) -> torch.Tensor:
        return F.l1_loss(gradient, normals, reduction="mean")

    def __call__(self, gradient: torch.Tensor, normals: torch.Tensor) -> torch.Tensor:
        return super().__call__(gradient, normals)


class EikonalLoss(LossBase):
    def __init__(self, weight: float = 1.0, type: str = "l1"):
        super().__init__(weight=weight, name=f"eikonal")
        self.type = type

    def _loss(self, gradient: torch.Tensor) -> torch.Tensor:
        """
        Eikonal loss is a loss that enforces the gradient to be a unit vector

        SDF is a solution of the eikonal equation,
        E_x|∇f(x)| = 1, x ∼ P(D). Where f(x) is the SDF and P(D) is some distribution in R^3

        Args:
            gradient (torch.Tensor): gradient of the SDF

        Returns:
            torch.Tensor: l1 loss between the gradient norm and 1
        """
        grad_norm = torch.linalg.norm(gradient, ord=2, dim=-1)
        if self.type == "l2":
            return torch.square(grad_norm - 1).mean()
        elif self.type == "l1":
            return torch.abs(grad_norm - 1).mean()
        else:
            raise NotImplementedError(f"Loss {self.type} is not implemented")


class ViscosityLoss(LaplacianLoss):
    def __init__(self, weight: float = 1.0, epsilon: float = 0.01):
        """
        From VisCo Grids: Surface Reconstruction with Viscosity and Coarea Grids


        Args:
            weight (float, optional): _description_. Defaults to 1.0.
            epsilon (float, optional): _description_. Defaults to 0.01.
        """
        super().__init__(weight=weight, name=f"Viscosity")
        self.epsilon = epsilon

    def _loss(
        self, laplacian: torch.Tensor, gradients: torch.Tensor, y: torch.Tensor
    ) -> torch.Tensor:
        grad_norm = torch.linalg.norm(gradients, ord=2, dim=-1)
        return torch.square(
            (grad_norm - 1) * torch.sign(y) - self.epsilon * laplacian
        ).mean()


class MarginLoss(OffSurfaceLoss):
    def __init__(self, weight: float = 1.0, margin: float = 0.1):
        super().__init__(weight=weight, name=f"offsurface_margin")
        self.margin = margin

    def _loss(self, distances: torch.Tensor, gradients: torch.Tensor) -> torch.Tensor:
        return (F.relu(self.margin - torch.abs(distances)) * 100).mean()


class CoareaLoss(OffSurfaceLoss):
    def __init__(self, weight: float = 1.0, beta: float = 0.1):
        super().__init__(weight=weight, name=f"offsurface_coarea")
        self.beta = beta

    def _loss(self, distances: torch.Tensor, gradients: torch.Tensor) -> torch.Tensor:
        laplace_pdf = 1 / (2 * self.beta) * torch.exp(-torch.abs(distances) / self.beta)
        grad_norm = torch.linalg.norm(gradients, ord=2, dim=-1)
        return (laplace_pdf * grad_norm).mean()


# TODO: investigate this loss, possibly it is not correct
class VolumeLoss(OffSurfaceLoss):
    def __init__(self, weight: float = 1.0, beta: float = 0.01):
        super().__init__(weight=weight, name=f"offsurface_volume")
        self.beta = beta

    def _loss(self, distances: torch.Tensor, gradients: torch.Tensor) -> torch.Tensor:
        volume = torch.exp(-F.relu(distances) / self.beta)
        return volume.mean()


class DivergenceLoss(LaplacianLoss):
    def __init__(self, weight: float = 1.0):
        """
        Divergence loss
        Is a second-order loss that pushes SDF gradient to be divergence-free

        Described in the following paper:
        "DiGS: Divergence guided shape implicit neural representation for unoriented point clouds"
        https://arxiv.org/pdf/2106.10811

        The authors minimize the following function:
        div(∇f(x)) = ∇ · ∇f(x)
        or in other words it is a divergence of the gradient of the SDF.
        which is equivalent to Laplacian of the SDF:
        Δ f(x) = ∇ · ∇f(x) = ∇ f(x)

        Therefore, to use this loss one need to pass the Laplacian of the SDF

        Args:
            weight (float, optional): weight of loss. Defaults to 1.0.
        """
        super().__init__(weight=weight, name=f"divergence")

    def _loss(
        self,
        laplacian: torch.Tensor,
        gradients: torch.Tensor,
        y: torch.Tensor,
    ) -> torch.Tensor:
        return torch.abs(laplacian).mean()
    
class PullLoss(LossBase):
    def __init__(self, weight: float = 1.0, sigma: float = 10.):
        """
        Based on the following papers:
        https://arxiv.org/abs/2011.13495
        https://arxiv.org/abs/2305.11601

        Args:
            weight (float, optional): _description_. Defaults to 1.0.
        """
        super().__init__(weight=weight, name=f"pull")
        self.sigma = sigma
    
    def _loss(self, points: torch.Tensor, distances: torch.Tensor, gradient: torch.Tensor, sdf_model: SDF) -> torch.Tensor:
        gradient_norm = F.normalize(gradient, dim=-1)
        points_moved = points - gradient_norm * distances

        _, gradient_movied = sdf_model.forward_with_grad(points_moved)
        # gradient_movied_norm = F.normalize(gradient_movied, dim=-1)
        consis_constraint = 1 - F.cosine_similarity(gradient_movied, gradient, dim=-1)
        weight_moved = torch.exp(-self.sigma * torch.abs(distances)).reshape(-1,consis_constraint.shape[-1])

        return (consis_constraint * weight_moved).mean() * self.weight





