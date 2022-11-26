from enum import Enum
from typing import Callable
import pytorch_lightning as pl
import torch
from torch.nn import functional as F
from torch import autograd
from torch.utils.data import DataLoader
from models.sdf import SDF

from training.dataset import MeshDataset


class HighOrderLoss(Enum):
    Div = 1
    Hessian = 2


class ApplyHOLossTo(Enum):
    Surface = 1
    OffSurface = 2
    Both = 3


class SdfExperiment(pl.LightningModule):
    def __init__(
        self,
        sdf_model: SDF,
        mesh_path: str,
        batch_size: int = 1024,
        level_set_loss_weight: float = 10,
        eikonal_loss_weight: float = 1,
        grad_direction_loss_weight: float = 1,
        offsurface_loss_weight: float = 0,
        offsurface_loss_margin: float = 0.0025,
        high_order_loss_type: HighOrderLoss = HighOrderLoss.Div,
        apply_high_order_loss_to: ApplyHOLossTo = ApplyHOLossTo.Both,
        high_order_loss_weight: float = 0.001,
        random_points_per_vertex: int = 3,
    ):
        super().__init__()

        self.sdf_model = sdf_model
        self.level_set_loss_weight = level_set_loss_weight
        self.eikonal_loss_weight = eikonal_loss_weight
        self.grad_direction_loss_weight = grad_direction_loss_weight
        self.offsurface_loss_weight = offsurface_loss_weight
        self.high_order_loss_weight = high_order_loss_weight
        self.high_order_loss_type = high_order_loss_type
        self.apply_high_order_loss_to = apply_high_order_loss_to
        self.offsurface_loss_margin = offsurface_loss_margin
        self.mesh_path = mesh_path  # it's a bit of an antipatern to have this here TODO: decouple data from experiment
        self.batch_size = batch_size
        self.random_points_per_vertex = random_points_per_vertex

        self.random_sampler = torch.distributions.uniform.Uniform(
            torch.tensor([-1.1]), torch.tensor([1.1])
        )

        self._high_order_loss: Callable[[torch.Tensor, torch.Tensor], torch.Tensor]
        if high_order_loss_type == HighOrderLoss.Div:
            self._high_order_loss = self._divergence_loss
        elif high_order_loss_type == HighOrderLoss.Hessian:
            self._high_order_loss = self._hessian_loss
        else:
            raise NotImplementedError(f"Loss {high_order_loss_type} is not implemented")

    def _sample_offsurface_points(self, batch_size: int) -> torch.Tensor:
        return self.random_sampler.sample((batch_size, 3))[:, :, 0].to(self.device)  # type: ignore

    def training_step(self, batch: torch.Tensor, batch_idx: int) -> torch.Tensor:
        surface_points, surface_normals = batch
        batch_size = surface_points.shape[0]

        # sample off-surface points
        # There is no guarantee that these points are actually off-surface
        # but probability of landing on the surface is pretty low
        # TODO: consider using a more sophisticated sampling method
        offsurface_points = self._sample_offsurface_points(batch_size)

        surface_points.requires_grad_(True)
        offsurface_points.requires_grad_(True)

        points = torch.cat([surface_points, offsurface_points], dim=0)

        distances = self.sdf_model(points)
        surface_distances = distances[:batch_size]
        offsurface_distances = distances[batch_size:]

        # SDF must be 0 at the boundary
        # Most of works adopt l1 loss for this task, so we do the same
        level_set_loss = torch.abs(surface_distances).mean()
        loss = self.level_set_loss_weight * level_set_loss
        self.log("level_set_loss", level_set_loss, prog_bar=True)

        # the gradient is needed for eikonal and direction losses
        (gradient,) = autograd.grad(
            outputs=distances.sum(),
            inputs=points,
            retain_graph=True,
            create_graph=True,
        )

        # SDF is a solution of the eikonal equation,
        # E_x|∇f(x)| = 1, x ∼ P(D). Where f(x) is the SDF and P(D) is some distribution in R^3
        eikonal_loss = self._eikonal_loss(gradient)
        loss = loss + self.eikonal_loss_weight * eikonal_loss
        self.log("eikonal_loss", eikonal_loss, prog_bar=True)

        if self.grad_direction_loss_weight > 0:
            # SDF gradient must be directed towards the surface normal
            direction_loss = self._grad_direction_loss(
                gradient[:batch_size], surface_normals
            )
            loss = loss + self.grad_direction_loss_weight * direction_loss
            self.log("direction_loss", direction_loss, prog_bar=True)

        # Unfortunately, losses above are not enough to make distance field smooth and consistent
        if self.high_order_loss_weight > 0:
            # it depends on the type of the loss to which points it should be applied
            target_grad = gradient
            target_points = points
            if self.apply_high_order_loss_to == ApplyHOLossTo.Surface:
                target_grad = gradient[:batch_size]
                target_points = surface_points
            elif self.apply_high_order_loss_to == ApplyHOLossTo.OffSurface:
                target_grad = gradient[batch_size:]
                target_points = offsurface_points

            high_order_loss = self._high_order_loss(target_grad, target_points)
            loss = loss + self.high_order_loss_weight * high_order_loss
            self.log("high_order_loss", high_order_loss, prog_bar=True)

        # Off-surface loss is a regularization term that pushes SDF values away from 0
        # There are several kernels that can be used for this task
        # None of them is perfect :(
        # TODO: Implement more kernels, as such as Laplacian or Gaussian
        if self.offsurface_loss_weight > 0:
            offsurface_loss = self._offsurface_loss(offsurface_distances)
            loss = loss + self.offsurface_loss_weight * offsurface_loss
            self.log("offsurface_loss", offsurface_loss, prog_bar=True)

        self.log("loss", loss, prog_bar=True)  # TODO: remove this line 
        return loss

    def _offsurface_loss(self, distance: torch.Tensor) -> torch.Tensor:
        return (
            F.relu(
                self.offsurface_loss_margin * 2
                - F.relu(distance + self.offsurface_loss_margin)
            )
            * 100
        ).mean()

    def _eikonal_loss(self, gradient: torch.Tensor) -> torch.Tensor:
        """
        Eikonal loss is a loss that enforces the gradient to be a unit vector

        SDF is a solution of the eikonal equation,
        E_x|∇f(x)| = 1, x ∼ P(D). Where f(x) is the SDF and P(D) is some distribution in R^3

        Args:
            gradient (torch.Tensor): _description_

        Returns:
            torch.Tensor: _description_
        """
        grad_norm = torch.linalg.norm(gradient, ord=2, dim=-1)
        return (1.0 - grad_norm).abs().mean()

    def _grad_direction_loss(
        self, gradient: torch.Tensor, normals: torch.Tensor
    ) -> torch.Tensor:
        """
        Gradient direction loss
        Compute cosine similarity between the gradient and the gt surface normal

        Args:
            gradient (torch.Tensor): gradient of the SDF
            normals (torch.Tensor): normals of the points used to compute the gradient

        Returns:
            torch.Tensor: 1 - cosine similarity
        """
        return (1 - torch.abs(F.cosine_similarity(gradient, normals, dim=-1))).mean()

    def _divergence_loss(
        self, gradient: torch.Tensor, points: torch.Tensor
    ) -> torch.Tensor:
        """
        Divergence loss
        Is a second-order loss that pushes SDF gradient to be divergence-free

        Described in the following paper:
        "DiGS: Divergence guided shape implicit neural representation for unoriented point clouds"
        https://arxiv.org/pdf/2106.10811

        Args:
            gradient (torch.Tensor): gradient of the SDF at the points
            points (torch.Tensor): points where the gradient is computed

        Returns:
            torch.Tensor: abs value of the divergence
        """
        (dx,) = autograd.grad(
            gradient[:, 0].sum(), points, create_graph=True, retain_graph=True
        )
        (dy,) = autograd.grad(
            gradient[:, 1].sum(), points, create_graph=True, retain_graph=True
        )
        (dz,) = autograd.grad(
            gradient[:, 2].sum(), points, create_graph=True, retain_graph=True
        )

        div = dx[:, 0] + dy[:, 1] + dz[:, 2]
        return torch.abs(div).mean()

    def _hessian_loss(
        self, gradient: torch.Tensor, points: torch.Tensor
    ) -> torch.Tensor:
        """
        Hessian loss

        The gradient directions are expected not to change rapidly,
        so the Hessian matrix should be close to zero.

        Described in the following paper:
        "Critical Regularizations for Neural Surface Reconstruction"
        https://arxiv.org/abs/2206.03087

        Args:
            gradient (torch.Tensor): gradient of the SDF at the points
            points (torch.Tensor): points where the gradient is computed

        Returns:
            torch.Tensor: element-wise matrix 1-norm of the Hessian matrix
        """
        # TODO: consider use vmap feature, it can be much faster
        (dx,) = autograd.grad(
            gradient[:, 0].sum(), points, create_graph=True, retain_graph=True
        )
        (dy,) = autograd.grad(
            gradient[:, 1].sum(), points, create_graph=True, retain_graph=True
        )
        (dz,) = autograd.grad(
            gradient[:, 2].sum(), points, create_graph=True, retain_graph=True
        )

        # Folowwing Critical Regularizations for Neural Surface Reconstruction in the Wild (https://arxiv.org/abs/2206.03087)
        # Using "element-wise matrix 1-norm" which (hopefully) is just a sum of absolute values of all elements
        # https://en.wikipedia.org/wiki/Matrix_norm#%22Entry-wise%22_matrix_norms
        h_norm = torch.concat([dx, dy, dz], dim=1).abs().sum(dim=1)
        return h_norm.mean()

    # TODO: decouple dataset from experiment, add paprameters to config
    def train_dataloader(self) -> DataLoader:
        mesh_dataset = MeshDataset(
            self.mesh_path,
            frac_points_to_sample=self.random_points_per_vertex,
            device="cpu",
        )
        return DataLoader(
            mesh_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            pin_memory=True,
            num_workers=4,
            persistent_workers=False,
        )

