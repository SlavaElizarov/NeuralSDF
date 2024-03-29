from typing import Optional, TypeVar
import pytorch_lightning as pl
import torch
from losses.losses import (
    EikonalLoss,
    GradientDirectionLoss,
    LaplacianLoss,
    LossBase,
    OffSurfaceGTLoss,
    OffSurfaceLoss,
    SurfaceLoss,
    PullLoss,
)
from models.sdf import SDF

T = TypeVar("T", bound=Optional[LossBase])


class Cloud2SdfExperiment(pl.LightningModule):
    def __init__(
        self,
        sdf_model: SDF,
        level_set_loss: SurfaceLoss,
        eikonal_loss: EikonalLoss,
        grad_direction_loss: Optional[GradientDirectionLoss],
        offsurface_loss: Optional[OffSurfaceLoss] = None,
        laplacian_loss: Optional[LaplacianLoss] = None,
        offsurface_gt_loss: Optional[OffSurfaceGTLoss] = None,
        pull_loss: Optional[PullLoss] = None,
    ):
        super().__init__()
        self.sdf_model = sdf_model
        self.level_set_loss = self._inject_logger(level_set_loss)
        self.eikonal_loss = self._inject_logger(eikonal_loss)
        self.grad_direction_loss = self._inject_logger(grad_direction_loss)
        self.offsurface_loss = self._inject_logger(offsurface_loss)
        self.laplacian_loss = self._inject_logger(laplacian_loss)
        self.offsurface_gt_loss = self._inject_logger(offsurface_gt_loss)
        self.pull_loss = self._inject_logger(pull_loss)

    def training_step(self, batch: torch.Tensor, batch_idx: int) -> torch.Tensor:
        surface_points, surface_normals, offsurface_points, offsurface_distances_gt = batch
        batch_size = surface_points.shape[0]

        # we need to compute gradients for the points
        # so we need to set requires_grad to True
        surface_points.requires_grad_(True)
        offsurface_points.requires_grad_(True)

        points = torch.cat([surface_points, offsurface_points], dim=0)

        if self.laplacian_loss is not None:
            distances, gradient, laplacian = self.sdf_model.forward_with_grad_and_laplacian(points)
        else:
            distances, gradient = self.sdf_model.forward_with_grad(points)

        surface_distances = distances[:batch_size]
        offsurface_distances = distances[batch_size:]
        surface_grad = gradient[:batch_size]
        offsurface_grad = gradient[batch_size:]

        # SDF must be 0 at the boundary
        # Most of works adopt l1 loss for this task, so we do the same
        loss = self.level_set_loss(surface_distances)

        # SDF is a solution of the eikonal equation,
        # E_x|∇f(x)| = 1, x ∼ P(D). Where f(x) is the SDF and P(D) is some distribution in R^3
        loss += self.eikonal_loss(gradient)

        if self.grad_direction_loss is not None:
            # SDF gradient must be directed towards the surface normal
            loss += self.grad_direction_loss(surface_grad, surface_normals)

        # Unfortunately, losses above are not always enough to make distance field smooth and consistent
        if self.laplacian_loss is not None:
            loss += self.laplacian_loss(laplacian, gradient, distances)

        # SDF must be positive outside the surface
        # this loss aims to reduce a shadow geomtry around the surface
        if self.offsurface_loss is not None:
            loss += self.offsurface_loss(offsurface_distances, offsurface_grad)

        if self.offsurface_gt_loss is not None:
            loss += self.offsurface_gt_loss(offsurface_distances,
                                            offsurface_distances_gt)
            
        if self.pull_loss is not None:
            loss += self.pull_loss(offsurface_points, offsurface_distances_gt[:, None], offsurface_grad, self.sdf_model)
        
        self.log("loss", loss, prog_bar=True)
        return loss

    def _inject_logger(self, loss: T) -> T:
        if loss is not None:
            loss.set_logger(lambda name, loss: self.log(
                name, loss, prog_bar=True))
        return loss
