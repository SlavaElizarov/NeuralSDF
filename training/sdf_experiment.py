import pytorch_lightning as pl
from torch import nn
from torch.nn import functional as F
from torch import autograd


from torch.optim.optimizer import Optimizer
import torch


class SdfExperiment(pl.LightningModule):
    def __init__(self, 
                 sdf_model: nn.Module, 
                 level_set_loss_weight: float = 10,
                 eikonal_loss_weight: float = 1,
                 grad_direction_loss_weight: float = 1,
                 enforce_eikonality: bool = True,
                 ):
        super().__init__()

        self.sdf_model = sdf_model
        self.level_set_loss_weight = level_set_loss_weight
        self.eikonal_loss_weight = eikonal_loss_weight
        self.grad_direction_loss_weight = grad_direction_loss_weight
        self.enforce_eikonality = enforce_eikonality
        self.random_sampler = torch.distributions.uniform.Uniform(torch.tensor([-1.2]), torch.tensor([1.2]))
        
    def configure_optimizers(self) -> Optimizer: 
        """ 
            Can be overrided in config
        """
        return  torch.optim.Adam(self.sdf_model.parameters(), lr=0.00001, amsgrad=True)
    
    def eikonal_loss(self, gradient: torch.Tensor) -> torch.Tensor:
        grad_norm = torch.linalg.norm(gradient, p=2, dim=-1)
        return F.mse_loss(grad_norm, torch.ones_like(grad_norm)) 
    
    def grad_direction_loss(self, gradient: torch.Tensor, normals: torch.Tensor) -> torch.Tensor:
        grad_dot_normal = torch.sum(gradient * normals, dim=-1)
        return F.mse_loss(grad_dot_normal, torch.ones_like(grad_dot_normal))
        

    def training_step(self, batch: torch.Tensor, batch_idx: int, optimizer_idx: int) -> torch.Tensor:
        batch_vertices, batch_normals = batch
        batch_size = batch_vertices.shape[0]
        
        batch_vertices.requires_grad_(True)
        sdf_output = self.sdf_model.forward(batch_vertices)

        # SDF must be 0 at the boundary
        level_set_loss = F.mse_loss(sdf_output, torch.zeros_like(sdf_output))

        # ||grad(SDF)|| must be 1 (eikonal equation) 
        # and share direction with a normal to a point on the surface
        gradient, = autograd.grad(
            outputs=sdf_output.sum(), inputs=batch_vertices, retain_graph=True, create_graph=True,
        )
        
        eikonal_loss = self.eikonal_loss(gradient)
        grad_direction_loss = self.grad_direction_loss(gradient, batch_normals)
        
        # regularization on random points to enforce eikonal properties
        # there is a problem with this tecniqe, it enforce magnitude of the gradient 
        # but do nothing with direction. 
        # TODO: use divergence based regularization proposed in https://arxiv.org/abs/2106.10811

        if self.enforce_eikonality: 
            random_points = self.random_sampler.sample((batch_size, 3))  # type: ignore
            random_points = random_points.to(self.device).requires_grad_(True)
            sdf_output = self.sdf_model.forward(random_points)
            gradient, = autograd.grad(
                outputs=sdf_output.sum(), inputs=batch_vertices, retain_graph=True, create_graph=True,
            )
            eikonal_loss += self.eikonal_loss(gradient)
        
        loss = level_set_loss * self.level_set_loss_weight + \
                eikonal_loss * self.eikonal_loss_weight + \
                grad_direction_loss * self.grad_direction_loss_weight 
                
        self.log('level_set_loss', level_set_loss, prog_bar=True)
        self.log('eikonal_loss', eikonal_loss, prog_bar=True)
        self.log('grad_direction_loss', grad_direction_loss, prog_bar=True)
        
        return loss