import pytorch_lightning as pl
import torch
from torch import nn
from torch.nn import functional as F
from torch import autograd
from torch.utils.data import DataLoader
from torch.optim.optimizer import Optimizer
from models.siren import Siren

from training.dataset import MeshDataset


class SdfExperiment(pl.LightningModule):
    def __init__(self, 
                 sdf_model: Siren, 
                 mesh_path: str,
                 batch_size: int = 1024,
                 level_set_loss_weight: float = 10,
                 eikonal_loss_weight: float = 1,
                 grad_direction_loss_weight: float = 1,
                 enforce_eikonality: bool = True,
                 offsurface_loss_weight: float = 0.,
                 ):
        """
        Train a SDF model of a mesh.

        Args:
            sdf_model (Siren): _description_
            mesh_path (str): _description_
            batch_size (int, optional): _description_. Defaults to 1024.
            level_set_loss_weight (float, optional): _description_. Defaults to 10.
            eikonal_loss_weight (float, optional): _description_. Defaults to 1.
            grad_direction_loss_weight (float, optional): _description_. Defaults to 1.
            enforce_eikonality (bool, optional): _description_. Defaults to True.
            offsurface_loss_weight (float, optional): _description_. Defaults to 0..
        """
        super().__init__()
        

        self.sdf_model = sdf_model
        self.level_set_loss_weight = level_set_loss_weight
        self.eikonal_loss_weight = eikonal_loss_weight
        self.grad_direction_loss_weight = grad_direction_loss_weight
        self.enforce_eikonality = enforce_eikonality
        self.offsurface_loss_weight = offsurface_loss_weight
        self.mesh_path = mesh_path # it's a bit of an antipatern to have this here TODO: decouple data from experiment
        self.batch_size = batch_size
        
        
        self.random_sampler = torch.distributions.uniform.Uniform(torch.tensor([-1.2]), torch.tensor([1.2]))
        
        self.save_hyperparameters()
        
    def configure_optimizers(self) -> Optimizer: 
        """ 
            Can be overrided in config
        """
        return  torch.optim.Adam(self.sdf_model.parameters(), lr=0.00001, amsgrad=True)
    
    def eikonal_loss(self, gradient: torch.Tensor) -> torch.Tensor:
        grad_norm = torch.linalg.norm(gradient, ord=2, dim=-1)
        return F.mse_loss(grad_norm, torch.ones_like(grad_norm)) 
    
    def grad_direction_loss(self, gradient: torch.Tensor, normals: torch.Tensor) -> torch.Tensor:
        # grad_dot_normal = torch.sum(gradient * normals, dim=-1)
        # return F.mse_loss(grad_dot_normal, torch.ones_like(grad_dot_normal))
        return (1 - torch.abs(F.cosine_similarity(gradient, normals, dim=-1))).mean()
        

    def training_step(self, batch: torch.Tensor, batch_idx: int) -> torch.Tensor:
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
        
        
        if self.enforce_eikonality or self.offsurface_loss_weight > 0:
            random_points = self.random_sampler.sample((batch_size, 3))  # type: ignore
            random_points = random_points.to(self.device).requires_grad_(True)[:, :, 0]
            sdf_output = self.sdf_model.forward(random_points)
            
        # regularization on random points to enforce eikonal properties
        # there is a problem with this tecniqe, it enforce magnitude of the gradient 
        # but do nothing with direction. 
        # TODO: use divergence based regularization proposed in https://arxiv.org/abs/2106.10811
        if self.enforce_eikonality: 
            gradient, = autograd.grad(
                outputs=sdf_output.sum(), inputs=random_points, retain_graph=True, create_graph=True,
            )
            eikonal_loss += self.eikonal_loss(gradient)
            
                
        
        loss = level_set_loss * self.level_set_loss_weight + \
                eikonal_loss * self.eikonal_loss_weight + \
                grad_direction_loss * self.grad_direction_loss_weight 
        
        # there is a big problem with this loss. There is no garauntee that random points are off the surface.
        if self.offsurface_loss_weight > 0:
            offsurface_loss = torch.exp(-1e2 * torch.abs(sdf_output)).mean()
            loss += offsurface_loss * self.offsurface_loss_weight
            
            self.log('offsurface_loss', offsurface_loss, prog_bar=True)

                
        self.log('level_set_loss', level_set_loss, prog_bar=True)
        self.log('eikonal_loss', eikonal_loss, prog_bar=True)
        self.log('grad_direction_loss', grad_direction_loss, prog_bar=True)
        
        return loss
    
    def train_dataloader(self) -> DataLoader:
        mesh_dataset = MeshDataset(self.mesh_path)
        return DataLoader(mesh_dataset, batch_size=self.batch_size, shuffle=True)