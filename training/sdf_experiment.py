import pytorch_lightning as pl
import torch
from torch import nn
from torch.nn import functional as F
from torch import autograd
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ExponentialLR
from models.sdf import SDF

from training.dataset import MeshDataset


class SdfExperiment(pl.LightningModule):
    def __init__(self, 
                 sdf_model: SDF, 
                 mesh_path: str,
                 batch_size: int = 1024,
                 level_set_loss_weight: float = 10,
                 eikonal_loss_weight: float = 1,
                 grad_direction_loss_weight: float = 1,
                 enforce_eikonality: bool = True,
                 offsurface_loss_weight: float = 0,
                 offsurface_loss_margin: float = 0.0025,
                 divergence_loss_weight: float = 0.001,
                 learning_rate: float = 0.00001,
                 learning_rate_decay: float = 0.94,
                 kernel_coefficient : float = 70,
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
        self.divergence_loss_weight = divergence_loss_weight
        self.offsurface_loss_margin = offsurface_loss_margin
        self.mesh_path = mesh_path # it's a bit of an antipatern to have this here TODO: decouple data from experiment
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.learning_rate_decay = learning_rate_decay
        self.kernel_coefficient = kernel_coefficient
        
        self.random_sampler = torch.distributions.uniform.Uniform(torch.tensor([-1.1]), torch.tensor([1.1]))
                
    def configure_optimizers(self): 
        """ 
            Can be overrided in config
        """
        optimizer = torch.optim.Adam(self.sdf_model.parameters(), lr=self.learning_rate, amsgrad=True)
        scheduler = ExponentialLR(optimizer, gamma=self.learning_rate_decay)
        return  [optimizer], [scheduler]
        
    def eikonal_loss(self, gradient: torch.Tensor) -> torch.Tensor:
        grad_norm = torch.linalg.norm(gradient, ord=2, dim=-1)
        return F.l1_loss(grad_norm, torch.ones_like(grad_norm)) 
    
    def grad_direction_loss(self, gradient: torch.Tensor, normals: torch.Tensor) -> torch.Tensor:
        # grad_dot_normal = torch.sum(gradient * normals, dim=-1)
        # return F.mse_loss(grad_dot_normal, torch.ones_like(grad_dot_normal))
        return (1 - torch.abs(F.cosine_similarity(gradient, normals, dim=-1))).mean()
    
    def divergence_loss(self, points: torch.Tensor, gradient: torch.Tensor) -> torch.Tensor:
        dx, = autograd.grad(gradient[:,0].sum(), points, create_graph=True, retain_graph=True)
        dy, = autograd.grad(gradient[:,1].sum(), points, create_graph=True, retain_graph=True)
        dz, = autograd.grad(gradient[:,2].sum(), points, create_graph=True, retain_graph=True)
        
        div = dx[:, 0] + dy[:, 1] + dz[:, 2]
        return torch.abs(div).mean()
        

    def training_step(self, batch: torch.Tensor, batch_idx: int) -> torch.Tensor:
        batch_vertices, batch_normals = batch
        batch_size = batch_vertices.shape[0]
        
        batch_vertices.requires_grad_(True)
        sdf_output = self.sdf_model.forward(batch_vertices)

        # SDF must be 0 at the boundary
        level_set_loss = torch.abs(sdf_output).mean()

        # ||grad(SDF)|| must be 1 (eikonal equation) 
        # and share direction with a normal to a point on the surface
        gradient, = autograd.grad(
            outputs=sdf_output.sum(), inputs=batch_vertices, retain_graph=True, create_graph=True,
        )
        
        eikonal_loss = self.eikonal_loss(gradient)
        grad_direction_loss = self.grad_direction_loss(gradient, batch_normals)
        # manifold_loss = self.manifold_loss(batch_vertices, batch_normals)
        
        
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
            divergence_loss = self.divergence_loss(random_points, gradient)
            eikonal_loss += self.eikonal_loss(gradient)
            
                
        
        loss = level_set_loss * self.level_set_loss_weight + \
                eikonal_loss * self.eikonal_loss_weight + \
                grad_direction_loss * self.grad_direction_loss_weight + \
                divergence_loss * self.divergence_loss_weight
        
        # there is a big problem with this loss. There is no garauntee that random points are off the surface.
        if self.offsurface_loss_weight > 0:
            # TODO: careful analysis of this loss is required
            offsurface_loss_plus = (F.relu(self.offsurface_loss_margin * 2 - F.relu(sdf_output + self.offsurface_loss_margin)) * 100).mean()
            # offsurface_loss_minus = (F.relu(sdf_output * -1) * 100).mean()
            offsurface_loss = offsurface_loss_plus #+ offsurface_loss_minus
            # a = a[torch.abs(sdf_output) < 0.01]
            # offsurface_loss = torch.exp(-(sdf_output / 0.02)**2).mean()
            
            # TODO: Consider kernel size anealing 
            # offsurface_loss = torch.exp(-(self.kernel_coefficient * sdf_output) ** 2 +0.001).mean()

            # soft_delta = 2 / (torch.pi*(4 + sdf_output**2))
            # offsurface_loss = soft_delta.mean()
            loss += offsurface_loss * self.offsurface_loss_weight
            
            self.log('offsurface_loss', offsurface_loss, prog_bar=True)
        
        # if isinstance(self.sdf_model, SelfModulatedSiren):
        #     tensorboard: SummaryWriter = self.trainer.logger.experiment  # type: ignore
        #     for i, amp in enumerate(self.sdf_model.amplitide_mod):
        #         tensorboard.add_histogram(f'amplitide_mod_{i}', amp, global_step=self.global_step)
        #     for i, frq in enumerate(self.sdf_model.frequency_mod):
        #         tensorboard.add_histogram(f'frequency_mod_{i}', frq, global_step=self.global_step)

                
        self.log('level_set_loss', level_set_loss, prog_bar=True)
        self.log('eikonal_loss', eikonal_loss, prog_bar=True)
        self.log('grad_direction_loss', grad_direction_loss, prog_bar=True)
        self.log('divergence_loss', divergence_loss, prog_bar=True)
        
        return loss
    
    # TODO: decouple dataset from experiment, add paprameters to config
    def train_dataloader(self) -> DataLoader:
        mesh_dataset = MeshDataset(self.mesh_path, frac_points_to_sample=3.0, device='cpu')
        return DataLoader(mesh_dataset, batch_size=self.batch_size,
                          shuffle=True, 
                          pin_memory=True, 
                          num_workers=4,
                          persistent_workers = False)
        
    # learning rate warm-up
    def optimizer_step(
        self,
        epoch,
        batch_idx,
        optimizer,
        optimizer_idx,
        optimizer_closure,
        on_tpu=False,
        using_native_amp=False,
        using_lbfgs=False,
    ):
        # update params
        optimizer.step(closure=optimizer_closure)

        # skip the first 500 steps
        if self.trainer.global_step < 300:
            lr_scale = min(1.0, float(self.trainer.global_step + 1) / 300.0)
            for pg in optimizer.param_groups:
                pg["lr"] = lr_scale * self.learning_rate
