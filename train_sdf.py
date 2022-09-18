from typing import Callable
import torch
from torch.utils.tensorboard.writer import SummaryWriter
from pytorch_lightning.cli import LightningCLI
import pytorch_lightning as pl
from renderer.camera import Camera
from renderer.renderer import SphereTracingRenderer

from training.sdf_experiment import SdfExperiment

class VisualizationCalback(pl.Callback):
    def __init__(self, resolution: int = 256):
        self.resolution = resolution
        
    def render(
        self,
        get_distance: Callable[[torch.Tensor], torch.Tensor],
        dist=1.2,
        elev=0,
        azim=0,
        max_iteration = 70,
        max_depth = 2,
        device = 'cuda'):
        
        camera = Camera(dist, elev, azim, resolution=256, device=device)
        rend = SphereTracingRenderer(camera, max_iteration=max_iteration,max_depth=max_depth, min_dist=0.002)
        frame = rend.render(get_distance).detach()
        return frame
    
    def on_train_epoch_end(self, trainer: pl.Trainer, model: SdfExperiment):
            frame_front = self.render(model.sdf_model, dist=1.3, elev=0, azim=0, device=model.device, max_iteration=40)  # type: ignore
            frame_back = self.render(model.sdf_model, dist=1.3, elev=0, azim=180, device=model.device, max_iteration=40)  # type: ignore
            frame_side = self.render(model.sdf_model, dist=1.3, elev=0, azim=90, device=model.device, max_iteration=40)  # type: ignore
            frame_top = self.render(model.sdf_model, dist=1.3, elev=90, azim=0, device=model.device, max_iteration=40)  # type: ignore

            tensorboard: SummaryWriter = trainer.logger.experiment  # type: ignore
            
            col1 = torch.cat([frame_front, frame_back], dim=0)
            col2 = torch.cat([frame_side, frame_top], dim=0)
            grid = torch.cat([col1, col2], dim=1)               
            tensorboard.add_image('images', grid, dataformats='HWC', global_step=trainer.global_step)

if __name__ == "__main__":
    LightningCLI(
        SdfExperiment, 
        seed_everything_default=42
    )