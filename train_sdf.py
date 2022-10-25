from typing import Callable
import torch
from torch import nn
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
        max_iteration=70,
        max_depth=2,
        device="cuda",
    ):

        camera = Camera(dist, elev, azim, resolution=256, device=device)
        rend = SphereTracingRenderer(
            camera, max_iteration=max_iteration, max_depth=max_depth, min_dist=0.001
        )
        frame = rend.render(get_distance).detach()
        return frame

    def on_train_epoch_end(self, trainer: pl.Trainer, model: SdfExperiment):
        sdf_model = model.sdf_model
        frame_front = self.render(sdf_model, dist=1.3, elev=0, azim=0, device=model.device, max_iteration=40)  # type: ignore
        frame_back = self.render(sdf_model, dist=1.3, elev=0, azim=180, device=model.device, max_iteration=40)  # type: ignore
        frame_side = self.render(sdf_model, dist=1.3, elev=0, azim=90, device=model.device, max_iteration=40)  # type: ignore
        frame_top = self.render(sdf_model, dist=1.3, elev=90, azim=0, device=model.device, max_iteration=40)  # type: ignore

        tensorboard: SummaryWriter = trainer.logger.experiment  # type: ignore

        col1 = torch.cat([frame_front, frame_back], dim=0)
        col2 = torch.cat([frame_side, frame_top], dim=0)
        grid = torch.cat([col1, col2], dim=1)
        tensorboard.add_image(
            "images", grid, dataformats="HWC", global_step=trainer.global_step
        )


class ActivationDistributionCalback(pl.Callback):
    def __init__(self, log_every_n_batches: int = 30, 
                 number_of_samples: int = 16):
        self.log_every_n_batches = log_every_n_batches
        self.number_of_samples = number_of_samples

    def on_fit_start(self, trainer: pl.Trainer, experiment: SdfExperiment):
        tensorboard: SummaryWriter = trainer.logger.experiment  # type: ignore

        def hook_wrapper(name: str, trainer: pl.Trainer):
            def hook(
                module: nn.Module, input: torch.Tensor, output: torch.Tensor
            ) -> None:
                if trainer.global_step % self.log_every_n_batches != 0:
                    return
                if isinstance(output, tuple):
                    output = output[0]
                tensorboard.add_histogram(
                    f"{name} : {module.__class__.__name__}",
                    output[: self.number_of_samples],
                    global_step=trainer.global_step,
                )

            return hook

        for name, module in experiment.sdf_model.named_modules():
            hook = hook_wrapper(name, trainer)
            module.register_forward_hook(hook)
            
class SirenFrequencyCalback(pl.Callback):
    def __init__(self, log_every_n_batches: int = 30, 
                 number_of_samples: int = 64):
        self.log_every_n_batches = log_every_n_batches
        self.number_of_samples = number_of_samples
        
    def on_train_batch_end(self, trainer: pl.Trainer, pl_module: SdfExperiment, outputs, batch, batch_idx: int) -> None:
        if trainer.global_step % self.log_every_n_batches != 0:
            return
        tensorboard: SummaryWriter = trainer.logger.experiment
        
        freq = pl_module.sdf_model[0].weight.detach().norm(dim=-1, keepdim=False)
        tensorboard.add_histogram(
                    f"Siren freq",
                    freq[: self.number_of_samples],
                    global_step=trainer.global_step,
                )
            



if __name__ == "__main__":
    LightningCLI(SdfExperiment, seed_everything_default=42)
