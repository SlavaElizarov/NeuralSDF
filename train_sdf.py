from pytorch_lightning.cli import LightningCLI
from training.mesh_data_module import SdfDataModule

from training.sdf_experiment import Cloud2SdfExperiment
import torch

torch.set_float32_matmul_precision("medium")


if __name__ == "__main__":
    LightningCLI(
        Cloud2SdfExperiment, datamodule_class=SdfDataModule, seed_everything_default=42
    )
