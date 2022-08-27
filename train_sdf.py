from pytorch_lightning.utilities.cli import LightningCLI

from training.sdf_experiment import SdfExperiment

if __name__ == "__main__":
    LightningCLI(
        SdfExperiment, 
        seed_everything_default=42
    )