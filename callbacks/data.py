import numpy as np
import pytorch_lightning as pl


from training.mesh_data_module import SdfDataModule
from training.sdf_experiment import Cloud2SdfExperiment


class ResampleCallback(pl.Callback):
    def __init__(self, resample_every: int = 10):
        """
        Callback to resample dataset every N epochs

        Args:
            resample_every (int, optional): Defaults to 10.
        """        
        self.resample_every = resample_every

    def on_train_epoch_end(self, trainer: pl.Trainer, _: Cloud2SdfExperiment):
        assert trainer.datamodule is not None
        assert isinstance(trainer.datamodule, SdfDataModule)
        if trainer.current_epoch % self.resample_every == 0 and trainer.current_epoch > 0:
            print("Resampling...")
            trainer.datamodule.resample()
