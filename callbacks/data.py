import numpy as np
import pytorch_lightning as pl


from training.mesh_data_module import SdfDataModule
from training.sdf_experiment import SdfExperiment


class ResampleCallback(pl.Callback):
    def on_train_epoch_end(self, trainer: pl.Trainer, _: SdfExperiment):
        assert trainer.datamodule is not None
        assert isinstance(trainer.datamodule, SdfDataModule)
        if np.random.rand() < 0.2:
            print("Resampling...")
            trainer.datamodule.resample()
