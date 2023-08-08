
from training.sdf_experiment import Cloud2SdfExperiment
from models.sdf import GradComputationType
from layers.encodings import GridEmbedding
import pytorch_lightning as pl


class CurvatureLossScheduler(pl.Callback):
    def __init__(self, warmup_steps: int = 1000, decay_steps: int = 10000):
        self.warmup_steps = warmup_steps
        self.decay_steps = decay_steps
        self._config_loss_weight = None

    def on_train_batch_start(
        self, trainer: pl.Trainer, experiment: Cloud2SdfExperiment, batch, batch_idx
    ) -> None:
        if experiment.laplacian_loss is None:
            return

        if self._config_loss_weight is None:
            self._config_loss_weight = experiment.laplacian_loss.weight

        if trainer.global_step < self.warmup_steps:
            experiment.laplacian_loss.weight = self._config_loss_weight * \
                (trainer.global_step / self.warmup_steps)
        else:
            step = trainer.global_step - self.warmup_steps
            experiment.laplacian_loss.weight = self._config_loss_weight - \
                (step / self.decay_steps) * self._config_loss_weight
            if experiment.laplacian_loss.weight <= 0:
                experiment.laplacian_loss = None


class MaskLevelsCallback(pl.Callback):
    def __init__(self, steps_per_level: int = 512, encodimg_field_name: str = "encoding"):
        self.steps_per_level = steps_per_level
        self.encodimg_field_name = encodimg_field_name
        self._encoding = None

    def on_train_batch_end(
        self, trainer: pl.Trainer, experiment: Cloud2SdfExperiment, outputs, batch, batch_idx
    ) -> None:
        if self._encoding is None:
            assert hasattr(experiment.sdf_model, self.encodimg_field_name)
            encoding = getattr(experiment.sdf_model, self.encodimg_field_name)
            assert isinstance(encoding, GridEmbedding)
            self._encoding = encoding

        if self._encoding.mask_k_levels > 0 and trainer.global_step > 0 and \
           trainer.global_step % self.steps_per_level == 0:
            self._encoding.mask_k_levels -= 1
            print(f"Unmasking level {self._encoding.mask_k_levels}")


class GradDeltaScheduler(pl.Callback):
    def __init__(self, steps_per_level: int = 512,
                 min_resolution: int = 16,
                 max_resolution: int = 512,
                 swith_to_analytical_after_max: bool = False):
        assert min_resolution < max_resolution

        self.swith_to_analytical_after_max = swith_to_analytical_after_max
        num_levels = max_resolution / min_resolution
        self.total_steps = num_levels * steps_per_level
        self.min_resolution = min_resolution
        self.delta_init = None
        self.delta_final = 1 / max_resolution

    def on_train_batch_start(
        self, trainer: pl.Trainer, experiment: Cloud2SdfExperiment, batch, batch_idx
    ) -> None:
        assert experiment.sdf_model.grad_parameters is not None

        if self.delta_init is None:
            self.delta_init = min(
                experiment.sdf_model.grad_parameters.delta, 1. / self.min_resolution)
            print(f"Grad delta init: {self.delta_init}")

        if trainer.global_step < self.total_steps:
            delta = self.delta_init - \
                (trainer.global_step / self.total_steps) * \
                (self.delta_init - self.delta_final)
            experiment.sdf_model.grad_parameters.delta = delta
            # print(f"Grad delta: {delta}")
        else:
            if self.swith_to_analytical_after_max:
                experiment.sdf_model.grad_parameters.computation_type = GradComputationType.ANALYTICAL
            else:
                experiment.sdf_model.grad_parameters.delta = self.delta_final
