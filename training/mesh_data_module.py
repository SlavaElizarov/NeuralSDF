import pytorch_lightning as pl
from torch.utils.data import DataLoader

from training.dataset import MeshDataset


class SdfDataModule(pl.LightningDataModule):
    def __init__(
        self,
        dataset: MeshDataset,
        batch_size: int = 20000,
    ):
        super().__init__()
        self.dataset = dataset
        self.batch_size = batch_size

    def resample(self):
        self.dataset.resample()

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.dataset,
            batch_size=self.batch_size,
            shuffle=True,
            pin_memory=True,
            prefetch_factor=4,
            num_workers=8,
            persistent_workers=False,
        )
