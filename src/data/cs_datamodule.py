import multiprocessing as mp
import os
import pandas as pd
import numpy as np
from typing import Any, Dict, Optional

from lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset, RandomSampler, SequentialSampler
from torchvision.transforms.v2 import (
    Compose,
    ToTensor,
    Normalize,
    RandomGrayscale,
    RandomPerspective,
    RandomPhotometricDistort,
    Resize
)
from hydra.utils import instantiate

from omegaconf import DictConfig, OmegaConf
from src.data.components.cs_dataset import CSDataset


class CSDataModule(LightningDataModule):
    """`LightningDataModule` для датасета."""
    def __init__(
        self,
        batch_size: int = 64,
        num_workers: int = 4,
        pin_memory: bool = False,
        persistent_workers: bool = True,
        input_shape: int = 512,
        augmentations: str = 'default',
    ):
        super().__init__()
        self.save_hyperparameters(logger=False)
        self.input_shape = self.hparams.input_shape
        self.batch_size_per_device = self.hparams.batch_size

        if not isinstance(self.hparams.num_workers, int):
            self.num_workers = min(8, mp.cpu_count())
        else:
            self.num_workers = self.hparams.num_workers

        augs_path = f'configs/data/augmentations/{self.hparams.augmentations}.yaml'
        self.augmentations = instantiate(OmegaConf.load(augs_path))
        self.train_transform = Compose([
            ToTensor(),
            *self.augmentations
        ])
        self.val_transform = Compose([
            ToTensor(),
        ])

        self.train_dataset: Optional[Dataset] = None
        self.val_dataset: Optional[Dataset] = None
        self.test_dataset: Optional[Dataset] = None

    def prepare_data(self) -> None:
        """Опционально: проверка наличия данных"""
        pass
        #self.dataset_frame = pd.read_parquet(self.hparams.dataset)


    def setup(self, stage: Optional[str] = None) -> None:
        """Загрузка данных и создание датасетов"""
        # Адаптация batch size для multi-GPU
        if self.trainer is not None:
            if self.hparams.batch_size % self.trainer.world_size != 0:
                raise RuntimeError(
                    f"Batch size ({self.hparams.batch_size}) не делится на число устройств ({self.trainer.world_size})"
                )
            self.batch_size_per_device = (
                self.hparams.batch_size // self.trainer.world_size
            )

        if not self.train_dataset and not self.val_dataset:

            self.train_dataset = CSDataset(
                task='train',
                batch_size=self.hparams.batch_size,
                transform=self.train_transform
            )

            self.val_dataset = CSDataset(
                task='val',
                batch_size=self.hparams.batch_size,
                transform=self.val_transform
            )
            self.test_dataset = self.val_dataset

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            dataset=self.train_dataset,
            batch_size=self.batch_size_per_device,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=True,
            collate_fn=self.train_dataset.collate_fn,
            drop_last=True,
            persistent_workers=self.hparams.persistent_workers
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            dataset=self.val_dataset,
            batch_size=self.batch_size_per_device,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
            collate_fn=self.val_dataset.collate_fn,
            drop_last=True,
            persistent_workers=self.hparams.persistent_workers
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            dataset=self.test_dataset,
            batch_size=self.batch_size_per_device,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
            collate_fn=self.test_dataset.collate_fn,
            drop_last=True,
            persistent_workers=self.hparams.persistent_workers
        )

    def teardown(self, stage: Optional[str] = None) -> None:
        """Очистка ресурсов при необходимости"""
        pass

    def state_dict(self) -> Dict[Any, Any]:
        return {}

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        pass
