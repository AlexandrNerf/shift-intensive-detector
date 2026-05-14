import multiprocessing as mp
import torch
from typing import Any, Dict, Optional

from lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import v2 as T
from hydra.utils import instantiate

from omegaconf import DictConfig
from src.data.components.cs_dataset import CSDataset


class CSDataModule(LightningDataModule):
    """`LightningDataModule` для датасета."""
    def __init__(
        self,
        train_images_path: str,
        train_labels_path: str,
        val_images_path: str,
        val_labels_path: str,
        batch_size: int = 64,
        num_workers: int = 4,
        pin_memory: bool = False,
        persistent_workers: bool = True,
        input_shape: int = 512,
        augmentations: Optional[DictConfig] = None,
    ):
        super().__init__()
        self.train_images_path = train_images_path
        self.train_labels_path = train_labels_path
        self.val_images_path = val_images_path
        self.val_labels_path = val_labels_path
        self.batch_size_per_device = batch_size
        self.pin_memory = pin_memory
        self.persistent_workers = persistent_workers
        self.input_shape = input_shape

        if not isinstance(num_workers, int):
            self.num_workers = min(8, mp.cpu_count())
        else:
            self.num_workers = num_workers

        self.augmentations = []
        if augmentations is not None:
            self.augmentations = instantiate(augmentations)

        self.train_transform = T.Compose([
            *self.augmentations,
            T.Resize((self.input_shape, self.input_shape)),
            T.ToImage(),
            T.ToDtype(torch.float32, scale=True),
        ])
        self.val_transform = T.Compose([
            T.Resize((self.input_shape, self.input_shape)),
            T.ToImage(),
            T.ToDtype(torch.float32, scale=True),
        ])

        self.train_dataset: Optional[Dataset] = None
        self.val_dataset: Optional[Dataset] = None
        self.test_dataset: Optional[Dataset] = None

    def prepare_data(self) -> None:
        """Опционально: проверка наличия данных"""
        pass


    def setup(self, stage: Optional[str] = None) -> None:
        """Загрузка данных и создание датасетов"""
        # Адаптация batch size для multi-GPU
        if self.trainer is not None:
            if self.batch_size_per_device % self.trainer.world_size != 0:
                raise RuntimeError(
                    f"Batch size ({self.batch_size_per_device}) не делится на число устройств ({self.trainer.world_size})"
                )
            self.batch_size_per_device = (
                self.batch_size_per_device // self.trainer.world_size
            )

        if not self.train_dataset and not self.val_dataset:

            self.train_dataset = CSDataset(
                images_path=self.train_images_path,
                labels_path=self.train_labels_path,
                transform=self.train_transform
            )

            self.val_dataset = CSDataset(
                images_path=self.val_images_path,
                labels_path=self.val_labels_path,
                transform=self.val_transform
            )
            # не используем тестовый датасет
            self.test_dataset = self.val_dataset

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            dataset=self.train_dataset,
            batch_size=self.batch_size_per_device,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            shuffle=True,
            collate_fn=self.train_dataset.collate_fn,
            drop_last=True,
            persistent_workers=self.persistent_workers
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            dataset=self.val_dataset,
            batch_size=self.batch_size_per_device,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            shuffle=False,
            collate_fn=self.val_dataset.collate_fn,
            drop_last=True,
            persistent_workers=self.persistent_workers
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            dataset=self.test_dataset,
            batch_size=self.batch_size_per_device,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            shuffle=False,
            collate_fn=self.test_dataset.collate_fn,
            drop_last=True,
            persistent_workers=self.persistent_workers
        )

    def teardown(self, stage: Optional[str] = None) -> None:
        """Очистка ресурсов при необходимости"""
        pass

    def state_dict(self) -> Dict[Any, Any]:
        return {}

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        pass
