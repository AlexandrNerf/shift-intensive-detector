from typing import Any, Dict, Tuple, Optional, List
import torch
from torch import nn
import torchvision
import torch.nn.functional as F
from lightning import LightningModule
from torchmetrics import MaxMetric, MeanMetric
from torchmetrics.detection.mean_ap import MeanAveragePrecision
from torchmetrics.detection.giou import GeneralizedIntersectionOverUnion
from src.utils.metrics.metrics_fast import TorchLocalizationConfusion


class BaseDetectionModel(LightningModule):
    def __init__(
        self,
        net: str,
        pretrained: bool,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler,
        compile: bool
    ) -> None:
        super().__init__()

        self.save_hyperparameters(logger=False)

        # Подгружаем предобученную модель Faster R-CNN с MobileNetV3
        self.model = torchvision.models.detection.fasterrcnn_mobilenet_v3_large_fpn(
            num_classes=2,
            pretrained=self.hparams.pretrained
        )

        # Если нужно, можно настроить кастомную модель (например, на основе других слоев)
        # if self.hparams.net == 'your_model':
        #   self.model = Model(*args)    

        # Метрики (в классе TorchLocalizationConfussion мы можем настроить порог подсчёта iou)
        #self.val_metric = TorchLocalizationConfusion(iou_thresh=0.5)
        self.map_metric = MeanAveragePrecision(box_format="xyxy",
            iou_type="bbox",
            iou_thresholds=[0.5, 0.75],
            class_metrics=True
        )

    def forward(self, x):
        """Проход через модель."""
        return self.model(x)

    def model_step(self, batch):
        """Шаг для вычисления потерь."""
        images, targets = batch
        loss_dict = self.model(images, targets)
        losses = sum(loss for loss in loss_dict.values())
        return losses

    def training_step(self, batch, batch_idx: int):
        """Один шаг обучения."""
        loss = self.model_step(batch)

        # Логирование потерь
        self.log("train/loss", loss, on_step=True, on_epoch=False, prog_bar=True)

        return loss

    def on_train_epoch_end(self) -> None:
        """Шаг, который выполняется в конце каждой эпохи."""
        pass

    def validation_step(self, batch, batch_idx):
        images, targets = batch
        preds = self.model(images)

        # ОБЯЗАТЕЛЬНО: preds / targets — списки dict'ов (torchvision format)
        self.map_metric.update(preds, targets)

    def on_validation_epoch_end(self):
        metrics = self.map_metric.compute()
        self.map_metric.reset()

        self.log_dict(
            {
                "val/mAP": metrics["map"],           # mAP@[.5:.95]
                "val/mAP50": metrics["map_50"],      # mAP@0.5
                "val/mAP75": metrics["map_75"],      # mAP@0.75
                "val/recall": metrics["mar_100"],    # recall
            },
            prog_bar=True,
        )

    def test_step(self, batch, batch_idx):
        images, targets = batch
        preds = self.model(images)
        self.map_metric.update(preds, targets)

    def on_test_epoch_end(self):
        metrics = self.map_metric.compute()
        self.map_metric.reset()

        self.log_dict(
            {
                "test/mAP": metrics["map"],
                "test/mAP50": metrics["map_50"],
                "test/mAP75": metrics["map_75"],
                "test/recall": metrics["mar_100"],
            }
        )
    
    def configure_optimizers(self):
        """Настройка оптимизаторов и планировщиков."""
        optimizer = self.hparams.optimizer(params=[p for p in self.model.parameters() if p.requires_grad])
        if self.hparams.scheduler is not None:
            scheduler = self.hparams.scheduler(optimizer=optimizer)
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": "val/mAP50", # здесь можно настроить, по какой метрике будет мониторинг
                },
            }
        return {"optimizer": optimizer}
    
    def on_after_backward(self):
        """Шаг после обратного распространения, для обрезки градиентов."""
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=5)

    def setup(self, stage: str):
        """Хук для подготовки модели к тренировке или валидации."""
        # if stage == "fit":
        #     self.model = torch.compile(self.model)
        pass
