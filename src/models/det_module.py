from typing import Any, Dict, Tuple, Optional, List
import torch
from torch import nn
import torchvision
import torch.nn.functional as F
from lightning import LightningModule
from torchmetrics import MaxMetric, MeanMetric
from doctr.utils.metrics import LocalizationConfusion
from src.utils.metrics.metrics_fast import TorchLocalizationConfusion


class BaseDetectionModel(LightningModule):
    def __init__(
        self,
        net: str,
        pretrained: bool,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler,
        compile: bool,
        cuda: bool
    ) -> None:
        super().__init__()

        self.save_hyperparameters(logger=False)

        # Подгружаем предобученную модель Faster R-CNN с Resnet50
        self.model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=self.hparams.pretrained)
        
        if self.hparams.cuda and torch.cuda.is_available():
            self.model.to('cuda')

        # Если нужно, можно настроить кастомную модель (например, на основе других слоев)
        # if self.hparams.net == 'your_model':
        #   self.model = Model(*args)    

        # Метрики (в классе TorchLocalizationConfussion мы можем настроить порог подсчёта iou)
        self.val_metric = TorchLocalizationConfusion(iou_thresh=0.5)
        self.precision = MeanMetric()
        self.recall = MeanMetric()
        self.iou = MeanMetric()

        # Метрики для логирования
        self.train_loss = MeanMetric()
        self.val_loss = MeanMetric()
        self.test_loss = MeanMetric()

        self.precision_best = MaxMetric()
        self.recall_best = MaxMetric()
        self.iou_best = MaxMetric()

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
        self.train_loss(loss)
        self.log("train/loss", self.train_loss, on_step=True, on_epoch=False, prog_bar=True)

        return loss

    def on_train_epoch_end(self) -> None:
        """Шаг, который выполняется в конце каждой эпохи."""
        pass

    def validation_step(self, batch, batch_idx: int):
        """Шаг для валидации."""
        images, targets = batch

        # Метрики для точности, полноты и IoU
        with torch.no_grad():
            preds = self.model(images)

        for p, t in zip(preds, targets):
            pred_boxes = p["boxes"]
            gt_boxes = t["boxes"]

            # Если нет GT и нет предсказаний, ничего не делаем
            if pred_boxes.numel() == 0 and gt_boxes.numel() == 0:
                continue

            # Обновляем метрику
            self.val_metric.update(pred_boxes, gt_boxes)

        precision, recall, iou = self.val_metric.summary()
        
        self.precision(precision)
        self.recall(recall)
        self.iou(iou)

        # Логирование метрик
        self.log("val/precision", self.precision, on_step=True, on_epoch=False, prog_bar=True)
        self.log("val/recall", self.recall, on_step=True, on_epoch=False, prog_bar=True)
        self.log("val/iou", self.iou, on_step=True, on_epoch=False, prog_bar=True)

    def on_validation_epoch_end(self) -> None:
        """Шаг, который выполняется в конце каждой эпохи валидации."""
        self.iou_best.update(self.iou.compute())
        self.recall_best.update(self.recall.compute())
        self.precision_best.update(self.precision.compute())
        
        # Логирование лучших метрик
        self.log("val/best_precision", self.precision_best.compute(), on_step=False, on_epoch=True, prog_bar=True)
        self.log("val/best_recall", self.recall_best.compute(), on_step=False, on_epoch=True, prog_bar=True)
        self.log("val/best_iou", self.iou_best.compute(), on_step=False, on_epoch=True, prog_bar=True)

    def test_step(self, batch, batch_idx: int):
        """Шаг тестирования."""
        images, targets = batch

        # Метрики для точности, полноты и IoU
        with torch.no_grad():
            preds = self.model(images)

        for p, t in zip(preds, targets):
            pred_boxes = p["boxes"]
            gt_boxes = t["boxes"]

            # Если нет GT и нет предсказаний, ничего не делаем
            if pred_boxes.numel() == 0 and gt_boxes.numel() == 0:
                continue

            # Обновляем метрику
            self.val_metric.update(pred_boxes, gt_boxes)

        precision, recall, iou = self.val_metric.summary()
        
        self.precision(precision)
        self.recall(recall)
        self.iou(iou)

        # Логирование метрик
        self.log("test/iou", self.iou, on_step=True, on_epoch=False, prog_bar=True)
        self.log("test/precision", self.precision, on_step=False, on_epoch=True, prog_bar=True)
        self.log("test/recall", self.recall, on_step=False, on_epoch=True, prog_bar=True)

    def configure_optimizers(self):
        """Настройка оптимизаторов и планировщиков."""
        optimizer = self.hparams.optimizer(params=[p for p in self.model.parameters() if p.requires_grad])
        if self.hparams.scheduler is not None:
            scheduler = self.hparams.scheduler(optimizer=optimizer)
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": "val/loss",
                    "interval": "epoch",
                    "frequency": 1,
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
