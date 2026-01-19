from typing import Any, Dict, Tuple, Optional, List

import torch
from torch import nn
from lightning import LightningModule
from torchmetrics import MaxMetric, MeanMetrics
import timm

from src.utils.metrics.metrics_fast import TorchLocalizationConfusion


class BaseDetectionModel(LightningModule):
    def __init__(
        self,
        net: str,
        pretrained: bool,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler,
        compile: bool = False,
    ) -> None:
        super().__init__()

        self.save_hyperparameters(logger=False)

        self.model = timm.create_model(
            "fasterrcnn_mobilenet_v3_large_fpn",
            pretrained=pretrained,
        )

        self.val_metric = TorchLocalizationConfusion(iou_thresh=0.5)

        self.precision = MeanMetric()
        self.recall = MeanMetric()
        self.iou = MeanMetric()

        self.precision_best = MaxMetric()
        self.recall_best = MaxMetric()
        self.iou_best = MaxMetric()

        self.train_loss = MeanMetric()
        self.val_loss = MeanMetric()
        self.test_loss = MeanMetric()

    def forward(self, images, targets=None):
        return self.model(images, targets)

    def model_step(self, batch):
        images, targets = batch
        loss_dict = self.model(images, targets)
        loss = sum(loss_dict.values())
        return loss

    def training_step(self, batch, batch_idx):
        loss = self.model_step(batch)

        self.train_loss(loss)
        self.log(
            "train/loss",
            self.train_loss,
            on_step=True,
            on_epoch=False,
            prog_bar=True,
        )
        return loss

    def validation_step(self, batch, batch_idx):
        images, targets = batch

        preds = self.model(images)

        for pred, tgt in zip(preds, targets):
            pred_boxes = pred["boxes"]
            gt_boxes = tgt["boxes"]

            # (TP=0, FN=0) → игнорируем
            if pred_boxes.numel() == 0 and gt_boxes.numel() == 0:
                continue

            # (FP)
            if gt_boxes.numel() == 0:
                self.val_metric.update(
                    pred_boxes,
                    gt_boxes.new_zeros((0, 4))
                )
                continue

            # (FN)
            if pred_boxes.numel() == 0:
                self.val_metric.update(
                    pred_boxes.new_zeros((0, 4)),
                    gt_boxes
                )
                continue

            self.val_metric.update(pred_boxes, gt_boxes)

        precision, recall, iou = self.val_metric.summary()

        self.precision(precision)
        self.recall(recall)
        self.iou(iou)

        self.log("val/precision", self.precision, prog_bar=True)
        self.log("val/recall", self.recall, prog_bar=True)
        self.log("val/iou", self.iou, prog_bar=True)

    def on_validation_epoch_end(self):
        self.precision_best.update(self.precision.compute())
        self.recall_best.update(self.recall.compute())
        self.iou_best.update(self.iou.compute())

        self.log("val/best_precision", self.precision_best.compute(), prog_bar=True)
        self.log("val/best_recall", self.recall_best.compute(), prog_bar=True)
        self.log("val/best_iou", self.iou_best.compute(), prog_bar=True)

        self.val_metric.reset()

    def test_step(self, batch, batch_idx):
        images, targets = batch
        preds = self.model(images)

        for pred, tgt in zip(preds, targets):
            self.val_metric.update(pred["boxes"], tgt["boxes"])

        precision, recall, iou = self.val_metric.summary()

        self.log("test/precision", precision)
        self.log("test/recall", recall)
        self.log("test/iou", iou)

    def configure_optimizers(self):
        optimizer = self.hparams.optimizer(
            params=[p for p in self.model.parameters() if p.requires_grad]
        )

        if self.hparams.scheduler is None:
            return optimizer

        scheduler = self.hparams.scheduler(optimizer)

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",
                "monitor": "val/iou",
            },
        }

    def on_after_backward(self):
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 5.0)
