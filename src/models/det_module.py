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
import torchvision.utils as vutils


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
            pretrained=self.hparams.pretrained
        )

        # количество входных фич у классификатора
        in_features = self.model.roi_heads.box_predictor.cls_score.in_features

        # заменяем head под свои классы
        self.model.roi_heads.box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(
            in_features,
            num_classes=3 # Модели torchvision учитывают фон как класс!
        )

        # Если нужно, можно настроить кастомную модель (например, на основе других слоев)
        # if self.hparams.net == 'your_model':
        #   self.model = Model(*args)    

        # Метрики (в классе TorchLocalizationConfussion мы можем настроить порог подсчёта iou)
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
        total_loss = sum(loss_dict.values())
        return total_loss, loss_dict

    def training_step(self, batch, batch_idx: int):
        """Один шаг обучения."""
        total_loss, loss_dict = self.model_step(batch)

        # общий лосс
        self.log(
            "train/loss",
            total_loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
        )

        # отдельные лоссы (опционально)
        for k, v in loss_dict.items():
            self.log(
                f"train/{k}",
                v,
                on_step=True,
                on_epoch=True,
                sync_dist=True,
            )

        return total_loss

    def on_train_epoch_end(self) -> None:
        """Шаг, который выполняется в конце каждой эпохи."""
        opt = self.optimizers()
        lr = opt.param_groups[0]["lr"]
        
        # логируем learning_rate
        self.log(
            "lr",
            lr,
            on_epoch=True,
            prog_bar=False,
            sync_dist=True,
        )

    def validation_step(self, batch, batch_idx):
        images, targets = batch
        preds = self.model(images)

        # preds / targets — списки dict'ов (torchvision format!)
        self.map_metric.update(preds, targets)

        if batch_idx == 0:
            self.log_images(images, targets, preds, "val")


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
            on_epoch=True,
            sync_dist=True,
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


    def log_images(self, images, targets, preds, tag):
        img = images[0].detach().cpu()
        boxes_gt = targets[0]["boxes"].cpu()
        boxes_pred = preds[0]["boxes"].detach().cpu()

        img_gt = vutils.draw_bounding_boxes(
            (img * 255).byte(),
            boxes_gt,
            colors="green",
            width=2,
        )

        img_pred = vutils.draw_bounding_boxes(
            (img * 255).byte(),
            boxes_pred,
            colors="red",
            width=2,
        )

        self.logger.experiment.add_image(
            f"{tag}/gt",
            img_gt,
            self.current_epoch,
        )
        self.logger.experiment.add_image(
            f"{tag}/pred",
            img_pred,
            self.current_epoch,
        )
