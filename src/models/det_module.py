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
from collections import defaultdict

class BaseDetectionModel(LightningModule):
    def __init__(
        self,
        net: str,
        pretrained: bool,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler,
        nms_thresh: float,
        score_thresh: float,
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
        # вы также можете настроить пороги для nms и score
        # self.model.roi_heads.score_thresh = self.hparams.score_thresh
        # self.model.roi_heads.nms_thresh = self.hparams.nms_thresh

        # Если нужно, можно настроить кастомную модель (например, на основе других слоев)
        # if self.hparams.net == 'your_model':
        #   self.model = Model(*args)    

        # Метрики (в классе TorchLocalizationConfussion мы можем настроить порог подсчёта iou)
        self.map_metric = MeanAveragePrecision(box_format="xyxy",
            iou_type="bbox",
            iou_thresholds=[0.5, 0.75],
            class_metrics=True
        )
        self.class_names = {
            1: "c-ter",
            2: "ter",
        }


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

        return total_loss

    def on_train_epoch_end(self) -> None:
        """Шаг, который выполняется в конце каждой эпохи."""
        
    

    def validation_step(self, batch, batch_idx):
        images, targets = batch
        preds = self.model(images)

        # preds / targets — списки dict'ов (torchvision format!)
        self.map_metric.update(preds, targets)

        if batch_idx == 5:
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
                    "interval": "step",
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


    def log_images(
        self,
        images,
        targets,
        preds,
        tag: str,
        max_classes: int = 2,
        samples_per_class: int = 2,
    ):
        """
        Логирует изображения:
        - до max_classes разных классов
        - по samples_per_class примеров на класс
        """

        # --------- собираем class_id -> indices ----------
        class_to_indices = defaultdict(list)

        for i, tgt in enumerate(targets):
            labels = tgt["labels"].tolist()
            for cls in set(labels):
                if cls != 0:  # background игнорим
                    class_to_indices[cls].append(i)

        # --------- выбираем классы ----------
        selected_classes = list(class_to_indices.keys())[:max_classes]

        global_step = self.global_step

        for cls_id in selected_classes:
            class_name = self.class_names.get(cls_id, f"class_{cls_id}")

            indices = class_to_indices[cls_id][:samples_per_class]

            for j, idx in enumerate(indices):
                img = images[idx].detach().cpu()

                # ---------- GT ----------
                gt_boxes = targets[idx]["boxes"].cpu()
                gt_labels = targets[idx]["labels"].cpu()

                gt_mask = gt_labels == cls_id
                gt_boxes = gt_boxes[gt_mask]

                gt_text = [class_name] * len(gt_boxes)

                img_gt = vutils.draw_bounding_boxes(
                    (img * 255).byte(),
                    gt_boxes,
                    labels=gt_text,
                    colors="green",
                    width=2,
                    font_size=14,
                )

                # ---------- PRED ----------
                pred_boxes = preds[idx]["boxes"].detach().cpu()
                pred_labels = preds[idx]["labels"].detach().cpu()

                pred_mask = pred_labels == cls_id
                pred_boxes = pred_boxes[pred_mask]

                pred_text = [class_name] * len(pred_boxes)

                img_pred = vutils.draw_bounding_boxes(
                    (img * 255).byte(),
                    pred_boxes,
                    labels=pred_text,
                    colors="red",
                    width=2,
                    font_size=14,
                )

                # ---------- TB ----------
                self.logger.experiment.add_image(
                    f"{tag}/{class_name}/{j}/gt",
                    img_gt,
                    global_step,
                )
                self.logger.experiment.add_image(
                    f"{tag}/{class_name}/{j}/pred",
                    img_pred,
                    global_step,
                )
