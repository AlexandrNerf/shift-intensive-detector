from collections import defaultdict

import torch
from lightning import LightningModule
from omegaconf import DictConfig
import hydra
from torchmetrics.detection.mean_ap import MeanAveragePrecision
from torchvision import utils as vutils

from src.utils.metrics.metrics_fast import TorchLocalizationConfusion


class DetectionLitModel(LightningModule):
    def __init__(
        self,
        components: DictConfig,
        optimizer: DictConfig,
        scheduler: DictConfig,
        loss_weights: DictConfig,
        iou_thresh: float = 0.5,
        compile: bool = False,
    ) -> None:
        super().__init__()

        self.save_hyperparameters(logger=False)

        self.model = hydra.utils.instantiate(components)
        if compile:
            self.model = torch.compile(self.model)

        self.optimizer_partial = hydra.utils.instantiate(optimizer)
        self.scheduler_partial = hydra.utils.instantiate(scheduler)
        self.loss_weights = hydra.utils.instantiate(loss_weights)

        self.val_loc_metric = TorchLocalizationConfusion(iou_thresh=iou_thresh)
        self.test_loc_metric = TorchLocalizationConfusion(iou_thresh=iou_thresh)
        self.val_map = MeanAveragePrecision(box_format="xyxy", iou_type="bbox")
        self.test_map = MeanAveragePrecision(box_format="xyxy", iou_type="bbox")

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
        total_loss = self.loss_weights(loss_dict)
        return total_loss, loss_dict

    def training_step(self, batch, batch_idx: int):
        """Один шаг обучения."""
        images, _ = batch
        batch_size = len(images)
        total_loss, loss_dict = self.model_step(batch)

        # общий лосс
        self.log(
            "train/loss",
            total_loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
            batch_size=batch_size,
        )

        # отдельные лоссы (опционально)
        for k, v in loss_dict.items():
            self.log(
                f"train/{k}",
                v,
                on_step=True,
                on_epoch=True,
                sync_dist=True,
                batch_size=batch_size,
            )
        opt = self.optimizers()
        lr = opt.param_groups[0]["lr"]

        # логируем скорость обучения
        self.log(
            "lr",
            lr,
            on_epoch=True,
            prog_bar=False,
            sync_dist=True,
            batch_size=batch_size,
        )

        return total_loss

    def on_train_epoch_end(self) -> None:
        """Шаг, который выполняется в конце каждой эпохи."""
        

    def validation_step(self, batch, batch_idx):
        images, targets = batch
        preds = self.model(images)

        # preds и targets - списки словарей в формате torchvision
        self._update_localization_metric(self.val_loc_metric, preds, targets)
        self.val_map.update(preds, targets)

        if batch_idx == 5:
            self.log_images(images, targets, preds, "val")


    def on_validation_epoch_end(self):
        loc_metrics = self.val_loc_metric.summary()
        map_metrics = self.val_map.compute()
        self.val_loc_metric.reset()
        self.val_map.reset()
        batch_size = self.trainer.datamodule.batch_size_per_device

        self.log_dict(
            {
                "val/loc_precision": loc_metrics["precision"],
                "val/loc_recall": loc_metrics["recall"],
                "val/loc_mean_iou": loc_metrics["mean_iou"],
                "val/mAP": map_metrics["map"],
                "val/mAP50": map_metrics["map_50"],
                "val/mAP75": map_metrics["map_75"],
                "val/mAR100": map_metrics["mar_100"],
            },
            on_epoch=True,
            sync_dist=True,
            prog_bar=True,
            batch_size=batch_size,
        )

    def test_step(self, batch, batch_idx):
        images, targets = batch
        preds = self.model(images)
        self._update_localization_metric(self.test_loc_metric, preds, targets)
        self.test_map.update(preds, targets)

    def _update_localization_metric(self, metric, preds, targets):
        for pred, target in zip(preds, targets):
            metric.update(target["boxes"], pred["boxes"])

    def on_test_epoch_end(self):
        loc_metrics = self.test_loc_metric.summary()
        map_metrics = self.test_map.compute()
        self.test_loc_metric.reset()
        self.test_map.reset()
        batch_size = self.trainer.datamodule.batch_size_per_device

        self.log_dict(
            {
                "test/loc_precision": loc_metrics["precision"],
                "test/loc_recall": loc_metrics["recall"],
                "test/loc_mean_iou": loc_metrics["mean_iou"],
                "test/mAP": map_metrics["map"],
                "test/mAP50": map_metrics["map_50"],
                "test/mAP75": map_metrics["map_75"],
                "test/mAR100": map_metrics["mar_100"],
            },
            batch_size=batch_size,
        )
    
    def configure_optimizers(self):
        """Настройка оптимизаторов и планировщиков."""
        optimizer = self.optimizer_partial(params=[p for p in self.model.parameters() if p.requires_grad])
        if self.scheduler_partial is not None:
            scheduler = self.scheduler_partial(optimizer=optimizer)
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "interval": "step",
                    "monitor": "val/mAP50", # здесь можно настроить, по какой метрике будет мониторинг
                },
            }
        return {"optimizer": optimizer}

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
        experiment = getattr(self.logger, "experiment", None)
        if experiment is None or not hasattr(experiment, "add_image"):
            return

        # --------- собираем id класса -> индексы ----------
        class_to_indices = defaultdict(list)

        for i, tgt in enumerate(targets):
            labels = tgt["labels"].tolist()
            for cls in set(labels):
                if cls != 0:  # игнорируем фон
                    class_to_indices[cls].append(i)

        # --------- выбираем классы ----------
        selected_classes = list(class_to_indices.keys())[:max_classes]

        global_step = self.global_step

        for cls_id in selected_classes:
            class_name = self.class_names.get(cls_id, f"class_{cls_id}")

            indices = class_to_indices[cls_id][:samples_per_class]

            for j, idx in enumerate(indices):
                img = images[idx].detach().cpu()

                # ---------- разметка ----------
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

                # ---------- предсказания ----------
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

                # ---------- TensorBoard ----------
                experiment.add_image(
                    f"{tag}/{class_name}/{j}/gt",
                    img_gt,
                    global_step,
                )
                experiment.add_image(
                    f"{tag}/{class_name}/{j}/pred",
                    img_pred,
                    global_step,
                )
