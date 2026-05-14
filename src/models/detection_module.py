from collections import defaultdict

import torch
from lightning import LightningModule
from omegaconf import DictConfig
import hydra
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

        self.optimizer_partial = hydra.utils.instantiate(optimizer)
        self.scheduler_partial = hydra.utils.instantiate(scheduler)
        self.loss_weights = hydra.utils.instantiate(loss_weights)

        self.val_metric = TorchLocalizationConfusion(iou_thresh=iou_thresh)

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
        self._update_metric(preds, targets)

        if batch_idx == 5:
            self.log_images(images, targets, preds, "val")


    def on_validation_epoch_end(self):
        metrics = self.val_metric.summary()
        self.val_metric.reset()

        self.log_dict(
            {
                "val/mAP": metrics["precision"],           # mAP@[.5:.95]
                "val/mAP50": metrics["precision"],      # mAP@0.5
                #"val/mAP75": metrics["map_75"],      # mAP@0.75
                "val/recall": metrics["recall"],    # recall
                "val/mean_iou": metrics["mean_iou"],      # mean IoU    
            },
            on_epoch=True,
            sync_dist=True,
            prog_bar=True,
        )

    def test_step(self, batch, batch_idx):
        images, targets = batch
        preds = self.model(images)
        self._update_metric(preds, targets)

    def _update_metric(self, preds, targets):
        for pred, target in zip(preds, targets):
            self.val_metric.update(target["boxes"], pred["boxes"])

    def on_test_epoch_end(self):
        metrics = self.val_metric.summary()
        self.val_metric.reset()

        self.log_dict(
            {
                "val/mAP": metrics["precision"],           # mAP@[.5:.95]
                "val/mAP50": metrics["precision"],      # mAP@0.5
                #"val/mAP75": metrics["map_75"],      # mAP@0.75
                "val/recall": metrics["recall"],    # recall
                "val/mean_iou": metrics["mean_iou"],      # mean IoU    
            }
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
