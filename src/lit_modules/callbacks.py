import os
import csv
import torch
import logging
from lightning.pytorch import LightningModule, Trainer
from lightning.pytorch.callbacks import Callback, ModelCheckpoint
from hydra.utils import get_original_cwd

log = logging.getLogger(__name__)

class JITModelCheckpoint(ModelCheckpoint):
    def __init__(self, *args, **kwargs):
        """
        Коллбек для сохранения модели как в стандартном формате, так и в JIT формате.
        """
        super().__init__(*args, **kwargs)  # Инициализация стандартного ModelCheckpoint

    def on_save_checkpoint(self, trainer, pl_module, checkpoint):
        """
        Этот метод вызывается сразу после того, как checkpoint сохраняется.
        """
        # Сначала выполняем стандартное сохранение через ModelCheckpoint
        result = super().on_save_checkpoint(trainer, pl_module, checkpoint)

        # Получаем модель из LightningModule
        model = pl_module.model

        if not os.path.exists(self.dirpath):
            os.makedirs(self.dirpath)

        # Скомпилировать модель с помощью torch.jit.script
        try:
            jit_model = torch.jit.script(model)  # Используем `script` для модели
        except Exception as e:
            log.info(f"Ошибка при компиляции модели в JIT: {e}")
            return result

        # Сохраняем JIT модель
        model_path = os.path.join(self.dirpath, f"{self.filename.format(epoch=trainer.current_epoch)}_jit.pt")

        # Сохраняем JIT модель
        torch.jit.save(jit_model, model_path)
        
        return result


class HardNegativeLogger(Callback):
    def __init__(self, output_dir, csv_name="hnm.csv"):
        super().__init__()
        self.train_loader = None
        self.eval_loader = None
        self.csv_path_val = os.path.join(output_dir, 'val_' + csv_name)
        self.csv_path_train = os.path.join(output_dir, 'train_' + csv_name)
        self.results = []

    @torch.no_grad()
    def on_fit_end(self, trainer: Trainer, pl_module: LightningModule):
        if trainer.val_dataloaders is not None and len(trainer.val_dataloaders) > 0:
            self.eval_loader = trainer.val_dataloaders
        if trainer.train_dataloader is not None and len(trainer.train_dataloader) > 0:
            self.train_loader = trainer.train_dataloader
        if self.eval_loader is None and self.train_loader is None:
            log.info("⚠️ HardNegativeLogger: no train_loader and no eval_loader provided. Please check you enabled")
            return

        pl_module.eval()
        device = pl_module.device
        log.info("🔍 Collecting hard negatives...")

        for batch_idx, batch in enumerate(self.eval_loader):
            images, labels, paths = batch

            for sample_idx, (img, gt, path) in enumerate(zip(images, labels, paths)):
                img = img.unsqueeze(0).to(device)  # добавляем batch dim

                output = pl_module((img, [gt], path))
                
                preds = output.get("preds", [()])  # берем список предсказаний для 1-го примера]
                loss_val = output.get("loss", None)

                for pred_text, conf in preds:
                    entry = {
                        "batch_idx": batch_idx,
                        "sample_idx": sample_idx,
                        "path": path,
                        "pred_text": pred_text,
                        "confidence": float(conf),
                        "loss": loss_val,
                        "gt": gt
                    }
                    self.results.append(entry)
    
        # сохраняем csv
        with open(self.csv_path_val, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=self.results[0].keys())
            writer.writeheader()
            writer.writerows(self.results)

        self.results = []
        # разворачиваем preds: list[list[tuple[text, conf]]]
        for batch_idx, batch in enumerate(self.train_loader):
            images, labels, paths = batch

            for sample_idx, (img, gt, path) in enumerate(zip(images, labels, paths)):
                img = img.unsqueeze(0).to(device)  # добавляем batch dim

                output = pl_module((img, [gt], path))
                
                preds = output.get("preds", [()])  # берем список предсказаний для 1-го примера]
                loss_val = output.get("loss", None)

                for pred_text, conf in preds:
                    entry = {
                        "batch_idx": batch_idx,
                        "sample_idx": sample_idx,
                        "path": path,
                        "pred_text": pred_text,
                        "confidence": float(conf),
                        "loss": loss_val,
                        "gt": gt
                    }
                    self.results.append(entry)
        with open(self.csv_path_train, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=self.results[0].keys())
            writer.writeheader()
            writer.writerows(self.results)

        log.info(f"✅ Hard negatives saved to: {self.csv_path_val}, {self.csv_path_train}")
