# Copyright (C) 2021-2024, Mindee.

# Эта программа лицензирована по Apache License 2.0.
# Полные условия лицензии см. в LICENSE или на <https://opensource.org/licenses/Apache-2.0>.

from typing import Dict, Optional
import torch
from torchvision.ops import box_iou as torch_box_iou
from scipy.optimize import linear_sum_assignment

def box_iou(boxes_1: torch.Tensor, boxes_2: torch.Tensor) -> torch.Tensor:
    """Вычисляет IoU между двумя наборами bounding box-ов.

    Args:
        boxes_1: gt боксы размера (N, 4) и формата (xmin, ymin, xmax, ymax)
        boxes_2: pred боксы размера (M, 4) и формата (xmin, ymin, xmax, ymax)

    Returns:
        IoU-матрица (N, M)
    """
    return torch_box_iou(boxes_1, boxes_2)

class TorchLocalizationConfusion:

    def __init__(
        self,
        iou_thresh: float = 0.5,
        use_polygons: bool = False,
    ) -> None:
        """
        Класс кастомных метрик, принимающий тензоры и вычисляющий всё на GPU
        Args:
            iou_thresh (float): порог IoU для отсчёта
            use_polygons (bool): Использование полигонов
        """
        self.iou_thresh = iou_thresh
        self.use_polygons = use_polygons
        self.reset()

    def update(self, gts: torch.Tensor, preds: torch.Tensor) -> None:
        if preds.shape[0] > 0 and gts.shape[0] > 0:
            iou_tensor = box_iou(gts, preds)
            self.tot_iou += float(iou_tensor.max(axis=0).values.sum())

            # назначаем пары
            gt_indices, pred_indices = linear_sum_assignment(-iou_tensor.cpu().numpy())
            self.matches += int((iou_tensor[gt_indices, pred_indices] >= self.iou_thresh).sum())

        # обновляем счетчики
        self.num_gts += gts.shape[0]
        self.num_preds += preds.shape[0]

    def summary(self) -> Dict[str, Optional[float]]:
        """Вычисляет агрегированные метрики.

        Returns
        -------
            словарь со значениями recall, precision и meanIoU
        """
        # полнота
        recall = self.matches / self.num_gts if self.num_gts > 0 else 0.0

        # точность
        precision = self.matches / self.num_preds if self.num_preds > 0 else 0.0

        # средний IoU
        mean_iou = round(self.tot_iou / self.num_preds, 2) if self.num_preds > 0 else 0.0

        results = {
            "recall": recall,
            "precision": precision,
            "mean_iou": mean_iou,
            }

        return results

    def reset(self) -> None:
        self.num_gts = 0
        self.num_preds = 0
        self.matches = 0
        self.tot_iou = 0.0
