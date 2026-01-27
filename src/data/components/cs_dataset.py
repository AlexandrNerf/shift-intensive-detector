import os
from pathlib import Path

import torch
from torch.utils.data import Dataset
from PIL import Image

from torchvision.transforms import v2 as T
from torchvision import tv_tensors


class CSDataset(Dataset):
    def __init__(self, task, transform):
        self.images_path = Path(f"data/{task}/images")
        self.labels_path = Path(f"data/{task}/labels")

        self.file_list = sorted(
            [p.stem for p in self.labels_path.glob("*.txt")]
        )

        self.transform = transform

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        img_path = self.images_path / f"{self.file_list[idx]}.jpg"
        lbl_path = self.labels_path / f"{self.file_list[idx]}.txt"

        try:
            image = Image.open(img_path).convert("RGB")
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            return None

        img_w, img_h = image.size

        boxes = []
        labels = []

        try:
            with open(lbl_path, "r", encoding="utf-8") as f:
                for line in f:
                    parts = line.strip().split()
                    class_id = int(parts[0])
                    x_c, y_c, w, h = map(float, parts[1:])

                    # YOLO → pixel XYXY
                    x_c *= img_w
                    y_c *= img_h
                    w *= img_w
                    h *= img_h

                    x_min = x_c - w / 2
                    y_min = y_c - h / 2
                    x_max = x_c + w / 2
                    y_max = y_c + h / 2

                    boxes.append([x_min, y_min, x_max, y_max])
                    labels.append(class_id+1)

        except Exception as e:
            print(f"Error loading labels {lbl_path}: {e}")
            return None

        if len(boxes) == 0:
            boxes = torch.zeros((0, 4), dtype=torch.float32)
            labels = torch.zeros((0,), dtype=torch.int64)
        else:
            boxes = torch.tensor(boxes, dtype=torch.float32)
            labels = torch.tensor(labels, dtype=torch.int64)

        target = {
            "boxes": tv_tensors.BoundingBoxes(
                boxes,
                format="XYXY",
                canvas_size=(img_h, img_w),
            ),
            "labels": labels,
        }

        if self.transform is not None:
            image, target = self.transform(image, target)

        return image, target

    @staticmethod
    def collate_fn(batch):
        batch = [b for b in batch if b is not None]
        images, targets = zip(*batch)
        return list(images), list(targets)
