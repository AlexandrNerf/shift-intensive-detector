import json
import os

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from pathlib import Path

class CSDataset(Dataset):
    def __init__(self, task, batch_size, transform):
        self.batch_size = batch_size

        self.images_path = f'data/{task}/images'
        self.labels_path = f'data/{task}/labels'

        self.file_list = sorted(
            [Path(file).stem for file in os.listdir(self.labels_path)]
        )
        self.transform = transform

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        img_path = os.path.join(self.images_path, self.file_list[idx] + '.jpg')
        lbl_path = os.path.join(self.labels_path, self.file_list[idx] + '.txt')
        
        try:
            # Загрузка изображения
            img = Image.open(img_path)
            img = img.convert('RGB')
            #img = np.array(img)
            #img = torch.from_numpy(img).permute(2,0,1).float() / 255.0
        except Exception as e:
            print(f'Error loading {img_path}: {e}')
            return None, None

        try:
            # Загрузка аннотаций в формате YOLO
            boxes = []
            labels = []
            with open(lbl_path, 'r', encoding='utf-8') as file:
                for line in file:
                    # Чтение данных (класс, центр, ширина, высота)
                    parts = line.strip().split()
                    class_id = int(parts[0])
                    x_center, y_center, width, height = map(float, parts[1:])
                    
                    # Преобразование координат из нормализованных в пиксельные
                    img_width, img_height = img.size
                    x_center = x_center * img_width
                    y_center = y_center * img_height
                    width = width * img_width
                    height = height * img_height
                    
                    # Преобразование в координаты (x_min, y_min, x_max, y_max)
                    x_min = x_center - width / 2
                    y_min = y_center - height / 2
                    x_max = x_center + width / 2
                    y_max = y_center + height / 2
                    
                    boxes.append([x_min, y_min, x_max, y_max])
                    labels.append(class_id)
            if len(boxes) == 0:
                boxes = torch.zeros((0, 4), dtype=torch.float32)
                labels = torch.zeros((0,), dtype=torch.int64)
            else:
                boxes = torch.tensor(boxes, dtype=torch.float32)
                labels = torch.tensor(labels, dtype=torch.int64)
        except Exception as e:
            print(f'Error loading labels from {lbl_path}: {e}')
            return None, None

        return self.transform(img), {'boxes': boxes, 'labels': labels}

    def collate_fn(self, batch):
        batch = [b for b in batch if b[0] is not None]
        images, targets = zip(*batch)
        return list(images), list(targets)


