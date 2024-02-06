from torch.utils.data import Dataset
import os
from PIL import Image
import torch

class YourDataset(Dataset):
    def __init__(self, img_dir, ann_dir, transform=None):
        self.img_dir = img_dir
        self.ann_dir = ann_dir
        self.transform = transform

        self.img_names = [img_name for img_name in os.listdir(img_dir)]

    def __len__(self):
        return len(self.img_names)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_names[idx])
        ann_path = os.path.join(self.ann_dir, self.img_names[idx].replace('.jpg', '.txt'))

        # Загрузка изображения
        image = Image.open(img_path).convert('RGB')

        # Загрузка аннотаций
        annotations = self.load_annotations(ann_path)

        if self.transform:
            image = self.transform(image)

        return image, annotations

    @staticmethod
    def load_annotations(ann_path):
        # Функция для загрузки и обработки аннотаций из файла
        boxes = []
        with open(ann_path, 'r') as file:
            for line in file.readlines():
                class_id, x_center, y_center, width, height = map(float, line.split())
                boxes.append([class_id, x_center, y_center, width, height])
        return torch.tensor(boxes)
