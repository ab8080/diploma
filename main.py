import os
import torch
from torchvision import transforms
from torch.utils.data import DataLoader
import torch.nn.functional as F

from YOLOv3 import YOLOv3
from BoundingBoxLoss import BoundingBoxLoss
from YourDataset import YourDataset

img_dir = '/home/aleksandr/dataset/clips/all_images'
ann_dir = '/home/aleksandr/dataset/clips/all_annotations'

test_img_dir = '/home/aleksandr/dataset/clips/test_images'
test_ann_dir = '/home/aleksandr/dataset/clips/test_annotations'


print("Содержимое ann_dir:", os.listdir(ann_dir)[:10])

# Создание модели
model = YOLOv3(num_classes=20)

# Создание экземпляра функции потерь и оптимизатора
criterion = BoundingBoxLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

transform = transforms.Compose([
    transforms.Resize((640, 640)),
    transforms.ToTensor()
])

train_dataset = YourDataset(img_dir, ann_dir, transform=transform)
test_dataset = YourDataset(test_img_dir, test_ann_dir, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=True)

# Цикл обучения
num_epochs = 10  # Примерное количество эпох
for epoch in range(num_epochs):
    model.train()
    train_mse = 0
    for images, targets in train_loader:
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, targets)
        mse = F.mse_loss(outputs, targets).item()  # Вычисление MSE на обучающих данных
        train_mse += mse
        loss.backward()
        optimizer.step()
    # После цикла обучения для каждой эпохи
    average_train_mse = train_mse / len(train_loader)
    print(f"Epoch {epoch + 1}/{num_epochs}, Train MSE: {average_train_mse}, ...")

    model.eval()
    val_loss = 0
    mse_loss = 0
    with torch.no_grad():
        for images, targets in test_loader:
            outputs = model(images)
            val_loss += criterion(outputs, targets).item()  # Суммируем потери на валидации
            mse_loss += F.mse_loss(outputs, targets).item()  # Суммируем MSE

