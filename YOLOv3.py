import torch
from torch import nn
# Определение сверточного слоя с BatchNorm и LeakyReLU
class ConvBNLeaky(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super(ConvBNLeaky, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.leaky_relu = nn.LeakyReLU(0.1)

    def forward(self, x):
        return self.leaky_relu(self.bn(self.conv(x)))

# Определение остаточного блока
class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = ConvBNLeaky(channels, channels // 2, 1, 1, 0)
        self.conv2 = ConvBNLeaky(channels // 2, channels, 3, 1, 1)

    def forward(self, x):
        return x + self.conv2(self.conv1(x))

# Основная архитектура YOLOv3
class YOLOv3(nn.Module):
    def __init__(self, num_classes):
        super(YOLOv3, self).__init__()
        self.num_classes = num_classes
        self.layer1 = ConvBNLeaky(3, 32, 3, 1, 1)
        self.layer2 = ConvBNLeaky(32, 64, 3, 2, 1)
        self.residual_block1 = ResidualBlock(64)

        # Дополнительные слои
        self.layer3 = ConvBNLeaky(64, 128, 3, 2, 1)
        self.residual_block2 = ResidualBlock(128)
        self.layer4 = ConvBNLeaky(128, 256, 3, 2, 1)
        self.residual_block3 = ResidualBlock(256)
        self.layer5 = ConvBNLeaky(256, 512, 3, 2, 1)
        self.residual_block4 = ResidualBlock(512)
        self.layer6 = ConvBNLeaky(512, 1024, 3, 2, 1)
        self.residual_block5 = ResidualBlock(1024)

        # Слои обнаружения
        self.detection1 = nn.Conv2d(1024, 5 * num_classes, 1)
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.final_conv = nn.Conv2d(1024, 5, 1)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.residual_block1(x)
        x = self.layer3(x)
        x = self.residual_block2(x)
        x = self.layer4(x)
        x = self.residual_block3(x)
        x = self.layer5(x)
        x = self.residual_block4(x)
        x = self.layer6(x)
        x = self.residual_block5(x)

        x = self.global_avg_pool(x)  # Применение глобального среднего пулинга
        detection = self.final_conv(x)  # Применение конечного сверточного слоя
        detection = detection.view(x.size(0), -1)  # Изменение формы тензора
        return detection

# Создание модели
model = YOLOv3(num_classes=20)  # Пример с 20 классами
