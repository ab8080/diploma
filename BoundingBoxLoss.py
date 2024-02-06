import torch
from torch import nn

class BoundingBoxLoss(nn.Module):
    def __init__(self):
        super(BoundingBoxLoss, self).__init__()
        self.mse_loss = nn.MSELoss(reduction='sum')

    def forward(self, predictions, targets):

        # Вычисляем потери только для координат ограничивающих рамок
        loss = self.mse_loss(predictions, targets)
        return loss
