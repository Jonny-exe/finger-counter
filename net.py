import torch.nn as nn
from torch import tanh, softmax
import torch.nn.functional as F
import torch
import cv2 as cv
import numpy as np


class Net(nn.Module):
    def __init__(self):

        super().__init__()

        self.feature_maps = []
        self.a1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.a2 = nn.Conv2d(16, 16, kernel_size=3, padding=1)
        # self.a3 = nn.Conv2d(16, 32, kernel_size=3, stride=2)
        self.a3 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        # self.a4 = nn.AvgPool2d(3, stride=3)
        self.a4 = nn.AvgPool2d(3, stride=1, padding=1)

        self.b1 = nn.Conv2d(32, 32, kernel_size=3, padding=1)
        self.b2 = nn.Conv2d(32, 32, kernel_size=3, padding=1)
        # self.b3 = nn.Conv2d(32, 64, kernel_size=3, stride=2)
        self.b3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        # self.b4 = nn.AvgPool2d(3, stride=3)
        self.b4 = nn.AvgPool2d(3, stride=1, padding=1)

        self.c1 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.c2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.c3 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.c4 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        # self.c5 = nn.AvgPool2d(2, stride=2)
        self.c5 = nn.AvgPool2d(3, stride=1, padding=1)

        self.d1 = nn.Conv2d(128, 128, kernel_size=3, stride=3)
        self.d2 = nn.Conv2d(128, 128, kernel_size=3, stride=3)
        self.d3 = nn.Conv2d(128, 128, kernel_size=1, stride=1)
        self.d4 = nn.AvgPool2d(5, stride=5)
        # self.d4 = nn.AvgPool2d(2, stride=1)

        self.last = nn.Linear(128, 5)

    def forward(self, x):
        self.feature_maps = []

        self.feature_maps.append(x)
        x = F.relu(self.a1(x))
        x = F.relu(self.a2(x))
        self.feature_maps.append(x)
        x = F.relu(self.a3(x))
        x = F.relu(self.a4(x))

        # 4x4
        self.feature_maps.append(x)
        x = F.relu(self.b1(x))
        x = F.relu(self.b2(x))
        x = F.relu(self.b3(x))
        self.feature_maps.append(x)
        x = F.relu(self.b4(x))

        # 2x2
        self.feature_maps.append(x)
        x = F.relu(self.c1(x))
        x = F.relu(self.c2(x))
        x = F.relu(self.c3(x))
        self.feature_maps.append(x)
        x = F.relu(self.c4(x))
        x = F.relu(self.c5(x))

        # 1x128
        self.feature_maps.append(x)
        x = F.relu(self.d1(x))
        x = F.relu(self.d2(x))
        self.feature_maps.append(x)
        x = F.relu(self.d3(x))
        x = F.relu(self.d4(x))

        x = x.view(-1, 128)
        x = self.last(x)

        # value output
        # return softmax(x, dim=0)
        return tanh(x)

# ([1, 16, 50, 50])
# ([1, 16, 50, 50])
# ([1, 32, 24, 24])
# ([1, 32, 8, 8])
# ([1, 32, 8, 8])
# ([1, 32, 8, 8])
# ([1, 64, 3, 3])
# ([1, 64, 1, 1])
# ([1, 64, 2, 2])
# ([1, 64, 3, 3])
# ([1, 64, 4, 4])
# ([1, 128, 5, 5])
# ([1, 128, 2, 2])
# ([1, 128, 2, 2])


# torch.Size([32, 16, 50, 50])
# torch.Size([32, 16, 50, 50])
# torch.Size([32, 32, 50, 50])
# torch.Size([32, 32, 50, 50])
# torch.Size([32, 32, 50, 50])
# torch.Size([32, 32, 50, 50])
# torch.Size([32, 64, 50, 50])
# torch.Size([32, 64, 50, 50])
# torch.Size([32, 64, 50, 50])
# torch.Size([32, 64, 50, 50])
# torch.Size([32, 64, 50, 50])
# torch.Size([32, 128, 50, 50])
# torch.Size([32, 128, 50, 50])
# torch.Size([32, 128, 16, 16])
# torch.Size([32, 128, 5, 5])
# torch.Size([32, 128, 5, 5])
# torch.Size([32, 128, 5, 5])
# torch.Size([32, 128, 1, 1])
# torch.Size([32, 5])
