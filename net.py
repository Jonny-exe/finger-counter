import torch.nn as nn
from torch import tanh, softmax
import torch.nn.functional as F
import torch
import cv2 as cv
import numpy as np

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.a1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.a2 = nn.Conv2d(16, 16, kernel_size=3, padding=1)
        self.a3 = nn.Conv2d(16, 16, kernel_size=3, stride=2)
        self.a4 = nn.Conv2d(16, 32, kernel_size=3, stride=2)
        # self.drop1 = nn.Dropout(p=0.2)

        self.b1 = nn.Conv2d(32, 32, kernel_size=3, padding=1)
        self.b2 = nn.Conv2d(32, 32, kernel_size=3, padding=1)
        self.b3 = nn.Conv2d(32, 64, kernel_size=3, stride=2)
        # self.drop2 = nn.Dropout(p=0.2)

        self.c1 = nn.Conv2d(64, 64, kernel_size=2, padding=1)
        self.c2 = nn.Conv2d(64, 64, kernel_size=2, padding=1)
        self.c3 = nn.Conv2d(64, 128, kernel_size=2, stride=2)
        # self.drop3 = nn.Dropout(p=0.2)

        self.d1 = nn.Conv2d(128, 128, kernel_size=1)
        self.d2 = nn.Conv2d(128, 128, kernel_size=1)
        self.d3 = nn.Conv2d(128, 128, kernel_size=1)
        # self.drop4 = nn.Dropout(p=0.2)
        self.pool = nn.AvgPool2d(3, stride=2)

        self.last = nn.Linear(128, 5)

    def forward(self, x):
        x = F.relu(self.a1(x))
        x = F.relu(self.a2(x))
        x = F.relu(self.a3(x))
        # x = self.drop1(F.relu(self.a4(x)))
        x = F.dropout(F.relu(self.a4(x)), p=0.2)

        x = F.relu(self.b1(x))
        x = F.relu(self.b2(x))
        # x = self.drop2(F.relu(self.b3(x)))
        x = F.dropout(F.relu(self.b3(x)), p=0.2)

        x = F.relu(self.c1(x))
        x = F.relu(self.c2(x))
        # x = self.drop3(F.relu(self.c3(x)))
        x = F.dropout(F.relu(self.c3(x)), p=0.2)

        x = F.relu(self.d1(x))
        x = F.relu(self.d2(x))
        # x = self.drop4(F.relu(self.d3(x)))
        x = F.dropout(F.relu(self.d3(x)), p=0.2)
        x = F.relu(self.pool(x))

        x = x.view(-1, 128)

        x = self.last(x)

        # value output
        # return softmax(x, dim=1)
        return x