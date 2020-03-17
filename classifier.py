import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
from torch import optim
import numpy as np

NUM_CLASSES = 21

class Classifier(nn.Module):
    def __init__(self):
        super(Classifier, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, 5)
        self.conv1_bn = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 128, 3)
        self.conv2_bn = nn.BatchNorm2d(128)
        self.conv3 = nn.Conv2d(128, 128, 3)
        self.conv3_bn = nn.BatchNorm2d(128)
        self.conv4 = nn.Conv2d(128, 1, 3)
        self.conv4_bn = nn.BatchNorm2d(1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(1 * 53 * 53, NUM_CLASSES)
        self.fc1_bn = nn.BatchNorm1d(NUM_CLASSES)

    def forward(self, x):
        x = F.leaky_relu(self.conv1_bn(self.conv1(x)))
        x = self.pool(F.leaky_relu(self.conv2_bn(self.conv2(x))))
        x = F.leaky_relu(self.conv3_bn(self.conv3(x)))
        x = self.pool(F.leaky_relu(self.conv4_bn(self.conv4(x))))
        x = x.view(x.size()[0], 1 * 53 * 53)
        x = self.fc1(x)
        return x


