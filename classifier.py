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
        
        self.conv1 = nn.Conv2d(3, 64, 3, padding=1)
        self.conv1_bn = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 64, 3, padding=1)
        self.conv2_bn = nn.BatchNorm2d(64)
        
        self.conv3 = nn.Conv2d(64, 64, 3, padding=1)
        self.conv3_bn = nn.BatchNorm2d(64)
        self.conv4 = nn.Conv2d(64, 64, 3, padding=1)
        self.conv4_bn = nn.BatchNorm2d(64)
        
        self.conv5 = nn.Conv2d(64, 128, 3, padding=1)
        self.conv5_bn = nn.BatchNorm2d(128)
        self.conv6 = nn.Conv2d(128, 128, 3, padding=1)
        self.conv6_bn = nn.BatchNorm2d(128)
        self.conv7 = nn.Conv2d(128, 128, 3, padding=1)
        self.conv7_bn = nn.BatchNorm2d(128)
        
        self.conv8 = nn.Conv2d(128, 256, 3, padding=1)
        self.conv8_bn = nn.BatchNorm2d(256)
        self.conv9 = nn.Conv2d(256, 256, 3, padding=1)
        self.conv9_bn = nn.BatchNorm2d(256)
        self.conv10 = nn.Conv2d(256, 1, 1)
        self.conv10_bn = nn.BatchNorm2d(1)
        
        self.pool = nn.MaxPool2d(2, 2)
        
        self.fc1 = nn.Linear(1 * 227 * 227, NUM_CLASSES)

    def forward(self, x):
        # 64
        x = self.conv1_bn(F.relu(self.conv1_bn(self.conv1(x))))
        residual = x
        x = self.conv2_bn(F.relu(self.conv2_bn(self.conv2(x))))
        x += residual
        x = self.conv3_bn(F.relu(self.conv3_bn(self.conv3(x))))
        residual = x
        x = self.conv4_bn(F.relu(self.conv4_bn(self.conv4(x))))
        x += residual
        
        # 128
        x = self.conv5_bn(F.relu(self.conv5_bn(self.conv5(x))))
        residual = x
        x = self.conv6_bn(F.relu(self.conv6_bn(self.conv6(x))))
        x += residual
        x = self.conv7_bn(F.relu(self.conv7_bn(self.conv7(x))))
        
        # 256
        x = self.conv8_bn(F.relu(self.conv8_bn(self.conv8(x))))
        residual = x
        x = self.conv9_bn(F.relu(self.conv9_bn(self.conv9(x))))
        x += residual
        x = self.conv10_bn(F.relu(self.conv10_bn(self.conv10(x))))
        
        print(x.size())
        x = x.view(x.size()[0], 1 * 227 * 227)
        x = self.fc1(x)
        return x

