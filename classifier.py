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
        
        self.fc1 = nn.Linear(1 * 200 * 200, NUM_CLASSES)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, train=False):
        p = 0.5
        # 64
        x = F.dropout2d(self.conv1_bn(F.relu(self.conv1_bn(self.conv1(x)))), p, train)
        residual = x
        x = F.dropout2d(self.conv2_bn(F.relu(self.conv2_bn(self.conv2(x)))), p, train)
        x += residual
        x = F.dropout2d(self.conv3_bn(F.relu(self.conv3_bn(self.conv3(x)))), p, train)
        residual = x
        x = F.dropout2d(self.conv4_bn(F.relu(self.conv4_bn(self.conv4(x)))), p, train)
        x += residual
        
        # 128
        x = F.dropout2d(self.conv5_bn(F.relu(self.conv5_bn(self.conv5(x)))), p, train)
        residual = x
        x = F.dropout2d(self.conv6_bn(F.relu(self.conv6_bn(self.conv6(x)))), p, train)
        x += residual
        x = F.dropout2d(self.conv7_bn(F.relu(self.conv7_bn(self.conv7(x)))), p, train)
        
        # 256
        x = F.dropout2d(self.conv8_bn(F.relu(self.conv8_bn(self.conv8(x)))), p, train)
        residual = x
        x = F.dropout2d(self.conv9_bn(F.relu(self.conv9_bn(self.conv9(x)))), p, train)
        x += residual
        x = F.dropout2d(self.conv10_bn(F.relu(self.conv10_bn(self.conv10(x)))), p, train)
        
        x = x.view(x.size()[0], 1 * 200 * 200)
        x = self.sigmoid(self.fc1(x))
        return x

