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
        
        self.conv2 = nn.Conv2d(64, 128, 3, padding=1)
        self.conv2_bn = nn.BatchNorm2d(128)
        
        self.conv3 = nn.Conv2d(128, 152, 3, padding=1)
        self.conv3_bn = nn.BatchNorm2d(152)
        
        self.conv4 = nn.Conv2d(152, 192, 3)
        self.conv4_bn = nn.BatchNorm2d(192)
        
        self.conv5 = nn.Conv2d(192, 128, 3)
        self.conv5_bn = nn.BatchNorm2d(128)
        
        self.pool = nn.MaxPool2d(3, 3)
        
        self.fc1 = nn.Linear(128 * 5 * 5, 2048)
        self.fc1_bn = nn.BatchNorm1d(2048)
        
        self.fc2 = nn.Linear(2048, 1024)
        self.fc2_bn = nn.BatchNorm1d(1024)
        
        self.fc3 = nn.Linear(1024, NUM_CLASSES)
        self.dropout = nn.Dropout(p = 0.4)

    def forward(self, x):
        x = self.conv1_bn(F.relu(self.conv1(x)))
        x = self.pool(self.conv2_bn(F.relu(self.conv2(x))))
        x = self.pool(self.conv3_bn(F.relu(self.conv3(x))))
        x = self.pool(self.conv4_bn(F.relu(self.conv4(x))))
        x = self.conv5_bn(F.relu(self.conv5(x)))
#         print(x.size())
        x = x.view(x.size()[0], 128 * 5 * 5)
        x = self.dropout(F.relu(self.fc1_bn(self.fc1(x))))
#         print(x.size())
        x = self.dropout(F.relu(self.fc2_bn(self.fc2(x))))
#         print(x.size())
        x = F.relu(self.fc3(x))
        return x


