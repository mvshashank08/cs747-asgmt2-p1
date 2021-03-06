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
        
        self.conv3 = nn.Conv2d(128, 256, 3, padding=1)
        self.conv3_bn = nn.BatchNorm2d(256)
        
        self.conv4 = nn.Conv2d(256, 512, 3)
        self.conv4_bn = nn.BatchNorm2d(512)
        
        self.pool = nn.MaxPool2d(2, 2)
        
        self.fc1 = nn.Linear(512 * 13 * 13, 2048)
        self.fc1_bn = nn.BatchNorm1d(2048)
        
        self.fc2 = nn.Linear(2048, 1024)
        self.fc2_bn = nn.BatchNorm1d(1024)
        
        self.fc3 = nn.Linear(1024, NUM_CLASSES)
        self.dropout = nn.Dropout(p = 0.2)

    def forward(self, x):
        x = self.pool(self.conv1_bn(F.relu(self.conv1(x))))
        x = self.pool(self.conv2_bn(F.relu(self.conv2(x))))
        x = self.pool(self.conv3_bn(F.relu(self.conv3(x))))
        x = self.pool(self.conv4_bn(F.relu(self.conv4(x))))
#         print(x.size())
        x = x.view(x.size()[0], 512 * 13 * 13)
        x = self.dropout(self.fc1_bn(F.relu(self.fc1(x))))
#         print(x.size())
        x = self.dropout(self.fc2_bn(F.relu(self.fc2(x))))
#         print(x.size())
        x = F.relu(self.fc3(x))
        return x


