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
        nn.init.xavier_uniform_(self.conv1.weight)
        self.conv1_bn = nn.BatchNorm2d(64)
        
        self.conv2 = nn.Conv2d(64, 64, 3)
        nn.init.xavier_uniform_(self.conv2.weight)
        self.conv2_bn = nn.BatchNorm2d(64)
        
        self.conv3 = nn.Conv2d(64, 128, 3)
        nn.init.xavier_uniform_(self.conv3.weight)
        self.conv3_bn = nn.BatchNorm2d(128)
        
        self.conv4 = nn.Conv2d(128, 128, 3)
        nn.init.xavier_uniform_(self.conv4.weight)
        self.conv4_bn = nn.BatchNorm2d(128)
        
        self.conv5 = nn.Conv2d(128, 128, 3)
        nn.init.xavier_uniform_(self.conv5.weight)
        self.conv5_bn = nn.BatchNorm2d(128)
        
        self.pool = nn.MaxPool2d(3, 3)
        
        self.fc1 = nn.Linear(128 * 21 * 21, 1024)
        nn.init.xavier_uniform_(self.fc1.weight)
        self.fc1_bn = nn.BatchNorm1d(1024)
        
        self.fc2 = nn.Linear(1024, 512)
        nn.init.xavier_uniform_(self.fc2.weight)
        self.fc2_bn = nn.BatchNorm1d(512)
        
        self.fc3 = nn.Linear(512, NUM_CLASSES)
        nn.init.xavier_uniform_(self.fc3.weight)
        self.dropout = nn.Dropout2d(p = 0.1)

    def forward(self, x):
        p = 0.5
        x = self.conv1_bn(F.relu(self.conv1_bn(self.conv1(x))))
        x = self.pool(self.conv2_bn(F.relu(self.conv2_bn(self.conv2(x)))))
        x = self.conv3_bn(F.relu(self.conv3_bn(self.conv3(x))))
        x = self.pool(self.conv4_bn(F.relu(self.conv4_bn(self.conv4(x)))))
        x = self.conv5_bn(F.relu(self.conv5_bn(self.conv5(x))))
#         print(x.size())
        x = x.view(x.size()[0], 128 * 21 * 21)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        return x


