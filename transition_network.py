import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

#convolutiona neural network class for shot boundary detector creation
import torch.nn as nn
import torch.nn.functional as F

class TransitionCNN(nn.Module):
    def __init__(self, num_channels=3, num_classes=2):
        super(TransitionCNN, self).__init__()
        self.conv1 = nn.Conv3d(in_channels=num_channels, out_channels=16,kernel_size=(3,5,5))

        self.conv2 = nn.Conv3d(in_channels=16,out_channels=24,kernel_size=(3,3,3))

        self.conv3 = nn.Conv3d(in_channels=24, out_channels=32,kernel_size=(3,3,3))

        self.conv4 = nn.Conv3d(in_channels=32, out_channels=12,kernel_size=(1,6,6))

        self.softmaxConv = nn.Conv3d(in_channels=12, out_channels=2,kernel_size=(4,1,1))
        
        self.max_pool = nn.MaxPool3d(kernel_size=(1,2,2), stride=(1,2,2))
        
        self.softmax = nn.Softmax(dim=1)

        # softmax should be two dimensional -- or use sigmoid -- cross Etropy loss is used
        # in paper so use two dimensional softmax
        

    def forward(self, x):
        
        x = F.relu(self.conv1(x))
        x = self.max_pool(x)

        x = F.relu(self.conv2(x))
        x = self.max_pool(x)

        x = F.relu(self.conv3(x))
        x = self.max_pool(x)

        x = F.relu(self.conv4(x))

        x = F.relu(self.softmaxConv(x))

        # x = self.softmax(x)

        return x
        

