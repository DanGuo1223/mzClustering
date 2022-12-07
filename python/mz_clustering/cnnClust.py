import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models
import numpy as np
from torch.autograd import Variable
import torch.nn.functional as F
import torchvision.transforms as transforms
class cnnClust(nn.Module):
    def __init__(self, num_clust):
        super(cnnClust, self).__init__()
        self.num_clust = num_clust
#         resnet = models.resnet18(pretrained = True)
#         modules = list(resnet.children())[1:-1]
#         self.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
#         self.conv1.weight.data = resnet.conv1.weight.sum(dim = 1, keepdim = True)
#         self.resnet = nn.Sequential(self.conv1, *modules)
        #self.resnet = nn.Sequential(*modules)
        self.conv1 = nn.Sequential(nn.Conv2d(1, 1, kernel_size=(7, 7), stride=(1, 1), padding=(3, 3), bias=False),
                                   nn.BatchNorm2d(1),
                                   nn.ReLU()
        )
        self.conv2 = nn.Sequential(nn.Conv2d(1, 8, kernel_size=(2, 2), stride=(1, 1), bias=False),
                                   nn.BatchNorm2d(8),
                                   nn.ReLU(),
                                   nn.MaxPool2d((2,2), (2,2)),
                                   nn.BatchNorm2d(8),
                                   nn.ReLU()

        )
        self.conv3 = nn.Sequential(nn.Conv2d(8, 8, kernel_size=(2, 2), stride=(1, 1), bias=False),
                                   nn.BatchNorm2d(8),
                                   nn.ReLU()

        )
        self.conv4 = nn.Sequential(nn.Conv2d(8, 16, kernel_size=(3, 3), stride=(1, 1), bias=False),
                                   nn.BatchNorm2d(16),
                                   nn.ReLU()

        )
        self.conv5 = nn.Sequential(nn.Conv2d(16, 16, kernel_size=(3, 3), stride=(1, 1), bias=False),
                                   nn.BatchNorm2d(16),
                                   nn.ReLU(),
                                   nn.MaxPool2d((2,2), (2,2)),
                                   nn.BatchNorm2d(16),
                                   nn.ReLU()

        )
        self.conv6 = nn.Sequential(nn.Conv2d(16, 1, kernel_size=(3, 3), stride=(1, 1), bias=False),
                                   nn.BatchNorm2d(1),
                                   nn.ReLU()

        )
        self.fc = nn.Sequential(nn.Linear(25, num_clust),
                                   nn.BatchNorm1d(num_clust, momentum = 0.01),
                                   nn.Softmax())
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.conv6(x)
        x = x.view(-1,25)
        x = self.fc(x)

        return x