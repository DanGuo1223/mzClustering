import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models
import numpy as np
from torch.autograd import Variable
import torch.nn.functional as F
import torchvision.transforms as transforms


class CAE(nn.Module):
  def __init__(self, embed_dim = 256, train_mode = True):
    super(CAE, self).__init__()
    self.train_mode = train_mode
    self.fc_h1, self.fc_h2 = 768, 256
    self.k1, self.k2, self.k3 = (5,5), (3,3), (3,3)
    self.s1, self.s2, self.s3 = (3,3), (2,2), (2,2)

    # encoder

    self.conv1 = nn.Sequential(nn.Conv2d(1, 8, kernel_size=(3, 3), stride=(2, 2), padding=(0, 0), bias=False),
                               nn.BatchNorm2d(8, momentum = 0.01),
                               nn.ReLU())
    
    self.conv2 = nn.Sequential(nn.Conv2d(8, 16, kernel_size=(3, 3), stride=(3, 3), padding=(0, 0), bias=False),
                            nn.BatchNorm2d(16, momentum = 0.01),
                            nn.ReLU())

    
    self.encoder = nn.Linear(4080, 7)


    # decoder
    self.fc3 = nn.Linear(7, 4080)


    self.conv_trans1 = nn.Sequential(nn.ConvTranspose2d(in_channels=16, out_channels=8, kernel_size=self.k1, stride = self.s1, padding = (0,0)),
                                     nn.BatchNorm2d(8, momentum = 0.01),
                                     nn.ReLU(inplace=True)
                                     )
    
    self.conv_trans2 = nn.Sequential(nn.ConvTranspose2d(in_channels=8, out_channels=1, kernel_size=self.k2, stride = self.s2, padding = (0,0)),
                                     nn.BatchNorm2d(1, momentum = 0.01),
                                     nn.Sigmoid()
                                     )
    
    #self.conv_trans3 = nn.Sequential(nn.ConvTranspose2d(in_channels=8, out_channels=1, kernel_size=self.k3, stride = self.s3, padding = (0,0)),
    #                                nn.BatchNorm2d(1, momentum = 0.01),
    #                                nn.Sigmoid()
    #                                ) 

  def encode(self, x):
    #x = self.fc3(x)
    x = self.conv1(x)
    x = self.conv2(x)
    #print(x.size())
    x = x.view(x.size(0), -1)
    x = self.encoder(x)
    return x


  
  def decode(self, z):

    z = self.fc3(z)
    
    z = z.view(-1, 16, 17, 15)

    z = self.conv_trans1(z)
    z = self.conv_trans2(z)

    z = F.interpolate(z, size = (108,96), mode = 'bilinear')
    return z

  def forward(self, x):
    #mu, sigma = self.encode(x)
    z = self.encode(x)
    xp = self.decode(z)
    return xp


class cnnClust(nn.Module):
  def __init__(self, num_clust):
    super(cnnClust, self).__init__()
    self.num_clust = num_clust
    resnet = models.resnet18(pretrained = True)
    modules = list(resnet.children())[1:-1]
    self.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
    self.conv1.weight.data = resnet.conv1.weight.sum(dim = 1, keepdim = True)
    self.resnet = nn.Sequential(self.conv1, *modules)
    #self.resnet = nn.Sequential(*modules)
    self.conv1 = nn.Sequential(nn.Conv2d(1, 1, kernel_size=(2, 2), stride=(1, 1), padding=(3, 3), bias=False),
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
    self.fc = nn.Sequential(nn.Linear(460, num_clust),
                               nn.BatchNorm1d(num_clust, momentum = 0.01),
                               nn.Softmax())
  def forward(self, x):
    x = self.conv1(x)
    x = self.conv2(x)
    x = self.conv3(x)
    x = self.conv4(x)
    x = self.conv5(x)
    x = self.conv6(x)
    x = x.view(-1,460)
    x = self.fc(x)

    return x