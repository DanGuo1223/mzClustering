
import torch.nn as nn
import torchvision.models as models
import numpy as np
from torch.autograd import Variable
import torch.nn.functional as F

class resnet_vae(nn.Module):
  def __init__(self, embed_dim = 256, train_mode = True):
    super(resnet_vae, self).__init__()
    self.train_mode = train_mode
    self.fc_h1, self.fc_h2 = 768, 256
    self.k1, self.k2, self.k3 = (5,5), (3,3), (3,3)
    self.s1, self.s2, self.s3 = (2,2), (2,2), (2,2)
    # encoder
    self.train_mode = train_mode

    #model_name = 'B_16_imagenet1k'
    #model = ViT(model_name, pretrained=True)
    #model = models.resnet18(pretrained = True)
    #self.transformer = model

    resnet = models.resnet18(pretrained = True)
    modules = list(resnet.children())[1:-1]
    self.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
    self.conv1.weight.data = resnet.conv1.weight.sum(dim = 1, keepdim = True)
    self.resnet = nn.Sequential(self.conv1, *modules)

    self.fc0 = nn.Linear(resnet.fc.in_features, 1024)
    self.bn0 = nn.BatchNorm1d(1024, momentum = 0.01)

    self.fc1 = nn.Linear(resnet.fc.in_features, self.fc_h1)
    self.bn1 = nn.BatchNorm1d(self.fc_h1, momentum = 0.01)
    self.fc2_mu = nn.Linear(self.fc_h1, self.fc_h2)
    self.fc2_sigma = nn.Linear(self.fc_h1, self.fc_h2)

    # decoder
    self.fc3 = nn.Linear(self.fc_h2, self.fc_h1)
    self.bn3 = nn.BatchNorm1d(self.fc_h1)
    self.fc4 = nn.Linear(self.fc_h1, 64*4*4)
    self.bn4 = nn.BatchNorm1d(64*4*4)
    self.relu = nn.ReLU(inplace = True)

    self.conv_trans1 = nn.Sequential(nn.ConvTranspose2d(in_channels=64, out_channels=32, kernel_size=self.k1, stride = self.s1, padding = (0,0)),
                                     nn.BatchNorm2d(32, momentum = 0.01),
                                     nn.ReLU(inplace=True)
                                     )
    
    self.conv_trans2 = nn.Sequential(nn.ConvTranspose2d(in_channels=32, out_channels=8, kernel_size=self.k2, stride = self.s2, padding = (0,0)),
                                     nn.BatchNorm2d(8, momentum = 0.01),
                                     nn.ReLU(inplace=True)
                                     )
    
    self.conv_trans3 = nn.Sequential(nn.ConvTranspose2d(in_channels=8, out_channels=1, kernel_size=self.k3, stride = self.s3, padding = (0,0)),
                                    nn.BatchNorm2d(1, momentum = 0.01),
                                    nn.Sigmoid()
                                    ) 

  def encode(self, x):
    #x = self.transformer(x)
    x = self.resnet(x)
    x = x.view(x.size(0), -1)
    #x = self.bn0(self.fc0(x))
    x = self.bn1(self.fc1(x))
    x = self.relu(x)
    mu, sigma = self.fc2_mu(x), self.fc2_sigma(x)
    return mu, sigma

  def sampling(self, mu, sigma):
    if self.train_mode:
      std = sigma.mul(0.5).exp_()
      eps = Variable(std.data.new(std.size()).normal_())
      return eps.mul(std).add_(mu)
    else:
      return mu
  
  def decode(self, z):

    z = self.bn3(self.fc3(z))
    z = self.relu(z)
    z = self.bn4(self.fc4(z))
    z = self.relu(z).view(-1,64,4,4)

    z = self.conv_trans1(z)
    z = self.conv_trans2(z)
    z = self.conv_trans3(z)
    z = F.interpolate(z, size = (112,112), mode = 'bilinear')
    return z

  def forward(self, x):
    mu, sigma = self.encode(x)
    z = self.sampling(mu, sigma)
    x = self.decode(z)
    return x, z, mu, sigma


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
    self.clust = nn.Sequential(nn.Linear(resnet.fc.in_features, num_clust),
                               nn.BatchNorm1d(num_clust, momentum = 0.01),
                               nn.Softmax())
    
  def forward(self, x):
    x = self.resnet(x).view(-1,512)
    #print(x.size())
    x = self.clust(x)
    return x