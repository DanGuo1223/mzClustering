import timm
import torch
import torch.nn as nn
import torchvision.models as models
import numpy as np
from torch.autograd import Variable
import torch.nn.functional as F
import torchvision.transforms as transforms
import sys
from random import sample

# Detect devices
use_cuda = torch.cuda.is_available()                   # check if GPU exists
device = torch.device("cuda" if use_cuda else "cpu")   # use CPU or GPU


def get_batch(train_image, train_label, batch_size):
  sample_id = sample(range(len(train_image)), batch_size)
  batch_image = train_image[sample_id,]
  batch_label = train_label[sample_id,]
  return batch_image, batch_label


def get_batch_sequential(train_image, train_label, batch_size, i):
  if i < len(train_image)//batch_size:
    batch_image = train_image[(batch_size*i):(batch_size*(i+1)),:]
    batch_label = train_label[(batch_size*i):(batch_size*(i+1))]
  else:
    batch_image = train_image[(batch_size*i):len(train_image),:]
    batch_label = train_label[(batch_size*i):len(train_image)]
  return batch_image, batch_label
##### read data
np.set_printoptions(threshold=sys.maxsize)
num_cluster =10
height = 125
width = 110
sampleN = 4123 
channel = 1
spec = np.genfromtxt('spec.csv',delimiter=' ')
spec_train = np.reshape(spec, (-1, 1, height, width))
label = np.genfromtxt('fake_label.csv',delimiter=' ')
train_labels = np.asarray(label, dtype=np.int32)
image_data = spec_train
image_label = train_labels

#### normalize
for i in range(0, sampleN):
  current_min = np.min(image_data[i,::])
  current_max = np.max(image_data[i,::])
  image_data[i,::] = (current_max - image_data[i,::]) / (current_max - current_min)

#### replicate three chanels
image_data = np.repeat(image_data, 3, axis = 1)




##### creat model
model = timm.create_model('xception', pretrained=True)
model.eval()

activation = {}
def get_activation(name):
    def hook(model, input, output):
        activation[name] = output.detach()
    return hook



#model.avgpool.register_forward_hook(get_activation('avgpool'))
model.global_pool.register_forward_hook(get_activation('global_pool'))

##### inference run
batch_size = 699

features = np.zeros((1,2048))
with torch.no_grad():
  for iter in range(len(image_data)//batch_size + 1):
    train_x, train_y = get_batch_sequential(image_data, image_label, batch_size, iter)
    train_x = resize(train_x, (len(train_x), 3, 299,299))
    train_x = torch.Tensor(train_x).to(device)
    #train_x = transform(train_x)
    x_p = model(train_x)
    
    features = np.concatenate((features, np.reshape(activation['global_pool'].cpu().detach().numpy(),(-1,2048))))
    if iter%100 == 0:
      print(iter)


##### k-means clustering on features
kmeans = KMeans(n_clusters = num_cluster, init='k-means++', max_iter=500, n_init=10, random_state=2)
km_labels = kmeans.fit_predict(features[1:,:])

np.savetxt('xception_km.txt', km_labels)
