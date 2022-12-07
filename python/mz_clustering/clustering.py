import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models
import numpy as np
from torch.autograd import Variable
import torch.nn.functional as F
import torchvision.transforms as transforms

from random import sample
from sklearn import manifold
from time import time
from matplotlib import offsetbox
import seaborn as sns

import CAE
import cnnClust
import pseudo_labeling
from utils import clustering_acc, NMI, ARI

class clustering(object):
    def __init__(self, spec_path, label_path = None, num_cluster = 7, height = 40, width = 40, lr = 0.0001, batch_size = 128, KNN = True, k = 10):
        super(clustering, self).__init__()
        self.spec_path = spec_path
        self.label_path = label_path
        self.num_cluster = num_cluster
        self.height = height
        self.width = width
        self.lr = lr
        self.batch_size = batch_size
        self.KNN = KNN
        self.k = k
        
        self.spec = np.genfromtxt(self.spec_path,delimiter=' ')

        self.image_data = np.reshape(self.spec, (-1, self.height, self.width, 1))
        
        self.sampleN = len(self.image_data)
        
        if self.label_path:
            self.label = np.genfromtxt(self.label_path,delimiter=' ')

            self.image_label = np.asarray(self.label, dtype=np.int32)
        ### image normalization
        for i in range(0, self.sampleN):
            current_min = np.min(self.image_data[i,::])
            current_max = np.max(self.image_data[i,::])
            self.image_data[i,::] = (current_max - self.image_data[i,::]) / (current_max - current_min)
    
    def get_batch(self, train_image, train_label, batch_size):
        sample_id = sample(range(len(train_image)), batch_size)
        index = [[]]
        index[0] = [x for x in range(batch_size)]
        index.append(sample_id)
        batch_image = train_image[sample_id,]
        batch_label = train_label[sample_id,]
        return batch_image, batch_label, index


    def get_batch_sequential(self, train_image, train_label, batch_size, i):
        if i < len(train_image)//batch_size:
            batch_image = train_image[(batch_size*i):(batch_size*(i+1)),:]
            batch_label = train_label[(batch_size*i):(batch_size*(i+1))]
        else:
            batch_image = train_image[(batch_size*i):len(train_image),:]
            batch_label = train_label[(batch_size*i):len(train_image)]
        return batch_image, batch_label
    def train(self, use_gpu = True):
        device = torch.device("cuda" if use_gpu else "cpu")
        
        cae = CAE(train_mode = True).to(device)
        CLUST = cnnClust(num_clust = self.num_cluster).to(device)
        
        model_params = list(cae.parameters()) + list(CLUST.parameters())
        optimizer = torch.optim.RMSprop(params = model_params,lr = 0.001, weight_decay=0) #torch.optim.Adam(model_params, lr=lr)

        u = 98 
        l = 46 
        loss_list = list()

        random_seed = 1224
        torch.manual_seed(random_seed)
        torch.cuda.manual_seed(random_seed)
        torch.backends.cudnn.deterministic = True

        for epoch in range(0, 11):
            losses = list()
            for iter in range(501):

                train_x, train_y, index = get_batch(self.image_data, self.image_label, self.batch_size)

                train_x = torch.Tensor(train_x).to(device)
                train_x = train_x.reshape((-1, 1, height, width))
                optimizer.zero_grad()
                x_p = cae(train_x)

                loss = loss_func(x_p, train_x)
                loss.backward()
                optimizer.step()
                losses.append(loss.item())

            print('Train Epoch: {} Loss: {:.6f}'.format(
                      epoch + 1, sum(losses)/len(losses)))

        optimizer = torch.optim.RMSprop(params = model_params,lr = 0.01, weight_decay=0.0)
        for epoch in range(0, 11):

            losses = list()
            losses2 = list()

            train_x, train_y, index = get_batch(self.image_data, self.image_label, self.batch_size)


            train_x = torch.Tensor(train_x).to(device)

            train_x = train_x.reshape((-1, 1, height, width))




            x_p = cae(train_x)
            features = CLUST(x_p)
            features = F.normalize(features, p = 2, dim = -1)
            features = features / features.norm(dim=1)[:, None]
            sim_mat = torch.matmul(features, torch.transpose(features, 0,1))




            for iter in range(31):

                train_x, train_y, index = get_batch(self.image_data, self.image_label, self.sampleN)


                train_x = torch.Tensor(train_x).to(device)

                train_x = train_x.reshape((-1, 1, height, width))

                optimizer.zero_grad()
                x_p = cae(train_x)

                loss1 = loss_func(x_p, train_x)

                features = CLUST(x_p)
                features = F.normalize(features, p = 2, dim = -1)
                features = features / features.norm(dim=1)[:, None]
                sim_mat = torch.matmul(features, torch.transpose(features, 0,1))

                tmp = sim_mat.cpu().detach().numpy()
                tmp2 = [tmp[i][j] for i in range(0,self.batch_size-1) for j in range(self.batch_size-1) if i!=j ]
                ub = np.percentile(tmp2, u)
                lb = np.percentile(tmp2, l)
    
                pos_loc, neg_loc = pseudo_labeling(features, ub, lb, index = index, KNN = True, A = A)
                pos_entropy = torch.mul(-torch.log(torch.clip(sim_mat, 1e-10, 1)), pos_loc)
                neg_entropy = torch.mul(-torch.log(torch.clip(1-sim_mat, 1e-10, 1)), neg_loc)

                loss2 = pos_entropy.sum()/pos_loc.sum() + neg_entropy.sum()/neg_loc.sum()

                loss = 1000*loss1 + loss2
                losses.append(loss1.item())
                losses2.append(loss2.item())
                loss.backward()
                optimizer.step()
                loss_list.append(sum(losses)/len(losses))






            u = u - 1
            l = l + 4
        return cae, CLUST
    def inference(self, cae, CLUST)
        with torch.no_grad():
            pred_label = list()
            for iter in range(len(self.image_data)//self.batch_size):

                train_x, train_y, _ = get_batch_sequential(self.image_data, self.image_label, self.sampleN, iter)


                train_x = torch.Tensor(train_x).to(device)
                train_x = train_x.reshape((-1, 1, height, width))

                x_p = cae(train_x)
                psuedo_label = CLUST(x_p)

                psuedo_label = torch.argmax(psuedo_label, dim = 1)
                pred_label.extend(psuedo_label.cpu().detach().numpy())

            pred_label = np.array(pred_label)
            acc = clustering_acc(train_y, pred_label)

            nmi = NMI(image_label, pred_label)
            ari = ARI(image_label, pred_label)
            print('testing NMI, ARI, ACC at epoch %d is %f, %f, %f.' % (epoch, nmi, ari, acc))

        return nmi, ari, acc, pred_label
    def tsne_viz(self, pred_label):
        X = np.reshape(self.image_data, (-1, self.height*self.width))
        tsne = manifold.TSNE(n_components=2, init='pca', random_state=0)
        t0 = time()
        X_tsne = tsne.fit_transform(X)
        print('plot embedding')
        plt.figure(figsize=(5,3.5))
        sns.scatterplot(
            x=X_tsne[:,0], y=X_tsne[:,1],
            hue=pred_label,
            palette=sns.color_palette("hls", self.num_cluster),
            legend="full"
        )
        plt.xlabel('TSNE 1')
        plt.ylabel('TSNE 2')
    