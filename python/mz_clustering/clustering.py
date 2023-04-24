
import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models
import numpy as np
from torch.autograd import Variable
import torch.nn.functional as F
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import random
from sklearn import manifold
from time import time
from matplotlib import offsetbox
import seaborn as sns
from sklearn.cluster import KMeans

from CAE import CAE
from cnnClust import cnnClust
from pseudo_labeling import pseudo_labeling, K_nearst_neighbor
from utils import clustering_acc, NMI, ARI

def get_batch(train_image, train_label = None, batch_size = 128):
    batch_data = dict()
    sample_id = random.sample(range(len(train_image)), batch_size)

    index = [[]]
    index[0] = [x for x in range(batch_size)]
    index.append(sample_id)

    batch_data['image'] = train_image[sample_id,]

    if train_label is not None:
        batch_data['label'] = train_label[sample_id,]

    batch_data['index'] = index

    return batch_data


def get_batch_sequential(train_image, train_label = None, batch_size = 128, i = 0):
    batch_data = dict()

    if i < len(train_image)//batch_size:
        batch_data['image'] = train_image[(batch_size*i):(batch_size*(i+1)),:]
        if train_label is not None:
            batch_data['label'] = train_label[(batch_size*i):(batch_size*(i+1)),]
    else:
        batch_data['image'] = train_image[(batch_size*i):len(train_image),:]
        if train_label is not None:
                batch_data['label'] = train_label[(batch_size*i):len(train_image),]

    return batch_data

class clustering(object):
    def __init__(self, spec_path, label_path = None, num_cluster = 7, height = 40, width = 40, lr = 0.0001, batch_size = 128, KNN = True, k = 10, random_seed = 6):
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
        self.random_seed = random_seed
        torch.manual_seed(self.random_seed)
        torch.backends.cudnn.deterministic = True

        random.seed(self.random_seed)
        np.random.seed(self.random_seed)


        self.spec = np.genfromtxt(self.spec_path, delimiter=' ')
        self.image_data = np.reshape(self.spec, (-1, self.height, self.width, 1))
        self.sampleN = self.image_data.shape[0]
        device = 'cuda'
        self.cae = CAE(image_height = self.height, image_width = self.width, seed = self.random_seed).to(device)
        self.CLUST = cnnClust(num_clust = self.num_cluster, image_height = self.height, image_width = self.width, seed = self.random_seed).to(device)

        if self.label_path:
            self.label = np.genfromtxt(self.label_path,delimiter=' ')
            self.image_label = np.asarray(self.label, dtype=np.int32)

        ### image normalization
        print('min:', np.min(self.image_data[84,::]), 'max:', np.max(self.image_data[84,::]))
        for i in range(0, self.sampleN):
            current_min = np.min(self.image_data[i,::])
            current_max = np.max(self.image_data[i,::])
            self.image_data[i,::] = (current_max - self.image_data[i,::]) / (current_max - current_min)



    def loss_func(self, x_p, x):

        MSE = F.mse_loss(x_p, x, reduction = 'mean')

        return MSE

    def train(self, use_gpu = True):
        device = torch.device("cuda" if use_gpu else "cpu")
        results = dict()

        model_params = list(self.cae.parameters()) + list(self.CLUST.parameters())
        optimizer = torch.optim.RMSprop(params = model_params,lr = 0.001, weight_decay = 0)

        u = 98
        l = 46


        if self.KNN:
            A = K_nearst_neighbor(data = self.image_data, dist = 'euclidean', K = self.k)
        else:
            A = None

        self.cae.train()
        self.CLUST.train()
        for epoch in range(0, 11):
            losses = list()
            for iter in range(501):

                batch_data = get_batch(self.image_data, self.image_label, self.batch_size)

                train_x = torch.Tensor(batch_data['image']).to(device)
                train_x = train_x.reshape((-1, 1, self.height, self.width))

                optimizer.zero_grad()
                x_p, z = self.cae(train_x)
                loss = self.loss_func(x_p, train_x)
                loss.backward()
                optimizer.step()
                losses.append(loss.item())

            print('Train Epoch: {} Loss: {:.6f}'.format(
                      epoch + 1, sum(losses)/len(losses)))



        losses_pos = list()
        losses_neg = list()


        for epoch in range(0, 11):

            for iter in range(2*(epoch + 1)): #31 #10*(epoch + 1)

                batch_data = get_batch(self.image_data, self.image_label, self.batch_size)

                train_x = torch.Tensor(batch_data['image']).to(device)
                train_x = train_x.reshape((-1, 1, self.height, self.width))

                optimizer.zero_grad()
                x_p,z = self.cae(train_x)

                loss1 = self.loss_func(x_p, train_x)

                features = self.CLUST(x_p)
                features = F.normalize(features, p = 2, dim = -1)
                features = features / features.norm(dim=1)[:, None]



                sim_mat = torch.matmul(features, torch.transpose(features, 0,1))
                tmp = sim_mat.cpu().detach().numpy()
                tmp2 = [tmp[i][j] for i in range(0,self.batch_size-1) for j in range(self.batch_size-1) if i!=j ]
                ub = np.percentile(tmp2, u)
                lb = np.percentile(tmp2, l)

                pos_loc, neg_loc = pseudo_labeling(sim_mat, ub, lb, KNN = self.KNN, A = A, index = batch_data['index'])
                pos_entropy = torch.mul(-torch.log(torch.clip(sim_mat, 1e-10, 1)), pos_loc)
                neg_entropy = torch.mul(-torch.log(torch.clip(1-sim_mat, 1e-10, 1)), neg_loc)
                loss2 = pos_entropy.sum()/pos_loc.sum() + neg_entropy.sum()/neg_loc.sum()

                loss = 1000*loss1 + loss2
                loss.backward()
                optimizer.step()

                losses_pos.append(pos_entropy.sum()/pos_loc.sum().item())
                losses_neg.append(neg_entropy.sum()/neg_loc.sum().item())


            pred_label = torch.argmax(features, dim = 1)
            pred_label = pred_label.cpu().detach().numpy()

            if 'label' in batch_data:
                y =  batch_data['label']
                acc = clustering_acc(y, pred_label)
                nmi = NMI(y, pred_label)
                ari = ARI(y, pred_label)
                print('NMI, ARI, ACC at epoch %d is %f, %f, %f.' % (epoch, nmi, ari, acc))

            u = u - 1
            l = l + 4
        results['cae_weights'], results['CLUST_weights'] = self.cae, self.CLUST
        results['losses'] = list([losses_pos, losses_neg])
        results['KNN_list'] = A
        return results

    def inference(self, use_gpu = True):
        self.cae.eval()
        self.CLUST.eval()
        results = dict()
        with torch.no_grad():
            device = torch.device("cuda" if use_gpu else "cpu")
            pred_labels = list()

            for iter in range(int(np.ceil(len(self.image_data)/self.batch_size))):

                batch_data = get_batch_sequential(self.image_data, self.image_label, self.batch_size, iter)

                train_x = torch.Tensor(batch_data['image']).to(device)

                train_x = train_x.reshape((-1, 1, self.height, self.width))

                x_p,z = self.cae(train_x)
                raw_pred_label = self.CLUST(x_p)
                pred_label = torch.argmax(raw_pred_label, dim = 1)
                pred_labels.extend(pred_label.cpu().detach().numpy())


            pred_labels = np.array(pred_labels)

            if 'label' in batch_data:
                y = batch_data['label']
                acc = clustering_acc(y, pred_labels)
                nmi = NMI(y, pred_labels)
                ari = ARI(y, pred_labels)
                print('testing NMI, ARI, ACC is %f, %f, %f.' % (nmi, ari, acc))
                results['nmi'], results['ari'], results['acc'] = nmi, ari, acc
                results['pred_label'], results['pred_logits'] = pred_label, raw_pred_label
        return results

    def tsne_viz(self, pred_label):
        device = torch.device("cuda")

        batch_data = get_batch_sequential(self.image_data, self.image_label, self.sampleN, 0)
        train_x = torch.Tensor(batch_data['image']).to(device)
        train_x = train_x.reshape((-1, 1, self.height, self.width))
        x_p,z = self.cae(train_x)
        x_p = x_p.cpu().detach().numpy()
        x_p = x_p.reshape(-1, self.height*self.width)
        tsne = manifold.TSNE(n_components=2, init='pca', random_state=0)
        X_tsne = tsne.fit_transform(x_p)

        print('plot embedding')
        plt.figure(figsize=(5,3.5))
        sns.scatterplot(
            x=X_tsne[:,0], y=X_tsne[:,1],
            hue=pred_label,
            palette=sns.color_palette("hls", len(set(pred_label))),
            legend="full"
        )
        plt.xlabel('TSNE 1')
        plt.ylabel('TSNE 2')


