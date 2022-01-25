import sys
import numpy as np
from sklearn.cluster import KMeans
from sklearn import manifold
from time import time
from matplotlib import offsetbox
import matplotlib.pyplot as plt
import seaborn as sns
#### plot embedding

def plot_embedding(X, label, title=None):
    x_min, x_max = np.min(X, 0), np.max(X, 0)
    X = (X - x_min) / (x_max - x_min)
    plt.figure()
    ax = plt.subplot(111)
    for i in range(X.shape[0]):
        plt.text(X[i, 0], X[i, 1], str(label[i]),
                 color=plt.cm.Set1((label[i]+2) / 10.),
                 fontdict={'weight': 'bold', 'size': 9})
    if hasattr(offsetbox, 'AnnotationBbox'):
        # only print thumbnails with matplotlib > 1.0
        shown_images = np.array([[1., 1.]])  # just something big
        for i in range(X.shape[0]):
            dist = np.sum((X[i] - shown_images) ** 2, 1)
            if np.min(dist) < 4e-3:
                # don't show points that are too close
                continue
            shown_images = np.r_[shown_images, [X[i]]]

    plt.xticks([]), plt.yticks([])
    if title is not None:
        plt.title(title)

## Data loader
np.set_printoptions(threshold=sys.maxsize)
# Option
mode = 'Training'
num_cluster =10
eps = 1e-10
height = 125
width = 110
channel = 1
sampleN = 4123 #746
spec = np.genfromtxt('spec.csv',delimiter=' ')
spec_train = np.reshape(spec, (-1, 1, height, width))
label = np.genfromtxt('fake_label.csv',delimiter=' ')
train_labels = np.asarray(label, dtype=np.int32)
image_data = spec_train
image_label = train_labels

##### normalize

print('min:', np.min(image_data[84,::]), 'max:', np.max(image_data[84,::]))
for i in range(0, sampleN):
  current_min = np.min(image_data[i,::])
  current_max = np.max(image_data[i,::])
  image_data[i,::] = (current_max - image_data[i,::]) / (current_max - current_min)

#### plot ion image
print(np.shape(image_data))
plt.imshow(np.reshape(image_data[84,::],(125,110)))

############K-means

X=np.reshape(image_data, (-1, height*width))

kmeans = KMeans(n_clusters=10, init='k-means++', max_iter=500, n_init=10, random_state=2)
km_labels = kmeans.fit_predict(X)

########### t-sne embeddings
tsne = manifold.TSNE(n_components=2, init='pca', random_state=0)
t0 = time()
X_tsne = tsne.fit_transform(X)
print('plot embedding')
plot_embedding(X_tsne, label = km_labels,
               title = "t-SNE embedding of the digits (time %.2fs)" %
               (time() - t0))
plt.show()

plt.figure(figsize=(5,3.5))
sns.scatterplot(
    x=X_tsne[:,0], y=X_tsne[:,1],
    hue=km_labels,
    palette=sns.color_palette("hls", num_cluster),#3
    legend="full"
)
plt.xlabel('TSNE 1')
plt.ylabel('TSNE 2')