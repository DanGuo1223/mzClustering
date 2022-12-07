# mzClustering
This repository contains R implementation of MSI data preprocessing, visualization, and spatial centroid segmentation using package `Cardinal` and Python implementation of $m/z$ clustering using convolutional neural network based deep clustering method.

## $m/z$ clustering

```python
from mz_clustering import *
```
### Self-supervised training of the clustering network

```python
clusterNet = clustering (spec_path, label_path, 
                        num_cluster = 7, height = 40, width = 40, KNN = True, k = 10)
```

`spec_path` :  path to the .csv file of MSI spectra data

`label_path`:  path to the cluster labels of $m/z$ if available. Default is None. It is only used to calculate the clustering accuracy.

`num_clusters`:  specifies the number of $m/z$ clusters.

`height`  and `width`:  specify the height and width of ion images.

`KNN`: True if including KNN for pseudo labeling. Default is True.

`k`: k in KNN if KNN is used.

```python
cae, CLUST = clusterNet.train(use_gpu = True)
```

`use_gpu` specifies whether to use gpu, default is True.

### Running inference on all $m/z$

```python
pred_label = clusterNet.inference(cae, CLUST)
```

### t-SNE visualization of $m/z$ and predicted cluster memberships

```python
clusterNet.tsne_viz(pred_label)
```
