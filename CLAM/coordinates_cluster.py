### System
import os, sys
from os.path import join
import h5py
import math
from math import floor
import pdb
from time import time
from tqdm import tqdm

### Numerical Packages
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
from scipy.stats import percentileofscore

### Graph Network Packages
import nmslib
import networkx as nx

### PyTorch / PyG
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric
from torch_geometric.utils import convert

### CLAM Path
clam_path = '/home/compu/doanhbc/WSIs-classification/CLAM/'
sys.path.append(clam_path)
from models.resnet_custom import resnet50_baseline
from wsi_core.WholeSlideImage import WholeSlideImage
from utils.utils import *
from datasets.wsi_dataset import Wsi_Region

import numpy as np
from fast_pytorch_kmeans import KMeans
from sklearn.metrics.pairwise import euclidean_distances

from torch_geometric.data import Data as geomData
from itertools import chain
from tqdm import tqdm


class Hnsw:
    def __init__(self, space='cosinesimil', index_params=None,
                 query_params=None, print_progress=True):
        self.space = space
        self.index_params = index_params
        self.query_params = query_params
        self.print_progress = print_progress

    def fit(self, X):
        index_params = self.index_params
        if index_params is None:
            index_params = {'M': 16, 'post': 0, 'efConstruction': 400}

        query_params = self.query_params
        if query_params is None:
            query_params = {'ef': 90}

        # this is the actual nmslib part, hopefully the syntax should
        # be pretty readable, the documentation also has a more verbiage
        # introduction: https://nmslib.github.io/nmslib/quickstart.html
        index = nmslib.init(space=self.space, method='hnsw')
        index.addDataPointBatch(X)
        index.createIndex(index_params, print_progress=self.print_progress)
        index.setQueryTimeParams(query_params)

        self.index_ = index
        self.index_params_ = index_params
        self.query_params_ = query_params
        return self

    def query(self, vector, topn):
        # the knnQuery returns indices and corresponding distance
        # we will throw the distance away for now
        indices, dist = self.index_.knnQuery(vector, k=topn)
        return indices

ratio = 0.05 # N_cluster = 5% number of patches

def pt2graph(wsi_h5, radius=1):
    total_coords, total_features = np.array(wsi_h5['coords']), np.array(wsi_h5['features'])
    assert total_coords.shape[0] == total_features.shape[0]
    total_num_patches = total_coords.shape[0]
    N_clusters = 256

    # distances = euclidean_distances(coords, coords)
    # import pdb; pdb.set_trace()
    cuda_coords = torch.from_numpy(total_features).float().cuda()
    kmeans = KMeans(n_clusters=N_clusters, mode='euclidean')
    kmeans.fit(cuda_coords)
    cluster_labels = kmeans.predict(cuda_coords).cpu().detach().numpy()
    cluster_data = dict()

    cluster_data['cluster_labels'] = cluster_labels

    return cluster_data

def createDir_h5toPyG(h5_path, save_path):
    pbar = tqdm(os.listdir(h5_path))
    for h5_fname in pbar:
        pbar.set_description('%s - Creating Graph' % (h5_fname[:12]))
        save_fname = os.path.join(save_path, h5_fname[:-3] + '.pt')
        try:
            # if not os.path.exists(save_fname):
            wsi_h5 = h5py.File(os.path.join(h5_path, h5_fname), "r")
            G = pt2graph(wsi_h5)
            torch.save(G, save_fname)
            wsi_h5.close()
        except OSError:
            pbar.set_description('%s - Broken H5' % (h5_fname[:12]))
            print(h5_fname, 'Broken')
            
h5_path = '/data4/doanhbc/camelyon_patches_20x_bwh_biopsy/features/h5_files/'
save_path = '/data4/doanhbc/camelyon_patches_20x_bwh_biopsy/features/graph_cluster_files_256_resnet_features'

# h5_path = '/data4/doanhbc/TCGA-BRCA-breast_patches/features_ctranspath/h5_files/'
# save_path = '/data4/doanhbc/TCGA-BRCA-breast_patches/features_ctranspath/graph_cluster_files_256'
os.makedirs(save_path, exist_ok=True)

createDir_h5toPyG(h5_path, save_path)