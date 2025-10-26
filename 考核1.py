import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torch_geometric.nn import GCNConv
import torch.nn.functional as F
import os
import scipy.sparse as sp

class CoraDataset(Dataset):
    def __init__(self,data_root="cora/",transfrom=None):
        content = np.genfromtxt(os.path.join(data_root, 'Cora.content'), dtype=str)[np.argsort(np.genfromtxt(os.path.join(data_root, 'Cora.content'), dtype=str)[:,0].astype(int))]
        self.features =content [:,1:-1].astype(np.float32) / np.maximum(content[:,1:-1].astype(np.float32).sum(1, keepdims=True), 1)
        self.labels = np.unique(content[:,-1], return_inverse=True)[1].astype(np.int64)
        cites = np.genfromtxt(os.path.join(data_root, 'Cora.cites'), dtype=int)
        edges = np.searchsorted(content[:,0].astype(int), cites)
        self.adj = sp.coo_matrix((np.ones(len(np.unique(np.vstack([edges, edges[:,::-1]]), axis=0))), 
                                 tuple(np.unique(np.vstack([edges, edges[:,::-1]]), axis=0).T)), 
                                shape=(len(self.labels),)*2)
    def __getitem__(self, idx): 
        return torch.tensor(self.features[idx]), torch.tensor(self.labels[idx]) 
    def __len__(self):
        return len(self.labels)