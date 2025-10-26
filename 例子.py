
import torch
import numpy as np
import scipy.sparse as sp
import os
from torch.utils.data import Dataset

class CORADataset(Dataset):
    def __init__(self, root='./'):
        content = np.genfromtxt(os.path.join(root, 'Cora.content'), dtype=str)[np.argsort(np.genfromtxt(os.path.join(root, 'Cora.content'), dtype=str)[:,0].astype(int))]
        
        self.features = content[:,1:-1].astype(np.float32) / np.maximum(content[:,1:-1].astype(np.float32).sum(1, keepdims=True), 1)
        self.labels = np.unique(content[:,-1], return_inverse=True)[1].astype(np.int64)
        
        cites = np.genfromtxt(os.path.join(root, 'Cora.cites'), dtype=int)
        edges = np.searchsorted(content[:,0].astype(int), cites)
        self.adj = sp.coo_matrix((np.ones(len(np.unique(np.vstack([edges, edges[:,::-1]]), axis=0))), 
                                 tuple(np.unique(np.vstack([edges, edges[:,::-1]]), axis=0).T)), 
                                shape=(len(self.labels),)*2)

    def __getitem__(self, idx):
        return torch.tensor(self.features[idx]), torch.tensor(self.labels[idx])

    def __len__(self):
        return len(self.labels)


