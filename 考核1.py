import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch_geometric.nn import GCNConv
import torch.nn.functional as F

class CoraDataset(Dataset):
    def __init__(self):
        self.num_nodes = 2700
        self.num_features = 1433
        self.num_classes = 7
        self.features = torch.randn(self.num_nodes, self.num_features)
        self.labels = torch.randint(0, self.num_classes, (self.num_nodes,))
        self.edge_index = torch.randint(0, self.num_nodes, (2, 10556))  
    def __len__(self): return self.num_nodes
    def __getitem__(self, idx): 
        return {'x': self.features[idx], 'y': self.labels[idx]}


class GCN(nn.Module):
    def __init__(self, in_feats, hidden_size, num_classes):
        super().__init__()
        self.conv1 = GCNConv(in_feats, hidden_size)
        self.conv2 = GCNConv(hidden_size, num_classes)
    
    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1)

dataset = CoraDataset()
model = GCN(dataset.num_features, 16, dataset.num_classes)
data_loader = DataLoader(dataset, batch_size=32, shuffle=True)
print(f"数据集: {len(dataset)}节点, {dataset.num_features}特征, {dataset.num_classes}类别")
print(f"模型参数: {sum(p.numel() for p in model.parameters())}个")
