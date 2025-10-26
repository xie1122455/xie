import torch
from torch.utils.data import Dataset, DataLoader
from torch_geometric.datasets import Planetoid

class SimpleCoraDataset(Dataset):
    def __init__(self):
        self.data = Planetoid(root='./data', name='Cora')[0]
        
    def __len__(self):
        return self.data.num_nodes

    def __getitem__(self, idx):
        return self.data.x[idx], self.data.y[idx]


dataset = SimpleCoraDataset()
train_loader = DataLoader(dataset, batch_size=32, shuffle=True)

for x, y in train_loader:
    print(f"x: {x.shape}, y: {y.shape}")
    break