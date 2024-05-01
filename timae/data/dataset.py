import os
import numpy as np
import torch
from torch.utils.data import Dataset


class SynthDataset(Dataset):
    def __init__(self, n_samples, seq_len=100, n_features=50, alpha=200, betta=3):
        super().__init__()
        self.data = self.create_synth_data(n_samples, seq_len=seq_len, n_features=n_features, alpha=alpha, betta=betta)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

    def create_synth_data(self, n_samples, seq_len=100, n_features=50, alpha=200, betta=3):
        t = torch.linspace(0, 1, steps=seq_len).unsqueeze(1)
        t = t.repeat(n_samples, 1, n_features).float()
        X = torch.cos(alpha*t) + torch.cos(alpha*t/2) + torch.cos(alpha*t/4) + betta*t + torch.rand_like(t)
        return X


class ShallowWaterDataset(Dataset):
    def __init__(self, path):
        super().__init__()
        self.path = path
        self.files = os.listdir(self.path)
        
        # compute min and max values of h, u, v
        self.min_vals = np.array([np.inf, np.inf, np.inf]).reshape(1, 3, 1, 1)
        self.max_vals = np.array([-np.inf, -np.inf, -np.inf]).reshape(1, 3, 1, 1)
        for file in self.files:
            file_path = os.path.join(self.path, file)
            data = np.load(file_path)
            self.min_vals = np.minimum(self.min_vals, data.min(axis=(0, 2, 3), keepdims=True))
            self.max_vals = np.maximum(self.max_vals, data.max(axis=(0, 2, 3), keepdims=True))

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        file_path = os.path.join(self.path, self.files[idx])
        data = np.load(file_path)
        data = self.normalise(data)

        data = torch.from_numpy(data).float().flatten(start_dim=1, end_dim=-1)
        return data

    def normalise(self, data):
        '''
        Normalise a sequence of data with shape (seq_len, n_channels, height, width)
        '''
        return (data - self.min_vals) / (self.max_vals - self.min_vals)


if __name__ == '__main__':
    dataset = ShallowWaterDataset(path='shallow_water/train')  # SynthDataset(1000)
    print(len(dataset), dataset[0].shape)  # size, (seq_len, n_features)
    print(dataset.min_vals.squeeze())
    print(dataset.max_vals.squeeze())
