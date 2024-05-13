import os
import numpy as np
import torch
from torch.utils.data import Dataset
import matplotlib.pyplot as plt


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
    def __init__(self, split, flatten=False):
        super().__init__()
        if split not in ['train', 'val', 'test']:
            raise ValueError("Invalid split.")
        self.path = f'D:/Datasets/ShallowWater-simulation/{split}'
        self.files = os.listdir(self.path)
        self.flatten = flatten

        # compute min and max values of h, u, v for normalisation
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

        data = torch.from_numpy(data).float()
        if self.flatten:
            data = data.flatten(start_dim=1, end_dim=-1)

        return data

    def normalise(self, data):
        '''
        Normalise a sequence of data with shape (seq_len, n_channels, height, width)
        '''
        return (data - self.min_vals) / (self.max_vals - self.min_vals)

    @staticmethod
    def visualise_sequence(data, save_path=None):
        '''
        Visualise a sequence of data with shape (seq_len, n_channels, height, width).
        '''
        data = data.cpu().numpy()
        seq_len, n_channels, height, width = data.shape
        fig, axs = plt.subplots(n_channels, seq_len, figsize=(seq_len, n_channels))
        for i in range(n_channels):
            for j in range(seq_len):
                axs[i, j].imshow(data[j, i])
                axs[i, j].axis('off')
        if save_path:
            plt.savefig(save_path, bbox_inches='tight', dpi=300)
        else:
            plt.show()
        plt.close()


if __name__ == '__main__':
    dataset = ShallowWaterDataset(split='test')  # SynthDataset(1000)
    print(len(dataset), dataset[0].shape)  # size, (seq_len, n_features...)
    print(dataset.min_vals.squeeze())
    print(dataset.max_vals.squeeze())
    ShallowWaterDataset.visualise_sequence(dataset[0])
