import os
import numpy as np
import torch
from torch.utils.data import Dataset
import matplotlib.pyplot as plt


class ShallowWaterDataset(Dataset):
    '''
    Dataset for Shallow Water simulation data.
    '''
    def __init__(self, split, sequence_step=10, flatten=False):
        super().__init__()
        if split not in ['train', 'val', 'test']:
            raise ValueError("Invalid split.")
        self.path = f'D:/Datasets/ShallowWater-simulation/{split}'
        self.files = os.listdir(self.path)
        self.sequence_steps = sequence_step
        self.flatten = flatten

        # compute min and max values of h, u, v for normalisation
        self.min_vals = np.array([np.inf, np.inf, np.inf]).reshape(1, 3, 1, 1)
        self.max_vals = np.array([-np.inf, -np.inf, -np.inf]).reshape(1, 3, 1, 1)
        self.file_data = []
        for file in self.files:
            file_path = os.path.join(self.path, file)
            data = np.load(file_path)
            self.min_vals = np.minimum(self.min_vals, data.min(axis=(0, 2, 3), keepdims=True))
            self.max_vals = np.maximum(self.max_vals, data.max(axis=(0, 2, 3), keepdims=True))
            self.file_data.append(data)

        # calculate number of sequences
        self.sequence_num_per_file = self.file_data[0].shape[0] - self.sequence_steps + 1
        self.sequence_num = self.sequence_num_per_file * len(self.file_data)

    def __len__(self):
        return self.sequence_num

    def __getitem__(self, idx):
        file_idx = idx // self.sequence_num_per_file
        seq_start_idx = idx % self.sequence_num_per_file
        data = self.file_data[file_idx][seq_start_idx: seq_start_idx + self.sequence_steps]

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
        min_val = data.min(axis=(0, 2, 3))
        max_val = data.max(axis=(0, 2, 3))

        fig, axs = plt.subplots(n_channels, seq_len, figsize=(seq_len, n_channels))
        for i in range(n_channels):
            for j in range(seq_len):
                axs[i, j].imshow(data[j, i], vmin=min_val[i], vmax=max_val[i])
                axs[i, j].axis('off')
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, bbox_inches='tight', dpi=300)
        else:
            plt.show()
        plt.close()


if __name__ == '__main__':
    dataset = ShallowWaterDataset(split='train')
    print(len(dataset), dataset[0].shape)  # size, (seq_len, n_features...)
    print(dataset.min_vals.squeeze())
    print(dataset.max_vals.squeeze())
    ShallowWaterDataset.visualise_sequence(dataset[-1])
