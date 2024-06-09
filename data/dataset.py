import os
import numpy as np
import torch
from torch.utils.data import Dataset
import matplotlib.pyplot as plt


class ShallowWaterDataset(Dataset):
    '''
    Dataset for Shallow Water simulation data.
    '''

    def __init__(self, split, sequence_steps=10, forecast_steps=5, masking_steps=2, flatten=False):
        super().__init__()
        if split not in ['train', 'val', 'test']:
            raise ValueError("Invalid split.")
        self.path = f'D:/Datasets/ShallowWater-simulation/{split}'
        self.files = os.listdir(self.path)
        self.sequence_steps = sequence_steps
        self.forecast_steps = forecast_steps
        self.masking_steps = masking_steps
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

        x, y = data[:self.sequence_steps-self.forecast_steps], data[self.sequence_steps-self.forecast_steps:]

        # generate random mask
        mask = torch.zeros(x.size(0))
        mask_idx = np.random.choice(x.size(0), self.masking_steps, replace=False)
        mask[mask_idx] = 1

        return x, y, mask

    def normalise(self, data):
        '''
        Normalise a sequence of data with shape (seq_len, n_channels, height, width)
        '''
        return (data - self.min_vals) / (self.max_vals - self.min_vals)

    @staticmethod
    def visualise_sequence(data, vmin=None, vmax=None, save_path=None):
        '''
        Visualise a sequence of data with shape (seq_len, n_channels, height, width).
        '''
        data = data.cpu().numpy()
        seq_len, n_channels, height, width = data.shape
        vmin = [vmin] * n_channels if isinstance(vmin, (int, float)) else vmin
        vmax = [vmax] * n_channels if isinstance(vmax, (int, float)) else vmax
        min_val = data.min(axis=(0, 2, 3)) if vmin is None else vmin
        max_val = data.max(axis=(0, 2, 3)) if vmax is None else vmax

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

    @staticmethod
    def interpolate(data, mask):
        '''
        Interpolate the masked steps in a sequence of data.

        Args:
            data: torch.Tensor, shape (seq_len, n_channels, height, width)
            mask: torch.Tensor, shape (seq_len), 1 for masked steps, 0 for observed steps
        '''
        data = data.clone()
        seq_len = data.shape[0]

        # Convert mask to boolean for easier indexing
        mask = mask.bool()

        # Handle missing first image
        if mask[0]:
            for i in range(1, seq_len):
                if not mask[i]:
                    data[0] = data[i].clone()
                    break
            else:
                data[0].zero_()

        # Handle missing last image
        if mask[-1]:
            for i in range(seq_len - 2, -1, -1):
                if not mask[i]:
                    data[-1] = data[i].clone()
                    break
            else:
                data[-1].zero_()

        # Interpolating internal missing images
        i = 1
        while i < seq_len - 1:
            if mask[i]:
                start_index = i - 1
                end_index = i + 1
                while end_index < seq_len - 1 and mask[end_index]:
                    end_index += 1

                num_missing = end_index - start_index - 1

                # Linearly interpolate missing images
                for j in range(1, num_missing + 1):
                    weight_start = (num_missing + 1 - j) / (num_missing + 1)
                    weight_end = j / (num_missing + 1)
                    data[start_index + j] = weight_start * data[start_index] + weight_end * data[end_index]

                i = end_index + 1
            else:
                i += 1

        return data


if __name__ == '__main__':
    dataset = ShallowWaterDataset(split='train', forecast_steps=5, masking_steps=2)
    print(len(dataset))  # size
    print(dataset.min_vals.squeeze())
    print(dataset.max_vals.squeeze())
    x, y, mask = dataset[-1]
    print(x.shape, y.shape)
    print(mask)
    
    interpolated_x = ShallowWaterDataset.interpolate(x, mask)
    print((x - interpolated_x).numpy().max(axis=(1, 2, 3)))
    ShallowWaterDataset.visualise_sequence(torch.cat([x, interpolated_x], dim=0))
