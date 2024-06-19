import os
import h5py
import numpy as np
import torch
from torch.utils.data import Dataset
from data.utils import normalise


class ShallowWaterDataset(Dataset):
    '''
    Dataset for Shallow Water simulation data.
    '''

    def __init__(self, split, sequence_steps=15, forecast_steps=5, masking_steps=5, flatten=False):
        super().__init__()
        if split not in ['train', 'val', 'test']:
            raise ValueError("Invalid split.")
        self.path = f'/homes/yx723/bucket/Datasets/ShallowWater-simulation/{split}'
        self.files = os.listdir(self.path)
        self.sequence_steps = sequence_steps
        self.forecast_steps = forecast_steps
        self.masking_steps = masking_steps
        self.flatten = flatten

        # compute min and max values of h, u, v for normalisation
        self.min_vals = np.array([0.70, -0.16, -0.16]).reshape(1, 3, 1, 1)
        self.max_vals = np.array([1.23, 0.16, 0.16]).reshape(1, 3, 1, 1)
        # for file in self.files:
        #     file_path = os.path.join(self.path, file)
        #     data = np.load(file_path)
        #     self.min_vals = np.minimum(self.min_vals, data.min(axis=(0, 2, 3), keepdims=True))
        #     self.max_vals = np.maximum(self.max_vals, data.max(axis=(0, 2, 3), keepdims=True))

        # calculate number of sequences
        self.sequence_num_per_file = np.load(os.path.join(self.path, self.files[0])).shape[0] - self.sequence_steps + 1
        self.sequence_num = self.sequence_num_per_file * len(self.files)

    def __len__(self):
        return self.sequence_num

    def __getitem__(self, idx):
        file_idx = idx // self.sequence_num_per_file
        seq_start_idx = idx % self.sequence_num_per_file
        data = np.load(os.path.join(self.path, self.files[file_idx]))[seq_start_idx: seq_start_idx + self.sequence_steps]

        data = normalise(data, self.min_vals, self.max_vals)

        data = torch.from_numpy(data).float()
        if self.flatten:
            data = data.flatten(start_dim=1, end_dim=-1)

        x, y = data[:self.sequence_steps-self.forecast_steps], data[self.sequence_steps-self.forecast_steps:]

        # generate random mask
        mask = torch.zeros(x.size(0))
        mask_idx = np.random.choice(x.size(0), self.masking_steps, replace=False)
        mask[mask_idx] = 1

        return x, y, mask


class DiffusionReactionDataset(Dataset):
    '''
    Dataset for 2D diffusion reaction data from PEDBench.
    '''

    def __init__(self, sequence_steps=15, forecast_steps=5, masking_steps=5, dilation=4):
        super().__init__()
        self.path = '/homes/yx723/b/Datasets/2d-diffusion-reaction/2D_diff-react_NA_NA.h5'
        self.sequence_steps = sequence_steps
        self.forecast_steps = forecast_steps
        self.masking_steps = masking_steps
        self.dilation = dilation
        self.h5file = h5py.File(self.path, 'r')
        self.names = list(self.h5file.keys())

        self.min_vals = np.array([-0.74, -0.40]).reshape(1, 1, 1, 2)
        self.max_vals = np.array([0.74, 0.34]).reshape(1, 1, 1, 2)

        self.unit_seuqence_num = (self.h5file[f'{self.names[0]}/data'].shape[0] - 1) - self.dilation * (self.sequence_steps - 1)
        self.total_sequence_num = self.unit_seuqence_num * len(self.names)

    def __len__(self):
        return self.total_sequence_num

    def __getitem__(self, index):
        batch_idx = index // self.unit_seuqence_num
        seq_start_idx = index % self.unit_seuqence_num + 1
        data = self.h5file[f'{self.names[batch_idx]}/data'][seq_start_idx: seq_start_idx + self.sequence_steps * self.dilation: self.dilation]

        data = normalise(data, self.min_vals, self.max_vals)
        data = torch.from_numpy(data).float().permute(0, 3, 1, 2)

        x, y = data[:self.sequence_steps-self.forecast_steps], data[self.sequence_steps-self.forecast_steps:]

        # generate random mask
        mask = torch.zeros(x.size(0))
        mask_idx = np.random.choice(x.size(0), self.masking_steps, replace=False)
        mask[mask_idx] = 1

        return x, y, mask


if __name__ == '__main__':
    ### ShallowWater
    # dataset = ShallowWaterDataset(split='train')
    # print(len(dataset))  # size
    # print(dataset.min_vals.squeeze())
    # print(dataset.max_vals.squeeze())
    # x, y, mask = dataset[-1]
    # print(x.shape, y.shape)
    # print(mask)

    ### 2d diffusion reaction
    dataset = DiffusionReactionDataset()
    print(len(dataset))
    x, y, mask = dataset[0]
    print(x.shape, y.shape)
    print(mask)

    from data.utils import interpolate_sequence, visualise_sequence
    visualise_sequence(x, save_path='../sequence.png')
    x_int = interpolate_sequence(x, mask)
    print((x - x_int).numpy().max(axis=(1, 2, 3)))
