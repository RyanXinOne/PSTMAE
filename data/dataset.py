import os
import h5py
import numpy as np
import torch
from torch.utils.data import Dataset
from data.utils import generate_random_mask, normalise


class DummyDataset(Dataset):
    '''
    Dummy dataset for testing purposes.
    '''

    def __init__(self, sequence_steps=15, forecast_steps=5, masking_steps=5):
        super().__init__()
        self.sequence_steps = sequence_steps
        self.forecast_steps = forecast_steps
        self.masking_steps = masking_steps

    def __len__(self):
        return 1000

    def __getitem__(self, idx):
        x = torch.rand(self.sequence_steps - self.forecast_steps, 1, 128, 128)
        y = torch.rand(self.forecast_steps, 1, 128, 128)

        mask = generate_random_mask(x.size(0), self.masking_steps)
        mask = torch.from_numpy(mask).float()

        return x, y, mask


class ShallowWaterDataset(Dataset):
    '''
    Dataset for Shallow Water simulation data.
    '''

    def __init__(self, sequence_steps=15, forecast_steps=5, masking_steps=5, dilation=1):
        super().__init__()
        self.path = f'/homes/yx723/b/Datasets/ShallowWater-simulation/res128'
        self.files = os.listdir(self.path)
        self.sequence_steps = sequence_steps
        self.forecast_steps = forecast_steps
        self.masking_steps = masking_steps
        self.dilation = dilation

        # compute min and max values of h, u, v for normalisation
        self.min_vals = np.array([0.66, -0.17, -0.17]).reshape(1, 3, 1, 1)
        self.max_vals = np.array([1.20, 0.17, 0.17]).reshape(1, 3, 1, 1)

        # calculate number of sequences
        self.sequence_num_per_file = np.load(os.path.join(self.path, self.files[0])).shape[0] - self.dilation * (self.sequence_steps - 1)
        self.sequence_num = self.sequence_num_per_file * len(self.files)

    def __len__(self):
        return self.sequence_num

    def __getitem__(self, idx):
        file_idx = idx // self.sequence_num_per_file
        seq_start_idx = idx % self.sequence_num_per_file
        data = np.load(os.path.join(self.path, self.files[file_idx]))[seq_start_idx: seq_start_idx + self.sequence_steps * self.dilation: self.dilation]

        data = normalise(data, self.min_vals, self.max_vals)
        data = torch.from_numpy(data).float()

        x, y = data[:self.sequence_steps-self.forecast_steps], data[self.sequence_steps-self.forecast_steps:]

        mask = generate_random_mask(x.size(0), self.masking_steps)
        mask = torch.from_numpy(mask).float()

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

        mask = generate_random_mask(x.size(0), self.masking_steps)
        mask = torch.from_numpy(mask).float()

        return x, y, mask


class CompressibleNavierStokesDataset(Dataset):
    '''
    Dataset for compressible Navier-Stokes data from PEDBench.
    '''

    def __init__(self, sequence_steps=15, forecast_steps=5, masking_steps=5):
        super().__init__()
        self.path = '/homes/yx723/b/Datasets/2d-cfd/2D_CFD_Rand_M0.1_Eta0.01_Zeta0.01_periodic_128_Train/'
        self.files = os.listdir(self.path)
        self.sequence_steps = sequence_steps
        self.forecast_steps = forecast_steps
        self.masking_steps = masking_steps

        if '2D_CFD_Rand_M0.1_Eta0.01_Zeta0.01_periodic_128_Train' in self.path:
            self.min_vals = np.array([-1.56, -1.56, 0.0, 0.0]).reshape(1, 4, 1, 1)
            self.max_vals = np.array([1.56, 1.56, 39.8, 163.1]).reshape(1, 4, 1, 1)
        elif '2D_CFD_Rand_M1.0_Eta0.01_Zeta0.01_periodic_128_Train' in self.path:
            self.min_vals = np.array([-15.60, -15.60, 0.0, 0.0]).reshape(1, 4, 1, 1)
            self.max_vals = np.array([15.60, 15.60, 42.31, 715.92]).reshape(1, 4, 1, 1)
        else:
            raise ValueError("Unsupported dataset path.")

        self.unit_seuqence_num = np.load(os.path.join(self.path, self.files[0])).shape[0] - self.sequence_steps + 1
        self.total_sequence_num = self.unit_seuqence_num * len(self.files)

    def __len__(self):
        return self.total_sequence_num

    def __getitem__(self, index):
        file_idx = index // self.unit_seuqence_num
        seq_start_idx = index % self.unit_seuqence_num

        data = np.load(os.path.join(self.path, self.files[file_idx]))[seq_start_idx: seq_start_idx + self.sequence_steps]

        data = normalise(data, self.min_vals, self.max_vals)
        data = torch.from_numpy(data).float()

        x, y = data[:self.sequence_steps-self.forecast_steps], data[self.sequence_steps-self.forecast_steps:]

        mask = generate_random_mask(x.size(0), self.masking_steps)
        mask = torch.from_numpy(mask).float()

        return x, y, mask


if __name__ == '__main__':
    # dataset = DummyDataset()
    # dataset = ShallowWaterDataset()
    dataset = DiffusionReactionDataset()
    # dataset = CompressibleNavierStokesDataset()
    print(len(dataset))
    x, y, mask = dataset[0]
    print(x.shape, y.shape)
    print(mask)

    from data.utils import interpolate_sequence, visualise_sequence
    visualise_sequence(x, save_path='../sequence.png')
    x_int = interpolate_sequence(x, mask)
    print((x - x_int).numpy().max(axis=(1, 2, 3)))
