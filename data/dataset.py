import os
import json
import h5py
import numpy as np
import torch
from torch.utils.data import Dataset
from data.utils import generate_random_mask, normalise, unnormalise


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
        self.path = '/homes/yx723/b/Datasets/ShallowWater-simulation/res128'
        self.configs_path = '/homes/yx723/b/Datasets/ShallowWater-simulation/configs_res128.json'
        self.files = os.listdir(self.path)
        self.configs = json.load(open(self.configs_path, 'r'))
        self.sequence_steps = sequence_steps
        self.forecast_steps = forecast_steps
        self.masking_steps = masking_steps
        self.dilation = dilation

        self.min_vals = np.array([0.66, -0.17, -0.17]).reshape(1, 3, 1, 1)
        self.max_vals = np.array([1.29, 0.17, 0.17]).reshape(1, 3, 1, 1)

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

        config = self.configs[self.files[file_idx]]

        return x, y, mask, config

    @staticmethod
    def calculate_total_energy(data, min_vals, max_vals, dx=0.01, g=1.0):
        '''
        Calculate total energy for the data of sequence.

        Args:
            data (torch.Tensor): sequence data with shape ([N,] Nt, C, H, W) and C = h, u, v.

        Returns:
            torch.Tensor: total energy with shape ([N,] Nt).
        '''
        data = unnormalise(data, min_vals, max_vals)
        kinetic_energy = 0.5 * (torch.sum(data[..., 1, :, :] ** 2, axis=(-2, -1)) + torch.sum(data[..., 2, :, :] ** 2, axis=(-2, -1))) * dx**2
        potential_energy = torch.sum(0.5 * g * data[..., 0, :, :] ** 2, axis=(-2, -1)) * dx**2
        total_energy = kinetic_energy + potential_energy
        return total_energy

    @staticmethod
    def evolve_with_flow_operator(data, min_vals, max_vals, evolve_step=1, dt=0.0001, dx=0.01, g=1.0, b=0.0):
        """
        Step-wise evolve the data sequence with flow operator.

        Args:
            data (torch.Tensor): sequence data with shape ([N,] Nt, C, H, W) and C = h, u, v.

        Returns:
            torch.Tensor: evolved data with shape ([N,] Nt, C, H, W).
        """

        def dxy(A, dx, axis):
            return (torch.roll(A, -1, axis) - torch.roll(A, 1, axis)) / (dx * 2.0)

        def d_dx(A, dx):
            return dxy(A, dx, -1)

        def d_dy(A, dx):
            return dxy(A, dx, -2)

        def d_dt(h, u, v, dx, g, b):
            du_dt = -g * d_dx(h, dx) - b * u
            dv_dt = -g * d_dy(h, dx) - b * v
            dh_dt = -d_dx(u * h, dx) - d_dy(v * h, dx)
            return dh_dt, du_dt, dv_dt

        def evolve(h, u, v, dt, dx, g, b):
            dh_dt, du_dt, dv_dt = d_dt(h, u, v, dx, g, b)
            h += dh_dt * dt
            u += du_dt * dt
            v += dv_dt * dt
            return h, u, v

        data = unnormalise(data, min_vals, max_vals)
        h, u, v = data[..., 0, :, :], data[..., 1, :, :], data[..., 2, :, :]
        for _ in range(evolve_step):
            h, u, v = evolve(h, u, v, dt, dx, g, b)
        evolved_data = torch.stack([h, u, v], dim=-3)
        evolved_data = normalise(evolved_data, min_vals, max_vals)
        return evolved_data


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


class NOAASeaSurfacePressureDataset(Dataset):
    '''
    Dataset for NOAA sea surface pressure data.
    '''

    def __init__(self, sequence_steps=15, forecast_steps=5, masking_steps=5, dilation=1):
        super().__init__()
        self.path = '/homes/yx723/b/Datasets/NOAA_data/sst_weekly.mat'
        self.sequence_steps = sequence_steps
        self.forecast_steps = forecast_steps
        self.masking_steps = masking_steps
        self.dilation = dilation
        self.h5file = h5py.File(self.path, 'r')

        self.data = self.h5file["sst"][...].reshape(-1, 1, 180, 360, order='F')[:, :, ::-1, :]
        self.data_mask = ~np.isnan(self.data[0, 0])

        self.min_vals = -1.8
        self.max_vals = 36.16

        self.seuqence_num = self.data.shape[0] - self.dilation * (self.sequence_steps - 1)

    def __len__(self):
        return self.seuqence_num

    def __getitem__(self, index):
        index %= self.seuqence_num
        data = self.data[index: index + self.sequence_steps * self.dilation: self.dilation]

        data = normalise(data, self.min_vals, self.max_vals)
        data = np.nan_to_num(data)
        data = torch.from_numpy(data).float()

        x, y = data[:self.sequence_steps-self.forecast_steps], data[self.sequence_steps-self.forecast_steps:]

        mask = generate_random_mask(x.size(0), self.masking_steps)
        mask = torch.from_numpy(mask).float()

        return x, y, mask


if __name__ == '__main__':
    dataset = DummyDataset()
    # dataset = ShallowWaterDataset()
    # dataset = DiffusionReactionDataset()
    # dataset = CompressibleNavierStokesDataset()
    # dataset = NOAASeaSurfacePressureDataset()
    print(len(dataset))
    x, y, mask = dataset[0]
    print(x.shape, y.shape)
    print(mask)

    from data.utils import interpolate_sequence, visualise_sequence
    visualise_sequence(x, save_path='sequence.png')
    x_int = interpolate_sequence(x, mask)
    print((x - x_int).numpy().max(axis=(1, 2, 3)))
