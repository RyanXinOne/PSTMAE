import random
import numpy as np
from skimage.metrics import structural_similarity, peak_signal_noise_ratio
import matplotlib.pyplot as plt
import torch.nn.functional as F


def generate_random_mask(seq_len, masking_steps):
    '''
    Generate a random mask for a sequence of data. 0 is observed, 1 is missing.

    Args:
        seq_len: int, the length of the sequence.
        masking_steps: int or list, the number of steps to mask.
    '''
    if isinstance(masking_steps, int):
        masking_steps = [masking_steps]
    steps = random.choice(masking_steps)

    mask = np.zeros(seq_len)
    mask_idx = np.random.choice(seq_len, steps, replace=False)
    mask[mask_idx] = 1
    return mask


def normalise(data, min_vals, max_vals):
    '''
    Normalise data into the range [0, 1].
    '''
    data = (data - min_vals) / (max_vals - min_vals)
    return data


def unnormalise(data, min_vals, max_vals):
    '''
    Unnormalise data from the range [0, 1].
    '''
    data = data * (max_vals - min_vals) + min_vals
    return data


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

    fig, axs = plt.subplots(n_channels, seq_len, squeeze=False, figsize=(seq_len, n_channels))
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


def interpolate_sequence(data, mask):
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


def calculate_ssim_series(input_sequence, predicted_sequence):
    '''
    Calculate the mean and std SSIM value for a sequence of images.

    Args:
    - input_sequence (torch.Tensor of shape (b, l, c, h, w)): The input sequence.
    - predicted_sequence (torch.Tensor of shape (b, l, c, h, w)): The predicted sequence.
    '''
    input_sequence, predicted_sequence = input_sequence.cpu().numpy(), predicted_sequence.cpu().numpy()
    ssim_values = []
    for b in range(input_sequence.shape[0]):
        for l in range(input_sequence.shape[1]):
            ssim_value = structural_similarity(input_sequence[b, l], predicted_sequence[b, l], data_range=1, channel_axis=0)
            ssim_values.append(ssim_value)
    return np.mean(ssim_values), np.std(ssim_values)


def calculate_psnr_series(input_sequence, predicted_sequence):
    '''
    Calculate the mean and std PSNR value for a sequence of images.

    Args:
    - input_sequence (torch.Tensor of shape (b, l, c, h, w)): The input sequence.
    - predicted_sequence (torch.Tensor of shape (b, l, c, h, w)): The predicted sequence.
    '''
    input_sequence, predicted_sequence = input_sequence.cpu().numpy(), predicted_sequence.cpu().numpy()
    psnr_values = []
    for b in range(input_sequence.shape[0]):
        for l in range(input_sequence.shape[1]):
            psnr_value = peak_signal_noise_ratio(input_sequence[b, l], predicted_sequence[b, l], data_range=1)
            psnr_values.append(psnr_value)
    return np.mean(psnr_values), np.std(psnr_values)


def calculate_image_level_mse_std(input_sequence, predicted_sequence):
    '''
    Calculate the mean and std MSE value for a sequence of images.

    Args:
    - input_sequence (torch.Tensor of shape (b, l, c, h, w)): The input sequence.
    - predicted_sequence (torch.Tensor of shape (b, l, c, h, w)): The predicted sequence.
    '''
    mse_values = []
    for b in range(input_sequence.shape[0]):
        for l in range(input_sequence.shape[1]):
            mse_value = F.mse_loss(predicted_sequence[b, l], input_sequence[b, l])
            mse_values.append(mse_value.cpu().numpy())
    return np.mean(mse_values), np.std(mse_values)
