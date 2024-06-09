from torch import nn
import numpy as np
from skimage.metrics import structural_similarity, peak_signal_noise_ratio


class SeqConv2d(nn.Conv2d):
    '''
    Conv2d for sequence data.
    '''

    def forward(self, x):
        '''
        input shape: (b, l, c, h, w)
        '''
        b, l, c, h, w = x.size()
        x = x.reshape(-1, c, h, w)
        x = super().forward(x)
        x = x.reshape(b, l, x.size(1), x.size(2), x.size(3))
        return x


class SeqTransposeConv2d(nn.ConvTranspose2d):
    '''
    ConvTranspose2d for sequence data.
    '''

    def forward(self, x):
        '''
        input shape: (b, l, c, h, w)
        '''
        b, l, c, h, w = x.size()
        x = x.reshape(-1, c, h, w)
        x = super().forward(x)
        x = x.reshape(b, l, x.size(1), x.size(2), x.size(3))
        return x


def calculate_ssim_series(input_sequence, predicted_sequence):
    '''
    Calculate the mean SSIM value for a sequence of images.

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
    return np.mean(ssim_values)


def calculate_psnr_series(input_sequence, predicted_sequence):
    '''
    Calculate the mean PSNR value for a sequence of images.

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
    return np.mean(psnr_values)
