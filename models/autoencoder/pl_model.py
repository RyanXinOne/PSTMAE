import os
import torch
from torch import optim
import torch.nn.functional as F
import lightning.pytorch as pl
from models.autoencoder import SeqConvAutoEncoder
from data.utils import visualise_sequence, calculate_ssim_series, calculate_psnr_series


class LitAutoEncoder(pl.LightningModule):
    def __init__(self, dataset):
        super().__init__()
        self.model = SeqConvAutoEncoder(input_dim=1, latent_dim=128)
        self.dataset = dataset
        self.visualise_num = 5

        data_mask = torch.from_numpy(self.dataset.data_mask).float()
        self.register_buffer('data_mask', data_mask, persistent=False)

    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(), lr=1e-3, weight_decay=1e-6)
        return optimizer

    def compute_loss(self, x, y, z):
        mse_loss = F.mse_loss(y, x)
        reg_loss = torch.mean(z**2)
        loss = mse_loss + 1e-6 * reg_loss
        return loss, mse_loss, reg_loss

    def training_step(self, batch, batch_idx):
        data = batch[0]
        pred, z = self.model(data)
        pred = pred * self.data_mask
        loss, mse_loss, reg_loss = self.compute_loss(data, pred, z)
        self.log('train/loss', loss)
        self.log('train/mse', mse_loss)
        self.log('train/reg_loss', reg_loss)
        return loss

    def validation_step(self, batch, batch_idx):
        data = batch[0]
        with torch.no_grad():
            pred, z = self.model(data)
            pred = pred * self.data_mask
            loss, mse_loss, reg_loss = self.compute_loss(data, pred, z)
        self.log('val/loss', loss)
        self.log('val/mse', mse_loss)
        self.log('val/reg_loss', reg_loss)
        return loss

    def test_step(self, batch, batch_idx):
        data = batch[0]
        with torch.no_grad():
            pred, z = self.model(data)
            pred = pred * self.data_mask
            loss, mse_loss, reg_loss = self.compute_loss(data, pred, z)
            ssim_value = calculate_ssim_series(data, pred)
            psnr_value = calculate_psnr_series(data, pred)
        self.log('test/loss', loss)
        self.log('test/mse', mse_loss)
        self.log('test/reg_loss', reg_loss)
        self.log('test/ssim', ssim_value)
        self.log('test/psnr', psnr_value)
        return loss

    def predict_step(self, batch, batch_idx):
        data = batch[0]
        batch_size = len(data)
        os.makedirs('logs/autoencoder/output', exist_ok=True)

        with torch.no_grad():
            pred, _ = self.model(data)
            pred = pred * self.data_mask

        for i in range(batch_size):
            vi = batch_idx * batch_size + i
            if vi >= self.visualise_num:
                break
            input_ = data[i]
            output = pred[i]
            diff = torch.abs(input_ - output)
            visualise_sequence(input_, save_path=f'logs/autoencoder/output/input_{vi}.png')
            visualise_sequence(output, save_path=f'logs/autoencoder/output/predict_{vi}.png')
            visualise_sequence(diff, save_path=f'logs/autoencoder/output/diff_{vi}.png')
        return pred
