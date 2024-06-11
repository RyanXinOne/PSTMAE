import os
import torch
from torch import optim
import torch.nn.functional as F
import lightning.pytorch as pl
from models.autoencoder import SeqConvAutoEncoder
from models.common import calculate_ssim_series, calculate_psnr_series
from data.dataset import ShallowWaterDataset


class LitAutoEncoder(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.model = SeqConvAutoEncoder(input_dim=3, latent_dim=128)
        self.visualise_num = 5

    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(), lr=1e-3, weight_decay=1e-4)
        return optimizer

    def training_step(self, batch, batch_idx):
        pred, z = self.model(batch[0])
        loss, mse_loss, reg_loss = self.compute_loss(batch[0], pred, z)
        self.log('train/loss', loss)
        self.log('train/mse', mse_loss)
        self.log('train/reg_loss', reg_loss)
        return loss

    def validation_step(self, batch, batch_idx):
        with torch.no_grad():
            pred, z = self.model(batch[0])
            loss, mse_loss, reg_loss = self.compute_loss(batch[0], pred, z)
        self.log('val/loss', loss)
        self.log('val/mse', mse_loss)
        self.log('val/reg_loss', reg_loss)
        return loss

    def test_step(self, batch, batch_idx):
        with torch.no_grad():
            pred, z = self.model(batch[0])
            loss, mse_loss, reg_loss = self.compute_loss(batch[0], pred, z)
            ssim_value = calculate_ssim_series(batch[0], pred)
            psnr_value = calculate_psnr_series(batch[0], pred)
        self.log('test/loss', loss)
        self.log('test/mse', mse_loss)
        self.log('test/reg_loss', reg_loss)
        self.log('test/ssim', ssim_value)
        self.log('test/psnr', psnr_value)
        return loss

    def predict_step(self, batch, batch_idx):
        batch_size = len(batch[0])
        os.makedirs('logs/autoencoder/output', exist_ok=True)

        with torch.no_grad():
            pred, _ = self.model(batch[0])

        for i in range(batch_size):
            vi = batch_idx * batch_size + i
            if vi >= self.visualise_num:
                break
            input_ = batch[0][i]
            output = pred[i]
            diff = torch.abs(input_ - output)
            ShallowWaterDataset.visualise_sequence(input_, save_path=f'logs/autoencoder/output/input_{vi}.png')
            ShallowWaterDataset.visualise_sequence(output, save_path=f'logs/autoencoder/output/predict_{vi}.png')
            ShallowWaterDataset.visualise_sequence(diff, save_path=f'logs/autoencoder/output/diff_{vi}.png')
        return pred

    def compute_loss(self, x, y, z):
        mse_loss = F.mse_loss(y, x)
        reg_loss = torch.mean(z**2)
        loss = mse_loss + 0 * reg_loss
        return loss, mse_loss, reg_loss
