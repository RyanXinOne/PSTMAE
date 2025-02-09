import os
import torch
from torch import optim
import torch.nn.functional as F
import lightning.pytorch as pl
from models.vae import SeqConvVariationalAutoEncoder
from data.utils import visualise_sequence, calculate_ssim_series, calculate_psnr_series, calculate_image_level_mse_std


class LitVariationalAutoEncoder(pl.LightningModule):
    def __init__(self, dataset):
        super().__init__()
        self.model = SeqConvVariationalAutoEncoder(input_dim=1, latent_dim=128)
        self.dataset = dataset
        self.visualise_num = 5

    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(), lr=1e-3, weight_decay=1e-4)
        return optimizer

    def compute_loss(self, x, y, mu, logvar):
        recons_loss = F.mse_loss(y, x)
        kld_loss = torch.mean(-0.5 * torch.sum(1 + logvar - mu.square() - logvar.exp(), dim=-1))
        loss = recons_loss + 0 * kld_loss
        return loss, recons_loss, kld_loss

    def training_step(self, batch, batch_idx):
        data = batch[0]
        pred, mu, logvar = self.model(data)
        loss, recons_loss, kld_loss = self.compute_loss(data, pred, mu, logvar)
        self.log('train/loss', loss)
        self.log('train/mse', recons_loss)
        self.log('train/kld_loss', kld_loss)
        return loss

    def validation_step(self, batch, batch_idx):
        data = batch[0]
        with torch.no_grad():
            pred, mu, logvar = self.model(data)
            loss, recons_loss, kld_loss = self.compute_loss(data, pred, mu, logvar)
        self.log('val/loss', loss)
        self.log('val/mse', recons_loss)
        self.log('val/kld_loss', kld_loss)
        return loss

    def test_step(self, batch, batch_idx):
        data = batch[0]
        with torch.no_grad():
            pred, mu, logvar = self.model(data)
            loss, recons_loss, kld_loss = self.compute_loss(data, pred, mu, logvar)
            mse_value, mse_std = calculate_image_level_mse_std(data, pred)
            ssim_value, ssim_std = calculate_ssim_series(data, pred)
            psnr_value, psnr_std = calculate_psnr_series(data, pred)
        self.log('test/loss', loss)
        self.log('test/mse', recons_loss)
        self.log('test/kld_loss', kld_loss)
        self.log('test/ssim', ssim_value)
        self.log('test/psnr', psnr_value)

        self.log('test/mse_std', mse_std)
        self.log('test/ssim_std', ssim_std)
        self.log('test/psnr_std', psnr_std)
        return loss

    def predict_step(self, batch, batch_idx):
        data = batch[0]
        batch_size = len(data)
        os.makedirs('logs/vae/output', exist_ok=True)

        with torch.no_grad():
            pred, _, _ = self.model(data)

        for i in range(batch_size):
            vi = batch_idx * batch_size + i
            if vi >= self.visualise_num:
                break
            input_ = data[i]
            output = pred[i]
            diff = torch.abs(input_ - output)
            visualise_sequence(input_, save_path=f'logs/vae/output/input_{vi}.png')
            visualise_sequence(output, save_path=f'logs/vae/output/predict_{vi}.png')
            visualise_sequence(diff, save_path=f'logs/vae/output/diff_{vi}.png')
        return pred
