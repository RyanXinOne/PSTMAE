import os
import torch
from torch import optim
import torch.nn.functional as F
import lightning.pytorch as pl
from models.timae import TimeSeriesMaskedAutoencoder
from data.utils import visualise_sequence, calculate_ssim_series, calculate_psnr_series


class LitTiMAE(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.model = TimeSeriesMaskedAutoencoder(
            input_dim=4,
            latent_dim=128,
            hidden_dim=256,
            encoder_num_heads=2,
            encoder_depth=4,
            decoder_num_heads=2,
            decoder_depth=1,
            forecast_steps=5
        )
        self.visulise_num = 5

        # load pretrained autoencoder
        self.model.autoencoder.load_pretrained_freeze()

    def configure_optimizers(self):
        optimizer = optim.RAdam(
            self.parameters(),
            lr=3e-4,
            weight_decay=0)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=10,
            eta_min=3e-4)
        return [optimizer], [scheduler]

    def compute_loss(self, x, pred, z1=None, z2=None):
        full_state_loss = F.mse_loss(pred, x)
        if z1 is None or z2 is None:
            return full_state_loss

        latent_loss = F.mse_loss(z2, z1)

        loss = full_state_loss / (torch.linalg.norm(x) / x.numel() + 1e-8) + latent_loss / (torch.linalg.norm(z1) / z1.numel() + 1e-8)
        return loss, full_state_loss, latent_loss

    def training_step(self, batch, batch_idx):
        x, y, mask = batch
        data = torch.cat([x, y], dim=1)
        z1 = self.model.autoencoder.encode(data)
        pred, z2 = self.model(x, mask)
        loss, full_state_loss, latent_loss = self.compute_loss(data, pred, z1, z2)
        self.log('train/loss', loss)
        self.log('train/mse', full_state_loss)
        self.log('train/latent_mse', latent_loss)
        self.log('train/lr', self.trainer.optimizers[0].param_groups[0]['lr'])
        loss = torch.nan_to_num(loss, nan=10.0, posinf=10.0, neginf=10.0)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y, mask = batch
        data = torch.cat([x, y], dim=1)
        with torch.no_grad():
            z1 = self.model.autoencoder.encode(data)
            pred, z2 = self.model(x, mask)
            loss, full_state_loss, latent_loss = self.compute_loss(data, pred, z1, z2)
        self.log('val/loss', loss)
        self.log('val/mse', full_state_loss)
        self.log('val/latent_mse', latent_loss)
        return loss

    def test_step(self, batch, batch_idx):
        x, y, mask = batch
        data = torch.cat([x, y], dim=1)
        with torch.no_grad():
            z1 = self.model.autoencoder.encode(data)
            pred, z2 = self.model(x, mask)
            loss, full_state_loss, latent_loss = self.compute_loss(data, pred, z1, z2)
            ssim_value = calculate_ssim_series(data, pred)
            psnr_value = calculate_psnr_series(data, pred)
        self.log('test/loss', loss)
        self.log('test/mse', full_state_loss)
        self.log('test/latent_mse', latent_loss)
        self.log('test/ssim', ssim_value)
        self.log('test/psnr', psnr_value)
        return loss

    def predict_step(self, batch, batch_idx):
        x, y, mask = batch

        batch_size = len(x)
        os.makedirs('logs/timae/output', exist_ok=True)

        with torch.no_grad():
            pred, _ = self.model(x, mask)

        for i in range(batch_size):
            vi = batch_idx * batch_size + i
            if vi >= self.visulise_num:
                break
            input_ = torch.cat([x[i], y[i]], dim=0)
            output = pred[i]
            diff = torch.abs(input_ - output)
            visualise_sequence(input_, save_path=f'logs/timae/output/input_{vi}.png')
            visualise_sequence(output, save_path=f'logs/timae/output/predict_{vi}.png')
            visualise_sequence(diff, save_path=f'logs/timae/output/diff_{vi}.png')
        return pred
