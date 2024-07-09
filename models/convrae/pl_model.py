import os
import torch
from torch import optim
import torch.nn.functional as F
import lightning.pytorch as pl
from models.convrae import ConvRAE
from data.utils import interpolate_sequence, visualise_sequence, calculate_ssim_series, calculate_psnr_series


class LitConvRAE(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.model = ConvRAE(input_dim=1, latent_dim=128, hidden_dim=128)
        self.forecast_steps = 5
        self.visualise_num = 5

        # load pretrained autoencoder
        self.model.autoencoder.load_pretrained_freeze()

    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(), lr=1e-3, weight_decay=0)
        return optimizer

    def compute_loss(self, x, pred, z1=None, z2=None):
        full_state_loss = F.mse_loss(pred, x)
        if z1 is None or z2 is None:
            latent_loss = 0
        else:
            latent_loss = F.mse_loss(z2, z1)
        loss = full_state_loss + 0.5 * latent_loss
        return loss, full_state_loss, latent_loss

    def training_step(self, batch, batch_idx):
        x, y, mask = batch
        data = torch.cat([x, y], dim=1)
        z1 = self.model.autoencoder.encode(data)

        x_int = x.clone()
        for i in range(len(x_int)):
            x_int[i] = interpolate_sequence(x_int[i], mask[i])

        pred, z = self.model(x_int, y)
        loss, full_state_loss, latent_loss = self.compute_loss(data, pred, z1, z)
        self.log('train/loss', loss)
        self.log('train/mse', full_state_loss)
        self.log('train/latent_mse', latent_loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y, mask = batch
        data = torch.cat([x, y], dim=1)

        x_int = x.clone()
        for i in range(len(x_int)):
            x_int[i] = interpolate_sequence(x_int[i], mask[i])

        with torch.no_grad():
            pred = self.model.predict(x_int, self.forecast_steps)
            loss, full_state_loss, _ = self.compute_loss(data, pred)
        self.log('val/loss', loss)
        self.log('val/mse', full_state_loss)
        return loss

    def test_step(self, batch, batch_idx):
        x, y, mask = batch
        data = torch.cat([x, y], dim=1)

        x_int = x.clone()
        for i in range(len(x_int)):
            x_int[i] = interpolate_sequence(x_int[i], mask[i])

        with torch.no_grad():
            pred = self.model.predict(x_int, self.forecast_steps)
            loss, full_state_loss, _ = self.compute_loss(data, pred)
            ssim_value = calculate_ssim_series(data, pred)
            psnr_value = calculate_psnr_series(data, pred)
        self.log('test/loss', loss)
        self.log('test/mse', full_state_loss)
        self.log('test/ssim', ssim_value)
        self.log('test/psnr', psnr_value)
        return loss

    def predict_step(self, batch, batch_idx):
        x, y, mask = batch

        x_int = x.clone()
        for i in range(len(x_int)):
            x_int[i] = interpolate_sequence(x_int[i], mask[i])

        batch_size = len(x)
        os.makedirs('logs/convrae/output', exist_ok=True)

        with torch.no_grad():
            pred = self.model.predict(x_int, self.forecast_steps)

        for i in range(batch_size):
            vi = batch_idx * batch_size + i
            if vi >= self.visualise_num:
                break
            input_ = torch.cat([x[i], y[i]], dim=0)
            output = pred[i]
            diff = torch.abs(input_ - output)
            visualise_sequence(input_, save_path=f'logs/convrae/output/input_{vi}.png')
            visualise_sequence(output, save_path=f'logs/convrae/output/predict_{vi}.png')
            visualise_sequence(diff, save_path=f'logs/convrae/output/diff_{vi}.png')
        return pred
