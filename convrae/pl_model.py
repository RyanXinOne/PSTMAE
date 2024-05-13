import os
import torch
from torch import optim
import torch.nn.functional as F
import lightning.pytorch as pl
from convrae.model import ConvRAE
from data.dataset import ShallowWaterDataset


class LitConvRAE(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.model = ConvRAE(input_dim=3, latent_dim=128, hidden_dim=128)
        self.lr = 1e-3
        self.forecast_steps = 5
        self.visualise_num = 5

    def training_step(self, batch, batch_idx):
        x, y = batch[:, :-self.forecast_steps], batch[:, -self.forecast_steps:]
        y_pred, z1, z2 = self.model(x, y)
        loss, full_state_loss, latent_loss = self.compute_loss(y, y_pred, z1, z2)
        self.log('train/loss', loss)
        self.log('train/full_state_loss', full_state_loss)
        self.log('train/latent_loss', latent_loss)
        self.log('train/mse', full_state_loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch[:, :-self.forecast_steps], batch[:, -self.forecast_steps:]
        with torch.no_grad():
            y_pred = self.model.predict(x, self.forecast_steps)
            full_state_loss = self.compute_loss(y, y_pred)
        self.log('val/full_state_loss', full_state_loss)
        self.log('val/mse', full_state_loss)
        return full_state_loss

    def test_step(self, batch, batch_idx):
        x, y = batch[:, :-self.forecast_steps], batch[:, -self.forecast_steps:]
        with torch.no_grad():
            y_pred = self.model.predict(x, self.forecast_steps)
            full_state_loss = self.compute_loss(y, y_pred)
        self.log('test/full_state_loss', full_state_loss)
        self.log('test/mse', full_state_loss)
        return full_state_loss

    def predict_step(self, batch, batch_idx):
        batch_size = len(batch)
        os.makedirs('logs/convrae/output', exist_ok=True)

        x = batch[:, :-self.forecast_steps]
        with torch.no_grad():
            y_pred = self.model.predict(x, self.forecast_steps)

        for i in range(batch_size):
            if batch_idx*batch_size+i >= self.visualise_num:
                break
            data = torch.cat([x[i], y_pred[i]], dim=0)
            ShallowWaterDataset.visualise_sequence(data, f'logs/convrae/output/predict_{batch_idx*batch_size+i}.png')

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.lr)
        return optimizer

    def compute_loss(self, y, y_pred, z1=None, z2=None):
        full_state_loss = F.mse_loss(y_pred, y)
        if z1 is None or z2 is None:
            return full_state_loss

        latent_loss = F.mse_loss(z2, z1)

        loss = full_state_loss / (torch.linalg.norm(y) / y.numel() + 1e-8) + latent_loss / (torch.linalg.norm(z1) / z1.numel() + 1e-8)
        return loss, full_state_loss, latent_loss
