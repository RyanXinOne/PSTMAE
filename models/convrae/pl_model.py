import os
import torch
from torch import optim
import torch.nn.functional as F
import lightning.pytorch as pl
from models.convrae import ConvRAE
from models.autoencoder.pl_model import LitAutoEncoder
from data.dataset import ShallowWaterDataset


class LitConvRAE(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.model = ConvRAE(input_dim=3, latent_dim=512, hidden_dim=512)
        self.forecast_steps = 5
        self.visualise_num = 5

        # load pretrained autoencoder
        state_dict = LitAutoEncoder.load_from_checkpoint('logs/autoencoder/lightning_logs/prod/checkpoints/epoch=49-step=14950.ckpt').model.state_dict()
        self.model.autoencoder.load_state_dict(state_dict)
        # freeze autoencoder
        for param in self.model.autoencoder.parameters():
            param.requires_grad = False

    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(), lr=1e-3, weight_decay=1e-2)
        return optimizer

    def training_step(self, batch, batch_idx):
        self.model.autoencoder.eval()
        x, y, mask = batch
        for i in range(len(x)):
            x[i] = ShallowWaterDataset.interpolate(x[i], mask[i])

        x_pred, y_pred, zx_1, zx_2, zy_1, zy_2 = self.model(x, y)
        loss, full_state_loss, latent_loss = self.compute_loss(
            torch.cat([x, y], dim=1),
            torch.cat([x_pred, y_pred], dim=1),
            torch.cat([zx_1, zy_1], dim=1),
            torch.cat([zx_2, zy_2], dim=1),
        )
        self.log('train/loss', loss)
        self.log('train/mse', full_state_loss)
        self.log('train/latent_mse', latent_loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y, mask = batch
        for i in range(len(x)):
            x[i] = ShallowWaterDataset.interpolate(x[i], mask[i])

        with torch.no_grad():
            x_pred, y_pred = self.model.predict(x, self.forecast_steps)
            full_state_loss = self.compute_loss(
                torch.cat([x, y], dim=1),
                torch.cat([x_pred, y_pred], dim=1),
            )
        self.log('val/mse', full_state_loss)
        return full_state_loss

    def test_step(self, batch, batch_idx):
        x, y, mask = batch
        for i in range(len(x)):
            x[i] = ShallowWaterDataset.interpolate(x[i], mask[i])

        with torch.no_grad():
            x_pred, y_pred = self.model.predict(x, self.forecast_steps)
            full_state_loss = self.compute_loss(
                torch.cat([x, y], dim=1),
                torch.cat([x_pred, y_pred], dim=1),
            )
        self.log('test/mse', full_state_loss)
        return full_state_loss

    def predict_step(self, batch, batch_idx):
        x, y, mask = batch
        for i in range(len(x)):
            x[i] = ShallowWaterDataset.interpolate(x[i], mask[i])

        batch_size = len(x)
        os.makedirs('logs/convrae/output', exist_ok=True)

        with torch.no_grad():
            x_pred, y_pred = self.model.predict(x, self.forecast_steps)

        for i in range(batch_size):
            if batch_idx*batch_size+i >= self.visualise_num:
                break
            ShallowWaterDataset.visualise_sequence(
                torch.cat([x[i], y[i]], dim=0),
                save_path=f'logs/convrae/output/input_{batch_idx*batch_size+i}.png'
            )
            ShallowWaterDataset.visualise_sequence(
                torch.cat([x_pred[i], y_pred[i]], dim=0),
                save_path=f'logs/convrae/output/predict_{batch_idx*batch_size+i}.png'
            )
        return y_pred

    def compute_loss(self, x, pred, z1=None, z2=None):
        full_state_loss = F.mse_loss(pred, x)
        if z1 is None or z2 is None:
            return full_state_loss

        latent_loss = F.mse_loss(z2, z1)

        loss = full_state_loss / (torch.linalg.norm(x) / x.numel() + 1e-8) + latent_loss / (torch.linalg.norm(z1) / z1.numel() + 1e-8)
        return loss, full_state_loss, latent_loss
