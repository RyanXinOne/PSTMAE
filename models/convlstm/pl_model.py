import os
import torch
from torch import optim
import torch.nn.functional as F
import lightning.pytorch as pl
from models.convlstm import ConvLSTMForecaster
from data.dataset import ShallowWaterDataset


class LitConvLSTM(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.model = ConvLSTMForecaster(input_dim=3, hidden_dim=6, kernel_size=3, num_layers=2)
        self.lr = 1e-3
        self.forecast_steps = 5
        self.visualise_num = 5

    def training_step(self, batch, batch_idx):
        x, y, mask = batch
        for i in range(len(x)):
            x[i] = ShallowWaterDataset.interpolate(x[i], mask[i])

        x_pred, y_pred = self.model(x, y)
        loss = self.compute_loss(
            torch.cat([x, y], dim=1),
            torch.cat([x_pred, y_pred], dim=1),
        )
        self.log('train/mse', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y, mask = batch
        for i in range(len(x)):
            x[i] = ShallowWaterDataset.interpolate(x[i], mask[i])

        with torch.no_grad():
            x_pred, y_pred = self.model.predict(x, self.forecast_steps)
            loss = self.compute_loss(
                torch.cat([x, y], dim=1),
                torch.cat([x_pred, y_pred], dim=1),
            )
        self.log('val/mse', loss)
        return loss

    def test_step(self, batch, batch_idx):
        x, y, mask = batch
        for i in range(len(x)):
            x[i] = ShallowWaterDataset.interpolate(x[i], mask[i])

        with torch.no_grad():
            x_pred, y_pred = self.model.predict(x, self.forecast_steps)
            loss = self.compute_loss(
                torch.cat([x, y], dim=1),
                torch.cat([x_pred, y_pred], dim=1),
            )
        self.log('test/mse', loss)
        return loss

    def predict_step(self, batch, batch_idx):
        x, y, mask = batch
        for i in range(len(x)):
            x[i] = ShallowWaterDataset.interpolate(x[i], mask[i])

        batch_size = len(x)
        os.makedirs('logs/convlstm/output', exist_ok=True)

        with torch.no_grad():
            x_pred, y_pred = self.model.predict(x, self.forecast_steps)

        for i in range(batch_size):
            if batch_idx*batch_size+i >= self.visualise_num:
                break
            ShallowWaterDataset.visualise_sequence(
                torch.cat([x[i], y[i]], dim=0),
                save_path=f'logs/convlstm/output/input_{batch_idx*batch_size+i}.png'
            )
            ShallowWaterDataset.visualise_sequence(
                torch.cat([x_pred[i], y_pred[i]], dim=0),
                save_path=f'logs/convlstm/output/predict_{batch_idx*batch_size+i}.png'
            )
        return y_pred

    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(), lr=self.lr)
        return optimizer

    def compute_loss(self, y, y_pred):
        return F.mse_loss(y_pred, y)
