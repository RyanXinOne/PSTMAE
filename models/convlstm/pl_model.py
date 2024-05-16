import os
import torch
from torch import optim
import torch.nn.functional as F
import lightning.pytorch as pl
from models.convlstm.model import ConvLSTMForecaster
from data.dataset import ShallowWaterDataset


class LitConvLSTM(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.model = ConvLSTMForecaster(input_dim=3, hidden_dim=6, kernel_size=3, num_layers=2)
        self.lr = 1e-3
        self.forecast_steps = 5
        self.visualise_num = 10

    def training_step(self, batch, batch_idx):
        x, y = batch[:, :-self.forecast_steps], batch[:, -self.forecast_steps:]
        y_pred = self.model(x, y)
        loss = F.mse_loss(y_pred, y)
        self.log('train/loss', loss)
        self.log('train/mse', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch[:, :-self.forecast_steps], batch[:, -self.forecast_steps:]
        with torch.no_grad():
            y_pred = self.model.predict(x, self.forecast_steps)
            loss = F.mse_loss(y_pred, y)
        self.log('val/loss', loss)
        self.log('val/mse', loss)
        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch[:, :-self.forecast_steps], batch[:, -self.forecast_steps:]
        with torch.no_grad():
            y_pred = self.model.predict(x, self.forecast_steps)
            loss = F.mse_loss(y_pred, y)
        self.log('test/loss', loss)
        self.log('test/mse', loss)
        return loss

    def predict_step(self, batch, batch_idx):
        batch_size = len(batch)
        os.makedirs('logs/convlstm/output', exist_ok=True)

        x = batch[:, :-self.forecast_steps]
        with torch.no_grad():
            y_pred = self.model.predict(x, self.forecast_steps)

        for i in range(batch_size):
            if batch_idx*batch_size+i >= self.visualise_num:
                break
            data = torch.cat([x[i], y_pred[i]], dim=0)
            ShallowWaterDataset.visualise_sequence(data, f'logs/convlstm/output/predict_{batch_idx*batch_size+i}.png')
        return y_pred

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.lr)
        return optimizer
