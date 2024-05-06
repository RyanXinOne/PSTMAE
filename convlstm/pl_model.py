from torch import optim
import torch.nn.functional as F
import lightning.pytorch as pl
from convlstm.model import ConvLSTMForecaster


class LitConvLSTM(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.model = ConvLSTMForecaster(input_dim=3, hidden_dim=32, kernel_size=3, num_layers=2)
        self.lr = 1e-3
        self.forecast_steps = 5

    def training_step(self, batch, batch_idx):
        x, y = batch[:, :-self.forecast_steps], batch[:, -self.forecast_steps:]
        y_pred = self.model(x, y)
        loss = F.mse_loss(y_pred, y)
        self.log('train/loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch[:, :-self.forecast_steps], batch[:, -self.forecast_steps:]
        y_pred = self.model.predict(x, self.forecast_steps)
        loss = F.mse_loss(y_pred, y)
        self.log('val/loss', loss)
        return loss

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.lr)
        return optimizer
