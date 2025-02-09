import os
import torch
from torch import optim
import torch.nn.functional as F
import lightning.pytorch as pl
from models.convlstm import ConvLSTMForecaster
from data.utils import interpolate_sequence, visualise_sequence, calculate_ssim_series, calculate_psnr_series, calculate_image_level_mse_std


class LitConvLSTM(pl.LightningModule):
    def __init__(self, dataset):
        super().__init__()
        self.model = ConvLSTMForecaster(input_dim=1, hidden_dim=1, kernel_size=3, num_layers=1)
        self.dataset = dataset
        self.forecast_steps = 5
        self.visualise_num = 5

    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(), lr=1e-3, weight_decay=1e-2)
        return optimizer

    def compute_loss(self, x, pred):
        full_state_loss = F.mse_loss(pred, x)
        loss = full_state_loss
        return loss, full_state_loss

    def training_step(self, batch, batch_idx):
        x, y, mask = batch
        data = torch.cat([x, y], dim=1)

        x_int = x.clone()
        for i in range(len(x_int)):
            x_int[i] = interpolate_sequence(x_int[i], mask[i])

        pred = self.model(x_int, y)
        loss, full_state_loss = self.compute_loss(data, pred)

        self.log('train/loss', loss)
        self.log('train/mse', full_state_loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y, mask = batch
        data = torch.cat([x, y], dim=1)

        x_int = x.clone()
        for i in range(len(x_int)):
            x_int[i] = interpolate_sequence(x_int[i], mask[i])

        with torch.no_grad():
            pred = self.model.predict(x_int, self.forecast_steps)
            loss, full_state_loss = self.compute_loss(data, pred)

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
            loss, full_state_loss = self.compute_loss(data, pred)
            mse_value, mse_std = calculate_image_level_mse_std(data, pred)
            ssim_value, ssim_std = calculate_ssim_series(data, pred)
            psnr_value, psnr_std = calculate_psnr_series(data, pred)

        self.log('test/loss', loss)
        self.log('test/mse', full_state_loss)
        self.log('test/ssim', ssim_value)
        self.log('test/psnr', psnr_value)

        self.log('test/mse_std', mse_std)
        self.log('test/ssim_std', ssim_std)
        self.log('test/psnr_std', psnr_std)
        return loss

    def predict_step(self, batch, batch_idx):
        x, y, mask = batch

        x_int = x.clone()
        for i in range(len(x_int)):
            x_int[i] = interpolate_sequence(x_int[i], mask[i])

        batch_size = len(x)
        os.makedirs('logs/convlstm/output', exist_ok=True)

        with torch.no_grad():
            pred = self.model.predict(x_int, self.forecast_steps)

        for i in range(batch_size):
            vi = batch_idx * batch_size + i
            if vi >= self.visualise_num:
                break
            input_ = torch.cat([x[i], y[i]], dim=0)
            output = pred[i]
            diff = torch.abs(input_ - output)
            visualise_sequence(input_, save_path=f'logs/convlstm/output/input_{vi}.png')
            visualise_sequence(output, save_path=f'logs/convlstm/output/predict_{vi}.png')
            visualise_sequence(diff, save_path=f'logs/convlstm/output/diff_{vi}.png')
        return pred
