import os
import torch
from torch import optim
import lightning.pytorch as pl
from timae.model import TimeSeriesMaskedAutoencoder
from data.dataset import ShallowWaterDataset


class LitTiMAE(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.img_size = (3, 64, 64)
        self.model = TimeSeriesMaskedAutoencoder(
            img_size=self.img_size,
            embed_dim=512,
            num_heads=4,
            depth=16,
            decoder_embed_dim=512,
            decoder_num_heads=4,
            decoder_depth=4,
            d_hid=1024,
            dropout=0.1,
            mask_ratio=0.,
            forecast_ratio=1.,
            forecast_steps=5
        )
        self.lr = 1e-3
        self.visulise_num = 10

    def training_step(self, batch, batch_idx):
        (loss_removed, loss_seen, forecast_loss, foreseen_loss), _, _ = self.model(batch)
        loss = self.compute_loss(loss_removed, loss_seen, forecast_loss, foreseen_loss)
        self.log('train/loss_removed', loss_removed)
        self.log('train/loss_seen', loss_seen)
        self.log('train/forecast_loss', forecast_loss)
        self.log('train/foreseen_loss', foreseen_loss)
        self.log('train/loss', loss)
        self.log('train/mse', forecast_loss)
        return loss

    def validation_step(self, batch, batch_idx):
        with torch.no_grad():
            (loss_removed, loss_seen, forecast_loss, foreseen_loss), _, _ = self.model(batch)
            loss = self.compute_loss(loss_removed, loss_seen, forecast_loss, foreseen_loss)
        self.log('val/loss_removed', loss_removed)
        self.log('val/loss_seen', loss_seen)
        self.log('val/forecast_loss', forecast_loss)
        self.log('val/foreseen_loss', foreseen_loss)
        self.log('val/loss', loss)
        self.log('val/mse', forecast_loss)
        return loss

    def test_step(self, batch, batch_idx):
        with torch.no_grad():
            (loss_removed, loss_seen, forecast_loss, foreseen_loss), _, _ = self.model(batch)
            loss = self.compute_loss(loss_removed, loss_seen, forecast_loss, foreseen_loss)
        self.log('test/loss_removed', loss_removed)
        self.log('test/loss_seen', loss_seen)
        self.log('test/forecast_loss', forecast_loss)
        self.log('test/foreseen_loss', foreseen_loss)
        self.log('test/loss', loss)
        self.log('test/mse', forecast_loss)
        return loss

    def predict_step(self, batch, batch_idx):
        batch_size = len(batch)
        os.makedirs('logs/timae/output', exist_ok=True)

        with torch.no_grad():
            _, preds, masks = self.model(batch)
        inv_masks = (masks - 1) ** 2
        masks, inv_masks = masks.bool(), inv_masks.bool()
        for i in range(batch_size):
            if batch_idx*batch_size+i >= self.visulise_num:
                break
            x, pred, mask, inv_mask = batch[i], preds[i], masks[i], inv_masks[i]
            data = torch.zeros_like(x)
            data[inv_mask] = x[inv_mask]
            data[mask] = pred[mask]
            data = data.unflatten(1, self.img_size)
            ShallowWaterDataset.visualise_sequence(data, f'logs/timae/output/predict_{batch_idx*batch_size+i}.png')
        return preds

    def configure_optimizers(self):
        optimizer = optim.RAdam(self.parameters(), lr=self.lr)
        return optimizer

    def compute_loss(self, loss_removed, loss_seen, forecast_loss, foreseen_loss):
        loss = forecast_loss + 0.5 * foreseen_loss
        if torch.isnan(loss).any():
            raise ValueError('NaN loss')
        return loss
