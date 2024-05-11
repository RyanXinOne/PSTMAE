import torch
from torch import optim
import lightning.pytorch as pl
from timae.model import TimeSeriesMaskedAutoencoder


class LitTiMAE(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.model = TimeSeriesMaskedAutoencoder(
            embed_dim=256,
            num_heads=4,
            depth=16,
            decoder_embed_dim=256,
            decoder_num_heads=4,
            decoder_depth=8,
            d_hid=512,
            dropout=0.1,
            mask_ratio=0.,
            forecast_ratio=1.,
            forecast_steps=5
        )
        self.lr = 1e-3

    def training_step(self, batch, batch_idx):
        (loss_removed, loss_seen, forecast_loss, foreseen_loss), _ = self.model(batch)
        loss = self.compute_loss(loss_removed, loss_seen, forecast_loss, foreseen_loss)
        self.log('train/loss_removed', loss_removed)
        self.log('train/loss_seen', loss_seen)
        self.log('train/forecast_loss', forecast_loss)
        self.log('train/foreseen_loss', foreseen_loss)
        self.log('train/loss', loss)
        self.log('train/mse', forecast_loss)
        return loss

    def validation_step(self, batch, batch_idx):
        (loss_removed, loss_seen, forecast_loss, foreseen_loss), _ = self.model(batch)
        loss = self.compute_loss(loss_removed, loss_seen, forecast_loss, foreseen_loss)
        self.log('val/loss_removed', loss_removed)
        self.log('val/loss_seen', loss_seen)
        self.log('val/forecast_loss', forecast_loss)
        self.log('val/foreseen_loss', foreseen_loss)
        self.log('val/loss', loss)
        self.log('val/mse', forecast_loss)
        return loss

    def test_step(self, batch, batch_idx):
        (loss_removed, loss_seen, forecast_loss, foreseen_loss), _ = self.model(batch)
        loss = self.compute_loss(loss_removed, loss_seen, forecast_loss, foreseen_loss)
        self.log('test/loss_removed', loss_removed)
        self.log('test/loss_seen', loss_seen)
        self.log('test/forecast_loss', forecast_loss)
        self.log('test/foreseen_loss', foreseen_loss)
        self.log('test/loss', loss)
        self.log('test/mse', forecast_loss)
        return loss

    def configure_optimizers(self):
        optimizer = optim.RAdam(self.parameters(), lr=self.lr)
        return optimizer

    def compute_loss(self, loss_removed, loss_seen, forecast_loss, foreseen_loss):
        loss = forecast_loss + 0.5 * foreseen_loss
        if torch.isnan(loss).any():
            raise ValueError('NaN loss')
        return loss
