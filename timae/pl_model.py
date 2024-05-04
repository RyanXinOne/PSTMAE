from torch import optim
import lightning.pytorch as pl
from timae.model import TimeSeriesMaskedAutoencoder


class LitTiMAE(pl.LightningModule):
    def __init__(self, input_dim):
        super().__init__()
        self.model = TimeSeriesMaskedAutoencoder(input_dim)
        self.lr = 1e-3

    def training_step(self, batch, batch_idx):
        (loss_removed, loss_seen, forecast_loss, backcast_loss), _ = self.model(batch)

        loss = loss_removed + 0.75 * forecast_loss + 0.5 * loss_seen + 0.2 * backcast_loss

        self.log("train/loss_removed", loss_removed)
        self.log("train/loss_seen", loss_seen)
        self.log("train/forecast_loss", forecast_loss)
        self.log("train/backcast_loss", backcast_loss)
        self.log("train/loss", loss)

        return loss

    def validation_step(self, batch, batch_idx):
        (loss_removed, loss_seen, forecast_loss, backcast_loss), _ = self.model(batch)

        loss = loss_removed + 0.75 * forecast_loss + 0.5 * loss_seen + 0.2 * backcast_loss

        self.log("val/loss_removed", loss_removed)
        self.log("val/loss_seen", loss_seen)
        self.log("val/forecast_loss", forecast_loss)
        self.log("val/backcast_loss", backcast_loss)
        self.log("val/loss", loss)

        return loss

    def configure_optimizers(self):
        optimizer = optim.RAdam(self.parameters(), lr=self.lr)
        return optimizer
