from torch import optim
import lightning.pytorch as pl
from timae.nn.model import TimeSeriesMaskedAutoencoder


class LitTiMAE(pl.LightningModule):
    def __init__(self, in_chans, lr=1e-3):
        super().__init__()
        self.model = TimeSeriesMaskedAutoencoder(in_chans)

        self.lr = lr
        self.save_hyperparameters()

    def training_step(self, batch, batch_idx):
        losses, pred = self.model(batch)

        loss_removed, loss_seen, forecast_loss, backcast_loss = losses
        loss = loss_removed + 0.75 * forecast_loss + 0.5 * loss_seen + 0.2 * backcast_loss

        self.log("train/loss_removed", loss_removed, sync_dist=True)
        self.log("train/loss_seen", loss_seen, sync_dist=True)
        self.log("train/forecast_loss", forecast_loss, sync_dist=True)
        self.log("train/backcast_loss", backcast_loss, sync_dist=True)
        self.log("train/loss", loss, sync_dist=True)

        return loss

    def validation_step(self, batch, batch_idx):
        losses, pred = self.model(batch)

        loss_removed, loss_seen, forecast_loss, backcast_loss = losses
        loss = loss_removed + 0.75 * forecast_loss + 0.5 * loss_seen + 0.2 * backcast_loss

        self.log("eval/loss_removed", loss_removed, sync_dist=True)
        self.log("eval/loss_seen", loss_seen, sync_dist=True)
        self.log("eval/forecast_loss", forecast_loss, sync_dist=True)
        self.log("eval/backcast_loss", backcast_loss, sync_dist=True)
        self.log("eval/loss", loss, sync_dist=True)

        return loss

    def configure_optimizers(self):
        optimizer = optim.RAdam(self.parameters(), lr=self.lr)
        return optimizer
