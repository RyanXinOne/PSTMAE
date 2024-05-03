import torch
from torch import optim
import torch.nn.functional as F
import lightning.pytorch as pl
from convrae.model import ConvRecurrentAutoEncoder


class LitConvRAE(pl.LightningModule):
    def __init__(self, lr=1e-3):
        super().__init__()
        self.model = ConvRecurrentAutoEncoder()
        self.lr = lr

    def training_step(self, batch, batch_idx):
        x = batch
        pred, h1, h2 = self.model(x)

        full_state_loss = F.mse_loss(pred[:-1], x[1:]) / (torch.linalg.norm(x[1:]) / x[1:].numel() + 1e-6)
        latent_loss = F.mse_loss(h2[:-1], h1[1:]) / (torch.linalg.norm(h1[1:]) / h1[1:].numel() + 1e-6)
        loss = full_state_loss + latent_loss

        self.log('train/loss', loss)
        self.log('train/full_state_loss', full_state_loss)
        self.log('train/latent_loss', latent_loss)

        return loss

    def validation_step(self, batch, batch_idx):
        x = batch
        pred, h1, h2 = self.model(x)

        full_state_loss = F.mse_loss(pred[:-1], x[1:]) / (torch.linalg.norm(x[1:]) / x[1:].numel() + 1e-6)
        latent_loss = F.mse_loss(h2[:-1], h1[1:]) / (torch.linalg.norm(h1[1:]) / h1[1:].numel() + 1e-6)
        loss = full_state_loss + latent_loss

        self.log('val/loss', loss)
        self.log('val/full_state_loss', full_state_loss)
        self.log('val/latent_loss', latent_loss)

        return loss

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.lr)
        return optimizer
