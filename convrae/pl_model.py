import torch
from torch import optim
import torch.nn.functional as F
import lightning.pytorch as pl
from convrae.model import ConvRAE


class LitConvRAE(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.model = ConvRAE(input_dim=3, latent_dim=128, hidden_dim=128)
        self.lr = 1e-3

    def training_step(self, batch, batch_idx):
        pred, h1, h2 = self.model(batch)

        loss, full_state_loss, latent_loss = self.compute_loss(batch, pred, h1, h2)

        self.log('train/loss', loss)
        self.log('train/full_state_loss', full_state_loss)
        self.log('train/latent_loss', latent_loss)

        return loss

    def validation_step(self, batch, batch_idx):
        pred, h1, h2 = self.model(batch)

        loss, full_state_loss, latent_loss = self.compute_loss(batch, pred, h1, h2)

        self.log('val/loss', loss)
        self.log('val/full_state_loss', full_state_loss)
        self.log('val/latent_loss', latent_loss)

        return loss

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.lr)
        return optimizer

    def compute_loss(self, x, pred, h1, h2):
        full_state_loss = F.mse_loss(pred[:, :-1], x[:, 1:]) / (torch.linalg.norm(x[:, 1:]) / x[:, 1:].numel() + 1e-6)
        latent_loss = F.mse_loss(h2[:, :-1], h1[:, 1:]) / (torch.linalg.norm(h1[:, 1:]) / h1[:, 1:].numel() + 1e-6)
        loss = full_state_loss + latent_loss
        return loss, full_state_loss, latent_loss
