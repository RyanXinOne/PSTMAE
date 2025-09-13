import os
import torch
from torch import optim
import torch.nn.functional as F
import lightning.pytorch as pl
from models.timae import TimeSeriesMaskedAutoencoder
from data.utils import visualise_sequence, calculate_ssim_series, calculate_psnr_series, calculate_image_level_mse_std


class LitTiMAE(pl.LightningModule):
    # def __init__(self, dataset):
    def __init__(self, dataset, lambda_latent: float = 0.5):
        super().__init__()
        self.model = TimeSeriesMaskedAutoencoder(
            input_dim=3,
            latent_dim=128,
            hidden_dim=256,
            encoder_num_heads=2,
            encoder_depth=4,
            decoder_num_heads=2,
            decoder_depth=1,
            forecast_steps=5
        )
        self.dataset = dataset
        self.visualise_num = 5

         # === save λ ===
        self.lambda_latent = lambda_latent
        # self.save_hyperparameters({"lambda_latent": lambda_latent})2

        # load pretrained autoencoder
        self.model.autoencoder.load_pretrained_freeze()

    def configure_optimizers(self):
        optimizer = optim.RAdam(
            self.parameters(),
            lr=3e-4,
            weight_decay=0)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=10,
            eta_min=3e-4)
        return [optimizer], [scheduler]

    def compute_loss(self, x, pred, z1, z2):
        full_state_loss = F.mse_loss(pred, x)
        latent_loss = F.mse_loss(z2, z1)
        # loss = full_state_loss + 0.5 * latent_loss
        loss = full_state_loss + self.lambda_latent * latent_loss
        return loss, full_state_loss, latent_loss

    def training_step(self, batch, batch_idx):
        x, y, mask = batch
        data = torch.cat([x, y], dim=1)
        z1 = self.model.autoencoder.encode(data)

        pred, z2 = self.model(x, mask)
        loss, full_state_loss, latent_loss = self.compute_loss(data, pred, z1, z2)

        self.log('train/loss', loss)
        self.log('train/mse', full_state_loss)
        self.log('train/latent_mse', latent_loss)
        self.log('train/lr', self.trainer.optimizers[0].param_groups[0]['lr'])
        self.log('train/lambda_latent', self.lambda_latent, prog_bar=False, on_step=False, on_epoch=True) # save λ
        loss = torch.nan_to_num(loss, nan=10.0, posinf=10.0, neginf=10.0)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y, mask = batch
        data = torch.cat([x, y], dim=1)
        z1 = self.model.autoencoder.encode(data)

        with torch.no_grad():
            pred, z2 = self.model(x, mask)
            loss, full_state_loss, latent_loss = self.compute_loss(data, pred, z1, z2)

        self.log('val/loss', loss)
        self.log('val/mse', full_state_loss)
        self.log('val/latent_mse', latent_loss)
        return loss

    def test_step(self, batch, batch_idx):
        x, y, mask = batch
        data = torch.cat([x, y], dim=1)
        z1 = self.model.autoencoder.encode(data)

        with torch.no_grad():
            pred, z2 = self.model(x, mask)
            loss, full_state_loss, latent_loss = self.compute_loss(data, pred, z1, z2)
            mse_value, mse_std = calculate_image_level_mse_std(data, pred)
            ssim_value, ssim_std = calculate_ssim_series(data, pred)
            psnr_value, psnr_std = calculate_psnr_series(data, pred)

        self.log('test/loss', loss)
        self.log('test/mse', full_state_loss)
        self.log('test/latent_mse', latent_loss)
        self.log('test/ssim', ssim_value)
        self.log('test/psnr', psnr_value)

        self.log('test/mse_std', mse_std)
        self.log('test/ssim_std', ssim_std)
        self.log('test/psnr_std', psnr_std)
        return loss

    def predict_step(self, batch, batch_idx):
        x, y, mask = batch

        batch_size = len(x)
        os.makedirs(f'logs/timae/lambda_{self.lambda_latent}/output', exist_ok=True)

        with torch.no_grad():
            pred, _ = self.model(x, mask)

        for i in range(batch_size):
            vi = batch_idx * batch_size + i
            if vi >= self.visualise_num:
                break
            input_ = torch.cat([x[i], y[i]], dim=0)
            output = pred[i]
            diff = torch.abs(input_ - output)
            visualise_sequence(input_, save_path=f'logs/timae/lambda_{self.lambda_latent}/output/input_{vi}.png')
            visualise_sequence(output, save_path=f'logs/timae/lambda_{self.lambda_latent}/output/predict_{vi}.png')
            visualise_sequence(diff, save_path=f'logs/timae/lambda_{self.lambda_latent}/output/diff_{vi}.png')
        return pred