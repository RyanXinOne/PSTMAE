import os
import torch
from torch import optim
import torch.nn.functional as F
import lightning.pytorch as pl
from models.timae import TimeSeriesMaskedAutoencoder
from data.utils import visualise_sequence, calculate_ssim_series, calculate_psnr_series, calculate_image_level_mse_std
from data.dataset import ShallowWaterDataset as sw


class LitTiMAE(pl.LightningModule):
    def __init__(self, dataset):
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
        self.enable_energy_loss = False
        self.enable_operator_loss = False

        # load pretrained autoencoder
        self.model.autoencoder.load_pretrained_freeze()

        min_vals = torch.from_numpy(self.dataset.min_vals).float()
        max_vals = torch.from_numpy(self.dataset.max_vals).float()
        self.register_buffer('min_vals', min_vals)
        self.register_buffer('max_vals', max_vals)

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

    def compute_loss(self, data, pred, z1, z2, config):
        full_state_loss = F.mse_loss(pred, data)
        latent_loss = F.mse_loss(z2, z1)
        energy_loss = 0
        if self.enable_energy_loss:
            energy_loss = F.mse_loss(
                sw.calculate_total_energy(pred, self.min_vals, self.max_vals),
                sw.calculate_total_energy(data, self.min_vals, self.max_vals),
            )
        operator_loss = 0
        if self.enable_operator_loss:
            operator_loss += F.mse_loss(
                pred[:, 1:],
                sw.evolve_with_flow_operator(
                    pred[:, :-1],
                    self.min_vals,
                    self.max_vals,
                    evolve_step=config['step']*self.dataset.dilation,
                    b=config['b'],))
        loss = full_state_loss + 0.5 * latent_loss + 0.1 * energy_loss + 0.1 * operator_loss
        return loss, full_state_loss, latent_loss, energy_loss, operator_loss

    def training_step(self, batch, batch_idx):
        x, y, mask, config = batch
        data = torch.cat([x, y], dim=1)
        z1 = self.model.autoencoder.encode(data)

        pred, z2 = self.model(x, mask)
        loss, full_state_loss, latent_loss, energy_loss, operator_loss = self.compute_loss(data, pred, z1, z2, config)

        self.log('train/loss', loss)
        self.log('train/mse', full_state_loss)
        self.log('train/latent_mse', latent_loss)
        if self.enable_energy_loss:
            self.log('train/energy_mse', energy_loss)
        if self.enable_operator_loss:
            self.log('train/operator_mse', operator_loss)
        self.log('train/lr', self.trainer.optimizers[0].param_groups[0]['lr'])
        loss = torch.nan_to_num(loss, nan=10.0, posinf=10.0, neginf=10.0)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y, mask, config = batch
        data = torch.cat([x, y], dim=1)
        z1 = self.model.autoencoder.encode(data)

        with torch.no_grad():
            pred, z2 = self.model(x, mask)
            loss, full_state_loss, latent_loss, energy_loss, operator_loss = self.compute_loss(data, pred, z1, z2, config)

        self.log('val/loss', loss)
        self.log('val/mse', full_state_loss)
        self.log('val/latent_mse', latent_loss)
        if self.enable_energy_loss:
            self.log('val/energy_mse', energy_loss)
        if self.enable_operator_loss:
            self.log('val/operator_mse', operator_loss)
        return loss

    def test_step(self, batch, batch_idx):
        x, y, mask, config = batch
        data = torch.cat([x, y], dim=1)
        z1 = self.model.autoencoder.encode(data)

        with torch.no_grad():
            pred, z2 = self.model(x, mask)
            loss, full_state_loss, latent_loss, energy_loss, operator_loss = self.compute_loss(data, pred, z1, z2, config)
            mse_value, mse_std = calculate_image_level_mse_std(data, pred)
            ssim_value, ssim_std = calculate_ssim_series(data, pred)
            psnr_value, psnr_std = calculate_psnr_series(data, pred)

        self.log('test/loss', loss)
        self.log('test/mse', full_state_loss)
        self.log('test/latent_mse', latent_loss)
        if self.enable_energy_loss:
            self.log('test/energy_mse', energy_loss)
        if self.enable_operator_loss:
            self.log('test/operator_mse', operator_loss)
        self.log('test/ssim', ssim_value)
        self.log('test/psnr', psnr_value)

        self.log('test/mse_std', mse_std)
        self.log('test/ssim_std', ssim_std)
        self.log('test/psnr_std', psnr_std)
        return loss

    def predict_step(self, batch, batch_idx):
        x, y, mask = batch[:3]

        batch_size = len(x)
        os.makedirs('logs/timae/output', exist_ok=True)

        with torch.no_grad():
            pred, _ = self.model(x, mask)

        for i in range(batch_size):
            vi = batch_idx * batch_size + i
            if vi >= self.visualise_num:
                break
            input_ = torch.cat([x[i], y[i]], dim=0)
            output = pred[i]
            diff = torch.abs(input_ - output)
            visualise_sequence(input_, save_path=f'logs/timae/output/input_{vi}.png')
            visualise_sequence(output, save_path=f'logs/timae/output/predict_{vi}.png')
            visualise_sequence(diff, save_path=f'logs/timae/output/diff_{vi}.png')
        return pred
