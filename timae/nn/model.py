import math
import torch
from torch import nn
from nn.positional import PositionalEncoding


class TimeSeriesMaskedAutoencoder(nn.Module):
    """Masked Autoencoder with VanillaTransformer backbone for TimeSeries"""

    def __init__(
        self,
        in_chans,
        embed_dim=64,
        num_heads=4,
        depth=2,
        decoder_embed_dim=32,
        decoder_num_heads=4,
        decoder_depth=2,
        d_hid=128,
        dropout=0.1,
        mask_ratio=0.75,
        forecast_ratio=0.25,
        forecast_steps=10
    ):
        super().__init__()

        self.embed_dim = embed_dim
        self.mask_ratio = mask_ratio
        self.forecast_ratio = forecast_ratio
        self.forecast_steps = forecast_steps

        self.embedder = nn.Linear(in_chans, embed_dim, bias=False)
        self.pos_encoder_e = PositionalEncoding(embed_dim)
        self.pos_encoder_d = PositionalEncoding(decoder_embed_dim)

        self.blocks = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=embed_dim,
                nhead=num_heads,
                dim_feedforward=d_hid,
                dropout=dropout,
                batch_first=True,
                norm_first=True,
            ) for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(embed_dim)
        self.decoder_embed = nn.Linear(embed_dim, decoder_embed_dim, bias=True)

        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))

        self.decoder_blocks = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=decoder_embed_dim,
                nhead=decoder_num_heads,
                dim_feedforward=d_hid,
                dropout=dropout,
                batch_first=True,
                norm_first=True,
            ) for _ in range(decoder_depth)
        ])
        self.decoder_norm = nn.LayerNorm(decoder_embed_dim)
        self.decoder_pred = nn.Linear(decoder_embed_dim, in_chans, bias=True)

        self.initialize_weights()

    def initialize_weights(self):
        torch.nn.init.normal_(self.mask_token, std=0.02)

        # initialize nn.Linear and nn.LayerNorm
        for m in self.modules():
            if isinstance(m, nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)

    def random_masking(self, x):
        """
        Perform per-sample random masking by per-sample shuffling.
        Per-sample shuffling is done by argsort random noise.
        x: [N, L, D], sequence
        """
        N, L, D = x.shape  # batch, length, dim
        len_keep = int(L * (1 - self.mask_ratio))

        noise = torch.rand(N, L, device=x.device)  # noise in [0, 1]

        # perform forecasting masking
        if self.forecast_ratio:
            n_forecast_batches = int(N * self.forecast_ratio)
            noise[-n_forecast_batches:, :-self.forecast_steps] -= 1.

        # sort noise for each sample
        ids_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is remove
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        # keep the first subset
        ids_keep = ids_shuffle[:, :len_keep]
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))

        # generate the binary mask: 0 is keep, 1 is remove
        mask = torch.ones([N, L], device=x.device)
        mask[:, :len_keep] = 0
        # unshuffle to get the binary mask
        mask = torch.gather(mask, dim=1, index=ids_restore)

        return x_masked, mask, ids_restore

    def forward_encoder(self, x):
        # embed patches
        x = self.embedder(x)

        # add pos embed
        x = self.pos_encoder_e(x * math.sqrt(self.embed_dim))

        # masking: length -> length * mask_ratio
        x, mask, ids_restore = self.random_masking(x)

        # apply Transformer blocks
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)

        return x, mask, ids_restore

    def forward_decoder(self, x, ids_restore):
        # embed tokens
        x = self.decoder_embed(x)

        # append mask tokens to sequence
        mask_tokens = self.mask_token.repeat(x.shape[0], ids_restore.shape[1] - x.shape[1], 1)
        x = torch.cat([x, mask_tokens], dim=1)
        x = torch.gather(x, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2]))  # unshuffle

        # add pos embed
        x = self.pos_encoder_d(x)

        # apply Transformer blocks
        for blk in self.decoder_blocks:
            x = blk(x)
        x = self.decoder_norm(x)

        # predictor projection
        x = self.decoder_pred(x)

        return x

    def forward_loss(self, x, pred, mask):
        """
        x: [N, W, L]
        pred: [N, L, W]
        mask: [N, W], 0 is keep, 1 is remove,
        """
        # calculate mean square error
        loss = (x - pred) ** 2
        loss = loss.mean(dim=-1)  # [N, L], mean loss per timestamp

        inv_mask = (mask - 1) ** 2
        loss_removed = (loss * mask).sum() / mask.sum()  # mean loss on removed patches
        loss_seen = (loss * inv_mask).sum() / inv_mask.sum()  # mean loss on seen patches

        # calculate forecast and backcast loss
        if self.forecast_ratio:
            n_forecast_batches = int(pred.shape[0] * self.forecast_ratio)
            loss_forecast = loss[-n_forecast_batches:, :]
            mask_forecast = mask[-n_forecast_batches:, :]

            loss = loss[:-n_forecast_batches, :]
            mask = mask[:-n_forecast_batches, :]

            forecast_loss = loss_forecast[:, -self.forecast_steps:].sum() / mask_forecast[:, -self.forecast_steps:].sum()
            backcast_loss = (loss_forecast * mask_forecast)[:, :-self.forecast_steps].sum() / mask_forecast[:, :-self.forecast_steps].sum()
        else:
            forecast_loss = 0
            backcast_loss = 0

        return loss_removed, loss_seen, forecast_loss, backcast_loss

    def forward(self, x):
        latent, mask, ids_restore = self.forward_encoder(x)
        pred = self.forward_decoder(latent, ids_restore)

        loss = self.forward_loss(x, pred, mask)

        return loss, pred

    def predict(self, x, pred_samples=5):
        with torch.no_grad():
            # embed patches
            x = self.embedder(x[:, :-pred_samples, :])

            # add pos embed
            x = self.pos_encoder_e(x * math.sqrt(self.embed_dim))

            # apply Transformer blocks
            for blk in self.blocks:
                x = blk(x)
            x = self.norm(x)

            # embed tokens
            x = self.decoder_embed(x)

            # append mask tokens to sequence
            mask_tokens = self.mask_token.repeat(x.shape[0], pred_samples, 1)
            x = torch.cat([x, mask_tokens], dim=1)

            # add pos embed
            x = self.pos_encoder_d(x)

            # apply Transformer blocks
            for blk in self.decoder_blocks:
                x = blk(x)
            x = self.decoder_norm(x)

            # predictor projection
            x = self.decoder_pred(x)

        return x


if __name__ == '__main__':
    batch, seq_len, in_chans = 5, 100, 50
    x = torch.rand((batch, seq_len, in_chans))
    timae = TimeSeriesMaskedAutoencoder(in_chans)
    losses, pred = timae(x)

    print(pred.shape, x.shape)
    print(f'Losses : {[loss.item() for loss in losses]}')

    timae.predict(x, 5)
