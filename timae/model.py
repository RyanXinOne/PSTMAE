import math
import torch
from torch import nn


class SeqConv2d(nn.Conv2d):
    '''
    Conv2d for sequence data.
    '''

    def forward(self, x):
        '''
        input shape: (b, l, c, h, w)
        '''
        b, l, c, h, w = x.size()
        x = x.reshape(-1, c, h, w)
        x = super().forward(x)
        x = x.reshape(b, l, x.size(1), x.size(2), x.size(3))
        return x


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=1000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(1, max_len, d_model)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Args:
            x: Tensor, shape [batch_size, seq_len, d_model]

        Returns:
            output Tensor of shape [batch_size, seq_len, d_model]
        """

        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class TimeSeriesMaskedAutoencoder(nn.Module):
    """
    Masked Autoencoder with VanillaTransformer backbone for TimeSeries.

    input shape: (N, L, W)
    """

    def __init__(
        self,
        img_size=(3, 64, 64),
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

        self.mask_ratio = mask_ratio
        self.forecast_ratio = forecast_ratio
        self.forecast_steps = forecast_steps

        self.embedder = nn.Sequential(
            nn.Unflatten(2, img_size),
            SeqConv2d(img_size[0], 4, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Flatten(start_dim=2),
            nn.Linear(4 * (img_size[1] // 2) * (img_size[2] // 2), embed_dim),
        )
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
        self.decoder_embed = nn.Linear(embed_dim, decoder_embed_dim)

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
        self.decoder_pred = nn.Sequential(
            nn.Linear(decoder_embed_dim, img_size[0] * img_size[1] * img_size[2]),
            nn.Sigmoid(),
        )

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
        len_keep = int((L - self.forecast_steps) * (1 - self.mask_ratio))

        noise = torch.rand(N, L, device=x.device)  # noise in [0, 1]

        # perform forecasting masking
        n_forecast_batches = int(N * self.forecast_ratio)
        if n_forecast_batches > 0:
            noise[-n_forecast_batches:, -(len_keep + self.forecast_steps):-self.forecast_steps] -= 1.

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
        x = self.pos_encoder_e(x)

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
        x: [N, L, W]
        pred: [N, L, W]
        mask: [N, W], 0 is keep, 1 is remove,
        """
        # calculate mean square error
        loss = (x - pred) ** 2
        loss = loss.mean(dim=-1)  # [N, L], mean loss per timestamp

        # calculate forecast and foreseen loss
        n_forecast_batches = int(pred.shape[0] * self.forecast_ratio)
        if n_forecast_batches > 0:
            loss_forecast = loss[-n_forecast_batches:, :]
            mask_forecast = mask[-n_forecast_batches:, :]
            mask_foreseen = (mask_forecast - 1) ** 2

            forecast_loss = loss_forecast[:, -self.forecast_steps:].sum() / mask_forecast[:, -self.forecast_steps:].sum() if mask_forecast[:, -self.forecast_steps:].sum() > 0 else 0
            foreseen_loss = (loss_forecast * mask_foreseen).sum() / mask_foreseen.sum() if mask_foreseen.sum() > 0 else 0

            loss = loss[:-n_forecast_batches, :]
            mask = mask[:-n_forecast_batches, :]
        else:
            forecast_loss = 0
            foreseen_loss = 0

        # calculate masked and seen loss
        inv_mask = (mask - 1) ** 2
        loss_removed = (loss * mask).sum() / mask.sum() if mask.sum() > 0 else 0
        loss_seen = (loss * inv_mask).sum() / inv_mask.sum() if inv_mask.sum() > 0 else 0

        return loss_removed, loss_seen, forecast_loss, foreseen_loss

    def forward(self, x):
        latent, mask, ids_restore = self.forward_encoder(x)
        pred = self.forward_decoder(latent, ids_restore)

        losses = self.forward_loss(x, pred, mask)

        return losses, pred

    def predict(self, x, pred_samples=5):
        '''
        Forecast the future steps given the input sequence.
        '''
        with torch.no_grad():
            # embed patches
            x = self.embedder(x)

            # add pos embed
            x = self.pos_encoder_e(x)

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
    x = torch.rand((5, 50, 12288))

    timae = TimeSeriesMaskedAutoencoder(mask_ratio=0., forecast_ratio=1., forecast_steps=5)
    losses, pred = timae(x)
    print(pred.shape)
    print(f'Losses : {[loss for loss in losses]}')

    x = x[:, :-5]
    pred = timae.predict(x, 5)
    print(pred.shape)
