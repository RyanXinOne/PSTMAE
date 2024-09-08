import math
import torch
from torch import nn
import torch.nn.functional as F
from models.autoencoder import SeqConvAutoEncoder


# reference: [ti-mae](https://github.com/asmodaay/ti-mae/blob/master/src/nn/positional.py)
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=100, scaler=1.0):
        super().__init__()
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(1, max_len, d_model)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        pe *= scaler
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Args:
            x: Tensor, shape [batch_size, seq_len, d_model]

        Returns:
            output Tensor of shape [batch_size, seq_len, d_model]
        """

        x = x + self.pe[:, :x.size(1), :]
        return x


# reference: [ti-mae](https://github.com/asmodaay/ti-mae/blob/master/src/nn/model.py)
class TimeSeriesMaskedAutoencoder(nn.Module):
    """
    Masked Autoencoder with VanillaTransformer backbone for TimeSeries.

    input shape: (N, L, W)
    """

    def __init__(
        self,
        input_dim=3,
        latent_dim=64,
        hidden_dim=128,
        encoder_num_heads=4,
        encoder_depth=2,
        decoder_num_heads=4,
        decoder_depth=2,
        forecast_steps=5
    ):
        super().__init__()
        self.forecast_steps = forecast_steps

        # initialise autoencoder
        self.autoencoder = SeqConvAutoEncoder(input_dim=input_dim, latent_dim=latent_dim)

        # prepare positional encoder
        self.pos_encoder = PositionalEncoding(latent_dim, scaler=1.0)

        # initialise MAE encoder
        self.blocks = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=latent_dim,
                nhead=encoder_num_heads,
                dim_feedforward=hidden_dim,
                dropout=0.,
                activation=F.leaky_relu,
                batch_first=True,
                norm_first=True,
            ) for _ in range(encoder_depth)
        ])
        # self.norm = nn.LayerNorm(latent_dim)
        self.decoder_embed = nn.Linear(latent_dim, latent_dim)

        self.mask_token = nn.Parameter(torch.zeros(1, 1, latent_dim), requires_grad=True)

        # initialise MAE decoder
        self.decoder_blocks = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=latent_dim,
                nhead=decoder_num_heads,
                dim_feedforward=hidden_dim,
                dropout=0.,
                activation=F.leaky_relu,
                batch_first=True,
                norm_first=True,
            ) for _ in range(decoder_depth)
        ])
        # self.decoder_norm = nn.LayerNorm(latent_dim)

        self.initialize_weights()

    def initialize_weights(self):
        nn.init.xavier_normal_(self.mask_token)

        # initialize nn.Linear and nn.LayerNorm
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0)

    def handle_masking(self, x, mask):
        """
        Handle the masking of the sequence.
        x: [N, L, D], sequence
        mask: [N, L], binary mask, 0 is keep, 1 is remove
        """
        assert x.shape[:2] == mask.shape, 'x and mask should have the same shape'
        num_keep = mask.shape[1] - mask.sum(dim=1)
        assert num_keep.max() == num_keep.min(), 'all samples in batch should have the same masking steps'
        num_keep = num_keep[0].int()

        # append forecast steps to mask
        mask = torch.cat([mask,
                          torch.ones(mask.shape[0], self.forecast_steps, device=mask.device, dtype=mask.dtype)
                          ], dim=1)

        # rearange masking steps
        ids_sort = torch.argsort(mask, dim=1)
        ids_restore = torch.argsort(ids_sort, dim=1)

        # filter out masked steps
        ids_keep = ids_sort[:, :num_keep]
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).expand(-1, -1, x.shape[2]))

        return x_masked, ids_restore

    def forward_encoder(self, x, mask):
        # add pos embed
        x = self.pos_encoder(x)

        # apply masking
        x, ids_restore = self.handle_masking(x, mask)

        # apply Transformer blocks
        for blk in self.blocks:
            x = blk(x)
        # x = self.norm(x)

        return x, ids_restore

    def forward_decoder(self, x, ids_restore):
        # embed tokens
        x = self.decoder_embed(x)

        # append mask tokens to sequence
        mask_tokens = self.mask_token.repeat(x.shape[0], ids_restore.shape[1] - x.shape[1], 1)
        x = torch.cat([x, mask_tokens], dim=1)
        x = torch.gather(x, dim=1, index=ids_restore.unsqueeze(-1).expand(-1, -1, x.shape[2]))

        # add pos embed
        x = self.pos_encoder(x)

        # apply Transformer blocks
        for blk in self.decoder_blocks:
            x = blk(x)
        # x = self.decoder_norm(x)

        return x

    def forward(self, x, mask):
        latent = self.autoencoder.encode(x)
        latent, ids_restore = self.forward_encoder(latent, mask)
        latent = self.forward_decoder(latent, ids_restore)
        pred = self.autoencoder.decode(latent)
        return pred, latent


if __name__ == '__main__':
    x = torch.rand((2, 5, 3, 128, 128))
    mask = torch.tensor([[1, 0, 1, 0, 0],
                         [0, 0, 1, 1, 0]]).float()

    timae = TimeSeriesMaskedAutoencoder(forecast_steps=5)
    print(timae)
    pred, latent = timae(x, mask)
    print(pred.shape)
    print(latent.shape)
