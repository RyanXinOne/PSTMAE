import torch
from torch import nn


class SeqConvAutoEncoder(nn.Module):
    '''
    A Convolutional AutoEncoder that compresses sequence images into latent space.

    Image size: 128 * 128
    '''

    def __init__(self, input_dim, latent_dim):
        super(SeqConvAutoEncoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(input_dim, 8, 3, stride=1, padding=1),
            nn.GELU(),
            nn.Conv2d(8, 16, 3, stride=2, padding=1),
            nn.GELU(),
            nn.Conv2d(16, 32, 3, stride=2, padding=1),
            nn.GELU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.GELU(),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.GELU(),
            nn.Flatten(),
            nn.Linear(128 * 8 * 8, latent_dim)
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 128 * 8 * 8),
            nn.Unflatten(1, (128, 8, 8)),
            nn.GELU(),
            nn.ConvTranspose2d(128, 64, 3, stride=2, padding=1, output_padding=1),
            nn.GELU(),
            nn.ConvTranspose2d(64, 32, 3, stride=2, padding=1, output_padding=1),
            nn.GELU(),
            nn.ConvTranspose2d(32, 16, 3, stride=2, padding=1, output_padding=1),
            nn.GELU(),
            nn.ConvTranspose2d(16, 8, 3, stride=2, padding=1, output_padding=1),
            nn.GELU(),
            nn.Conv2d(8, input_dim, 3, stride=1, padding=1),
            nn.Sigmoid()
        )
        self.initialise_weights()
        self.freezed = False

    def initialise_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                nn.init.xavier_normal_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        '''
        input shape: (b, l, c, h, w)
        '''
        z = self.encode(x)
        x = self.decode(z)
        return x, z

    def encode(self, x):
        if self.freezed:
            self.eval()
        b, l, c, h, w = x.size()
        x = x.reshape(-1, c, h, w)
        z = self.encoder(x)
        z = z.reshape(b, l, -1)
        return z

    def decode(self, z):
        if self.freezed:
            self.eval()
        b, l, d = z.size()
        z = z.reshape(-1, d)
        x = self.decoder(z)
        x = x.reshape(b, l, x.size(-3), x.size(-2), x.size(-1))
        return x

    def load_pretrained_freeze(self):
        pl_ckpt_path = ''
        if pl_ckpt_path:
            # load pretrained autoencoder
            state_dict = torch.load(pl_ckpt_path, map_location='cpu')['state_dict']
            # drop prefix
            for key in list(state_dict):
                state_dict[key.replace("model.", "")] = state_dict.pop(key)
            self.load_state_dict(state_dict)
        # freeze autoencoder
        for param in self.parameters():
            param.requires_grad = False
        self.eval()
        self.freezed = True


if __name__ == '__main__':
    model = SeqConvAutoEncoder(input_dim=2, latent_dim=128)
    model.load_pretrained_freeze()
    x = torch.randn(5, 10, 2, 128, 128)
    y, z = model(x)
    print(y.size(), z.size())
