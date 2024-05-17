import torch
from torch import nn


class SeqConvAutoEncoder(nn.Module):
    '''
    A Convolutional AutoEncoder that compresses sequence images into latent space.

    Image size: 64 * 64
    '''

    def __init__(self, input_dim, latent_dim):
        super(SeqConvAutoEncoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(input_dim, 16, 3, stride=1, padding=1),
            nn.GELU(),
            nn.Conv2d(16, 16, 3, stride=2, padding=1),
            nn.GELU(),
            nn.Conv2d(16, 32, 3, stride=2, padding=1),
            nn.GELU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.GELU(),
            nn.Flatten(),
            nn.Linear(64 * 8 * 8, latent_dim)
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 64 * 8 * 8),
            nn.Unflatten(1, (64, 8, 8)),
            nn.GELU(),
            nn.ConvTranspose2d(64, 32, 3, stride=2, padding=1, output_padding=1),
            nn.GELU(),
            nn.ConvTranspose2d(32, 16, 3, stride=2, padding=1, output_padding=1),
            nn.GELU(),
            nn.ConvTranspose2d(16, 16, 3, stride=2, padding=1, output_padding=1),
            nn.GELU(),
            nn.Conv2d(16, input_dim, 3, stride=1, padding=1),
            nn.Sigmoid()
        )

        self.initialise_weights()

    def initialise_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                nn.init.kaiming_normal_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        '''
        input shape: (b, l, c, 64, 64)
        '''
        x = self.encode(x)
        z = x.clone()
        x = self.decode(x)
        return x, z

    def encode(self, x):
        b, l, c, h, w = x.size()
        x = x.reshape(-1, c, h, w)
        z = self.encoder(x)
        z = z.reshape(b, l, -1)
        return z

    def decode(self, z):
        b, l, d = z.size()
        z = z.reshape(-1, d)
        x = self.decoder(z)
        x = x.reshape(b, l, x.size(-3), x.size(-2), x.size(-1))
        return x


if __name__ == '__main__':
    model = SeqConvAutoEncoder(input_dim=3, latent_dim=128)
    x = torch.randn(2, 10, 3, 64, 64)
    y, z = model(x)
    print(y.size(), z.size())
