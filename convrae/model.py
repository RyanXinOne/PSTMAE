import torch
from torch import nn


class ConvAutoEncoder(nn.Module):
    '''
    A Convolutional AutoEncoder that compresses input image into latent space.

    Image size: c * 64 * 64
    '''

    def __init__(self, input_dim, latent_dim):
        super(ConvAutoEncoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(input_dim, 16, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(64 * 8 * 8, latent_dim)
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 64 * 8 * 8),
            nn.Unflatten(1, (64, 8, 8)),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 16, 3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(16, input_dim, 3, stride=2, padding=1, output_padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.encoder(x)
        h = x.clone()
        x = self.decoder(x)
        return x, h

    def encode(self, x):
        return self.encoder(x)

    def decode(self, x):
        return self.decoder(x)


class ConvRAE(nn.Module):
    '''
    A Convolutional Recurrent AutoEncoder for time series forecasting.

    Dimensionality reduction via convolutional autoencoder + 
    Learning feature dynamics via LSTM
    '''

    def __init__(self, input_dim, latent_dim, hidden_dim):
        super(ConvRAE, self).__init__()
        self.autoencoder = ConvAutoEncoder(input_dim, latent_dim)
        self.lstm = nn.LSTM(latent_dim, hidden_dim, batch_first=True)
        self.proj = nn.Linear(hidden_dim, latent_dim)

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
        b, l = x.size(0), x.size(1)

        x = x.view(-1, x.size(-3), x.size(-2), x.size(-1))
        x = self.autoencoder.encode(x)
        x = x.view(b, l, x.size(-1))

        h1 = x.clone()
        x, _ = self.lstm(x)
        x = self.proj(x)
        h2 = x.clone()

        x = x.view(-1, x.size(-1))
        x = self.autoencoder.decode(x)
        x = x.view(b, l, x.size(-3), x.size(-2), x.size(-1))

        return x, h1, h2


if __name__ == '__main__':
    model = ConvRAE(3, 128, 128)
    print(model)

    x = torch.randn(5, 10, 3, 64, 64)
    x, h1, h2 = model(x)
    print(x.shape, h1.shape, h2.shape)
