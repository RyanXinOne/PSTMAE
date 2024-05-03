import torch
from torch import nn


class ConvAutoEncoder(nn.Module):
    '''
    A convolutional autoencoder that compresses the input image into the latent space.

    Image size: 3 * 64 * 64
    '''

    def __init__(self):
        super(ConvAutoEncoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 16, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(64 * 8 * 8, 128)
        )
        self.decoder = nn.Sequential(
            nn.Linear(128, 64 * 8 * 8),
            nn.Unflatten(1, (64, 8, 8)),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 16, 3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(16, 3, 3, stride=2, padding=1, output_padding=1),
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


class ConvRecurrentAutoEncoder(nn.Module):
    '''
    A Convolutional Recurrent Autoencoder for time series forecasting.

    Dimensionality reduction via convolutional autoencoder + 
    Learning feature dynamics via LSTM
    '''

    def __init__(self):
        super(ConvRecurrentAutoEncoder, self).__init__()
        self.autoencoder = ConvAutoEncoder()
        self.lstm = nn.LSTM(128, 128)
        self.proj = nn.Linear(128, 128)

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
        input shape: (seq_len, 3, 64, 64)
        '''
        x = self.autoencoder.encode(x)
        h1 = x.clone()
        x, _ = self.lstm(x)
        x = self.proj(x)
        h2 = x.clone()
        x = self.autoencoder.decode(x)
        return x, h1, h2


if __name__ == '__main__':
    model = ConvRecurrentAutoEncoder()
    print(model)

    x = torch.randn(10, 3, 64, 64)
    x, h1, h2 = model(x)
    print(x.shape, h1.shape, h2.shape)
