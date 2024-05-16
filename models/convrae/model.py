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
        x = self.encoder(x)
        z = x.clone()
        x = self.decoder(x)
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


class ConvRAE(nn.Module):
    '''
    A Convolutional Recurrent AutoEncoder for time series forecasting.

    Dimensionality reduction via convolutional autoencoder + 
    Learning feature dynamics via LSTM
    '''

    def __init__(self, input_dim, latent_dim, hidden_dim):
        super(ConvRAE, self).__init__()
        self.autoencoder = SeqConvAutoEncoder(input_dim, latent_dim)
        self.encoder = nn.LSTM(latent_dim, hidden_dim, batch_first=True)
        self.forecaster = nn.LSTM(latent_dim, hidden_dim, batch_first=True)
        self.proj = nn.Linear(hidden_dim, latent_dim)

    def forward(self, x, y):
        '''
        input shape: (b, l, c, 64, 64)
        '''
        # image space -> latent space
        zx = self.autoencoder.encode(x)
        zy = self.autoencoder.encode(y)
        z1 = zy.clone()

        # encode input sequence
        _, hidden_state = self.encoder(zx[:, :-1])

        # forecast future sequence by teacher forcing
        zy_input = torch.cat([zx[:, -1:], zy[:, :-1]], dim=1)
        zy_pred, _ = self.forecaster(zy_input, hidden_state)
        zy_pred = self.proj(zy_pred)
        z2 = zy_pred.clone()

        # latent space -> image space
        y_pred = self.autoencoder.decode(zy_pred)

        return y_pred, z1, z2

    def predict(self, x, forecast_steps):
        '''
        input shape: (b, l, c, 64, 64)
        '''
        with torch.no_grad():
            # image space -> latent space
            z = self.autoencoder.encode(x)

            # encode input sequence
            _, hidden_state = self.encoder(z[:, :-1])

            # forecast future sequence
            z_input = z[:, -1:]
            z_pred = []
            for _ in range(forecast_steps):
                z_input, hidden_state = self.forecaster(z_input, hidden_state)
                z_input = self.proj(z_input)
                z_pred.append(z_input)
            z_pred = torch.cat(z_pred, dim=1)

            # latent space -> image space
            y_pred = self.autoencoder.decode(z_pred)

        return y_pred


if __name__ == '__main__':
    model = ConvRAE(3, 128, 128)
    print(model)

    x = torch.randn(5, 10, 3, 64, 64)
    y = torch.randn(5, 5, 3, 64, 64)
    y_pred, z1, z2 = model(x, y)
    print(y_pred.shape, z1.shape, z2.shape)
    y_pred = model.predict(x, 5)
    print(y_pred.shape)
