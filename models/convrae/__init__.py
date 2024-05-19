import torch
from torch import nn
from models.autoencoder import SeqConvAutoEncoder


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
        self.proj_e = nn.Linear(hidden_dim, latent_dim)
        self.proj_f = nn.Linear(hidden_dim, latent_dim)

    def forward(self, x, y):
        '''
        input shape: (b, l, c, 64, 64)
        '''
        # image space -> latent space
        zx = self.autoencoder.encode(x)
        zy = self.autoencoder.encode(y)
        zx_1 = zx.clone()
        zy_1 = zy.clone()

        # encode input sequence
        zx_pred, hidden_state = self.encoder(zx)
        zx_pred = self.proj_e(zx_pred)
        zx_2 = zx_pred.clone()

        # forecast future sequence by teacher forcing
        zy_input = torch.cat([zx[:, -1:], zy[:, :-1]], dim=1)
        zy_pred, _ = self.forecaster(zy_input, hidden_state)
        zy_pred = self.proj_f(zy_pred)
        zy_2 = zy_pred.clone()

        # latent space -> image space
        x_pred = self.autoencoder.decode(zx_pred)
        y_pred = self.autoencoder.decode(zy_pred)

        return x_pred, y_pred, zx_1, zx_2, zy_1, zy_2

    def predict(self, x, forecast_steps):
        '''
        input shape: (b, l, c, 64, 64)
        '''
        with torch.no_grad():
            # image space -> latent space
            zx = self.autoencoder.encode(x)

            # encode input sequence
            zx_pred, hidden_state = self.encoder(zx)
            zx_pred = self.proj_e(zx_pred)

            # forecast future sequence
            zy_input = zx[:, -1:]
            zy_pred = []
            for _ in range(forecast_steps):
                zy_input, hidden_state = self.forecaster(zy_input, hidden_state)
                zy_input = self.proj_f(zy_input)
                zy_pred.append(zy_input)
            zy_pred = torch.cat(zy_pred, dim=1)

            # latent space -> image space
            x_pred = self.autoencoder.decode(zx_pred)
            y_pred = self.autoencoder.decode(zy_pred)

        return x_pred, y_pred


if __name__ == '__main__':
    model = ConvRAE(3, 128, 128)
    print(model)

    x = torch.randn(5, 10, 3, 64, 64)
    y = torch.randn(5, 5, 3, 64, 64)
    x_pred, y_pred, zx_1, zx_2, zy_1, zy_2 = model(x, y)
    print(x_pred.shape, zx_1.shape, zx_2.shape)
    print(y_pred.shape, zy_1.shape, zy_2.shape)
    x_pred, y_pred = model.predict(x, 5)
    print(x_pred.shape, y_pred.shape)
