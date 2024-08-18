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
        input shape: (b, l, c, h, w)
        '''
        # image space -> latent space
        zx = self.autoencoder.encode(x)
        zy = self.autoencoder.encode(y)

        # encode input sequence
        zx_pred, hidden_state = self.encoder(zx)
        zx_pred = self.proj_e(zx_pred)

        # forecast future sequence by teacher forcing
        zy_input = torch.cat([zx[:, -1:], zy[:, :-1]], dim=1)
        zy_pred, _ = self.forecaster(zy_input, hidden_state)
        zy_pred = self.proj_f(zy_pred)

        # latent space -> image space
        x_pred = self.autoencoder.decode(zx_pred)
        y_pred = self.autoencoder.decode(zy_pred)

        pred = torch.cat([x_pred, y_pred], dim=1)
        z = torch.cat([zx_pred, zy_pred], dim=1)
        return pred, z

    def predict(self, x, forecast_steps):
        '''
        input shape: (b, l, c, h, w)
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

            pred = torch.cat([x_pred, y_pred], dim=1)
            z = torch.cat([zx_pred, zy_pred], dim=1)
        return pred, z


if __name__ == '__main__':
    model = ConvRAE(3, 128, 128)
    print(model)

    x = torch.randn(5, 10, 3, 128, 128)
    y = torch.randn(5, 5, 3, 128, 128)
    pred, z = model(x, y)
    print(pred.shape, z.shape)
    pred, z = model.predict(x, 5)
    print(pred.shape, z.shape)
