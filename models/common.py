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


class SeqTransposeConv2d(nn.ConvTranspose2d):
    '''
    ConvTranspose2d for sequence data.
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
