import torch
import torch.nn as nn

class Decoder(nn.Module):
    """
    Decoder for the auto-encoder/decoder network
    ref https://zo7.github.io/blog/2016/09/25/generating-faces.html
    """
    def __init__(self, z_dim, hidden_dim, num_layers, linear_out, de_conv_stuff, batch_size):
        super(Decoder, self).__init()
        self.z_dim = z_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.linear_out = linear_out
        # self.de_conv_stuff, \  TODO DO ME!!
        self.batch_size = batch_size

        self.lstm = nn.LSTM(input_size=z_dim,
                            hidden_size=hidden_dim,
                            num_layers=num_layers)  # TODO: Consider dropout
        self.linear = nn.Linear(in_features=hidden_dim,
                                out_features=linear_out)
        self.de_conv_1 = nn.ConvTranspose2d(in_channels=linear_out,
                                            out_channels=3,
                                            kernel_size=5)

    def forward(self, z_input, num_steps, batch_size=None):
        if batch_size is None:
            batch_size = self.batch_size
        working, _ = self.lstm(z_input)
        #  TODO: Consider dropout
        working = self.linear(-1, self.hidden_dim)
        # TODO: reshape to look like a video
        working = self.de_conv_1(working)


