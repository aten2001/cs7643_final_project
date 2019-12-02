import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from data import DataSet

class Deconv(nn.Module):
    """
    Decoder for the auto-encoder/decoder network
    ref https://zo7.github.io/blog/2016/09/25/generating-faces.html
    """
    def __init__(self):  # , input_dim, hidden_dim, num_layers, linear_out, de_conv_stuff, batch_size):
        super(Deconv, self).__init__()

        self.lstm_1 = nn.LSTM(input_size=25, hidden_size=25, num_layers=2)
        self.unpool_1 = nn.Upsample(scale_factor=5, mode='bilinear')
        self.deconv_1 = nn.ConvTranspose2d(in_channels=1, out_channels=3, kernel_size=200, stride=1)

        # self.input_dim = input_dim
        # self.hidden_dim = hidden_dim
        # self.num_layers = num_layers
        # self.linear_out = linear_out
        # # self.de_conv_stuff, \  TODO DO ME!!
        # self.batch_size = batch_size
        #
        # self.lstm = nn.LSTM(input_size=z_dim,
        #                     hidden_size=hidden_dim,
        #                     num_layers=num_layers)  # TODO: Consider dropout
        # self.linear = nn.Linear(in_features=hidden_dim,
        #                         out_features=linear_out)
        # self.de_conv_1 = nn.ConvTranspose2d(in_channels=linear_out,
        #                                     out_channels=3,
        #                                     kernel_size=5)

    def forward(self, input_vec):
        output = input_vec
        output, (_, _) = self.lstm_1(output)  # input of shape (seq_len, batch, input_size)
        output = output.view(2, 1, 5, 5)
        output = self.unpool_1(output)
        output = self.deconv_1(output)

        return output


if __name__ == "__main__":

    model = Deconv()
    criterion = F.mse_loss
    # criterion = F.binary_cross_entropy
    optimizer = optim.SGD(model.parameters(), lr=.001, momentum=.2, weight_decay=0)

    tensor_1 = torch.ones(1, 1, 25)
    tensor_1 = tensor_1.repeat(2, 1, 1)

    # Set values for DataSet object.
    seq_length = 2
    class_limit = 1  # Number of classes to extract. Can be 1-101 or None for all.
    video_limit = 1  # Number of videos allowed per class.  None for no limit
    data = DataSet(seq_length=seq_length, class_limit=class_limit, video_limit=video_limit)

    video_array = None
    for video in data.data:
        video_array = data.video_to_vid_array(video)  # Get numpy array of sequence
        break  # Only need one video to begin with.

    video_tensor = torch.from_numpy(video_array).type(torch.float32)

    model.train()
    for index in range(200):
        optimizer.zero_grad()

        output = model(tensor_1)
        # print(video_tensor.dtype)
        # print(output.dtype)
        loss = criterion(output, video_tensor)
        print("Step: {}, Loss: {}".format(index, loss))
        loss.backward()
        optimizer.step()

    model.eval()
    output = model(tensor_1)
    array_video = output.detach().numpy()
    data.vid_array_to_video("test_me", array_video)


