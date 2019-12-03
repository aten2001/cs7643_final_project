import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from data import DataSet


class Decoder(nn.Module):
    """
    Decoder for the auto-encoder/decoder network
    ref https://zo7.github.io/blog/2016/09/25/generating-faces.html
    """

    def __init__(self, batch_size, seq_len, l_input_size=25, l_hidden_size=25,
                 l_num_layers=1):  # , input_dim, hidden_dim, num_layers, linear_out, de_conv_stuff, batch_size):
        super(Decoder, self).__init__()
        self.batch_size = batch_size
        self.sequence_len = seq_len
        self.deconv_1_in = 16
        self.lstm_1 = nn.LSTM(input_size=l_hidden_size, hidden_size=l_hidden_size, num_layers=l_num_layers,
                              batch_first=True)
        self.unpool_1 = nn.Upsample(scale_factor=5, mode='bilinear')
        self.deconv_1 = nn.ConvTranspose2d(in_channels=self.deconv_1_in, out_channels=3, kernel_size=200, stride=1)

        # TODO: Consider dropout

    def forward(self, input_vec):
        # print(input_vec.shape)
        out = input_vec
        out, _ = self.lstm_1(out)  # input of shape (batch, seq_len, input_size)

        """
        # Below runs the LSTM cell by cell
        out = torch.zeros_like(input_vec)  # For running the LSTM Cell by cell
        combo = None
        for i in range(self.sequence_len):
            out[:, i], combo = self.lstm_1(input_vec[:, i].view(1, 1, self.input_size), combo)  # input of shape (batch, seq_len, input_size)
            print("line")
            print("hidden ", i)
            print(combo[0])
            print("cell ",  i)
            print(combo[1])
        """

        # print(out.detach().cpu())
        out = out.reshape(self.batch_size * self.sequence_len, self.deconv_1_in, 5, 5)  # .view didn't work
        out = self.unpool_1(out)
        out = self.deconv_1(out)
        out = out.view(self.batch_size, self.sequence_len, 3, 224, 224)

        return out


if __name__ == "__main__":

    seq_length = 8
    lr = 0.1

    # Check for cuda
    if torch.cuda.is_available():
        device = 'cuda'
    else:
        device = 'cpu'

    model = Decoder(seq_length).to(device)
    criterion = F.mse_loss
    # criterion = F.binary_cross_entropy
    optimizer = optim.Adam(model.parameters(), lr=lr)

    tensor_1 = torch.ones(1, 1, 100)
    tensor_1 = tensor_1.repeat(1, seq_length, 1).to(device)

    # Set values for DataSet object.

    class_limit = 101  # Number of classes to extract. Can be 1-101 or None for all.
    video_limit = 1  # Number of videos allowed per class.  None for no limit
    data = DataSet(seq_length=seq_length, class_limit=class_limit, video_limit=video_limit)

    video_array = None
    i = 0
    for video in data.data:
        video_array = data.video_to_vid_array(video)  # Get numpy array of sequence
        i += 1
        # print (video_array)
        if (i == 10):
            break

    data.vid_array_to_video("ground_truth", video_array)

    video_tensor = torch.from_numpy(video_array).type(torch.float32).to(device)

    model.train()
    for index in range(10000):

        if (index % 5000 == 0):
            for param_group in optimizer.param_groups:
                param_group['lr'] *= 0.1

            print("learning rate is now: ", param_group['lr'])

        optimizer.zero_grad()

        output = model.forward(tensor_1)
        # print(video_tensor.shape)
        # print(output.shape)
        loss = criterion(output, video_tensor)
        print("Step: {}, Loss: {}".format(index, loss))
        loss.backward()
        optimizer.step()

    model.eval()
    output = model(tensor_1)
    array_video = output.detach().cpu().numpy()
    data.vid_array_to_video("test_me", array_video)


