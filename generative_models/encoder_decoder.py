import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from data import DataSet
from deconv import Deconv
from encoder import Encoder

class EncoderDecoder(nn.Module):
    """
    Decoder for the auto-encoder/decoder network
    ref https://zo7.github.io/blog/2016/09/25/generating-faces.html
    """
    def __init__(self):  # , input_dim, hidden_dim, num_layers, linear_out, de_conv_stuff, batch_size):
        super(EncoderDecoder, self).__init__()

        self.encoder = Encoder(cnn_model="vgg", h_lstm=25, lstm_layers=1)
        self.decoder = Deconv()

    def forward(self, x):
        """

        :param x: (S,C,H,W)
        :return:
        """

        working = self.encoder.forward(x)
        output = self.decoder.forward(working)

        return output



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

model = EncoderDecoder()
criterion = F.mse_loss
# criterion = F.binary_cross_entropy
optimizer = optim.SGD(model.parameters(), lr=.001, momentum=.2, weight_decay=0)

model.train()
for index in range(2):
    optimizer.zero_grad()

    output = model.forward(video_tensor)
    # print(video_tensor.dtype)
    # print(output.dtype)
    loss = criterion(output, video_tensor)
    print("Step: {}, Loss: {}".format(index, loss))
    loss.backward()
    optimizer.step()

model.eval()
output = model.forward(video_tensor)
array_video = output.detach().numpy()
data.vid_array_to_video("test_me", array_video)


