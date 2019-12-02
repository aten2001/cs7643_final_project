import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from data import DataSet
from encoder import Encoder
from decoder_single import DecoderSingle


class EncoderDecoder(nn.Module):
    """
    Decoder for the auto-encoder/decoder network
    ref https://zo7.github.io/blog/2016/09/25/generating-faces.html
    """
    def __init__(self):  # , input_dim, hidden_dim, num_layers, linear_out, de_conv_stuff, batch_size):
        super(EncoderDecoder, self).__init__()

        #cnn_model, input_dim, hidden_dim, lstm_layers, embedding_dim
        self.encoder = Encoder(cnn_model="vgg", input_dim=512, hidden_dim=100, lstm_layers=1, embedding_dim=100)
        self.decoder = DecoderSingle()

    def forward(self, x):
        """

        :param x: (S,C,H,W)
        :return:
        """

        working = self.encoder.forward(x)
        # working = working[-1]
        output = self.decoder.forward(working)

        return output



# Set values for DataSet object.
seq_length = 6
class_limit = 1  # Number of classes to extract. Can be 1-101 or None for all.
video_limit = 1  # Number of videos allowed per class.  None for no limit
data = DataSet(seq_length=seq_length, class_limit=class_limit, video_limit=video_limit)

video_array = None
for video in data.data:
    video_array = data.video_to_vid_array(video)  # Get numpy array of sequence
    break  # Only need one video to begin with.

data.vid_array_to_video("ground_truth", video_array)

video_tensor = torch.from_numpy(video_array).type(torch.float32)

model = EncoderDecoder()
criterion = F.mse_loss
# criterion = F.binary_cross_entropy
optimizer = optim.Adam(model.parameters(), lr=.05, weight_decay=0)
# optimizer = optim.SGD(model.parameters(), lr=.01, momentum=.2, weight_decay=0)

model.train()
for index in range(150):
    optimizer.zero_grad()

    output = model.forward(video_tensor)
    # print(video_tensor.dtype)
    # print(output.dtype)
    loss = criterion(output, video_tensor)

    frame_one = output[0].detach().numpy()

    frame_one = np.transpose(frame_one, )

    print("Step: {}, Loss: {}".format(index, loss))
    loss.backward()
    optimizer.step()

model.eval()
output = model.forward(video_tensor)
array_video = output.detach().numpy()
data.vid_array_to_video("test_me", array_video)


