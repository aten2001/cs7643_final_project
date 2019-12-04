import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from data import DataSet
from encoder import Encoder
from decoder import Decoder01, Decoder02

import sys


class EncoderDecoder01(nn.Module):
    """
    Decoder for the auto-encoder/decoder network
    ref https://zo7.github.io/blog/2016/09/25/generating-faces.html
    """
    def __init__(self, batch_size, seq_len):  # , input_dim, hidden_dim, num_layers, linear_out, de_conv_stuff, batch_size):
        super(EncoderDecoder01, self).__init__()
        embedding_dim = 400
        #cnn_model, input_dim, hidden_dim, lstm_layers, embedding_dim
        self.encoder = Encoder(cnn_model="vgg", input_dim=4096, hidden_dim=1024, lstm_layers=5,
                               embedding_dim=embedding_dim, batch_size=batch_size, sequence_len=seq_len)
        self.decoder = Decoder01(batch_size=batch_size, seq_len=seq_len, l_input_size=embedding_dim,
                                 l_hidden_size=400)

    def forward(self, x):
        """

        :param x: (S,C,H,W)
        :return:
        """

        working = self.encoder.forward(x)
        # working = working[-1]
        output = self.decoder.forward(working)

        return output


class EncoderDecoder02(nn.Module):
    """
    Decoder for the auto-encoder/decoder network
    ref https://zo7.github.io/blog/2016/09/25/generating-faces.html
    """
    def __init__(self, batch_size, seq_len):  # , input_dim, hidden_dim, num_layers, linear_out, de_conv_stuff, batch_size):
        super(EncoderDecoder02, self).__init__()
        embedding_dim = 400
        #cnn_model, input_dim, hidden_dim, lstm_layers, embedding_dim
        self.encoder = Encoder(cnn_model="vgg", input_dim=5000, hidden_dim=5000, lstm_layers=2,
                               embedding_dim=embedding_dim, batch_size=batch_size, sequence_len=seq_len)
        decoder_lstm = 22 * 22 * 5 + 17 * 17 * 6 + 13 * 13 * 5
        self.decoder = Decoder02(batch_size=batch_size, seq_len=seq_len, l_input_size=embedding_dim,
                                 l_hidden_size=decoder_lstm, l_num_layers=2)

    def forward(self, x):
        """

        :param x: (S,C,H,W)
        :return:
        """

        working = self.encoder.forward(x)
        # working = working[-1]
        output = self.decoder.forward(working)

        return output

# Check for cuda
if torch.cuda.is_available():
    device = 'cuda'
else:
    device = 'cpu'

# torch.manual_seed(sys.argv[1])

# Set values for DataSet object.
batch_size = 2
seq_length = 2
class_limit = None  # Number of classes to extract. Can be 1-101 or None for all.
video_limit = 2  # Number of videos allowed per class.  None for no limit
data = DataSet(seq_length=seq_length, class_limit=class_limit, video_limit=video_limit)
H, W, C = data.image_shape
video_array = np.zeros((batch_size, seq_length, C, H, W))
i = 0
video_count = 0
for video in data.data:
    if i != 92 and i != 100:
        i += 1
        continue
    else:
        # this_video =
        video_array[video_count] = data.video_to_vid_array(video)  # Get numpy array of sequence
        video_count += 1
        i += 1
        if video_count >= batch_size:
            break

    #print (video_array)
for index in range(video_count):
    data.vid_array_to_video("ground_truth_" + str(index) + "_", video_array[index])

video_tensor = torch.from_numpy(video_array).type(torch.float32).to(device)

model = EncoderDecoder02(batch_size=batch_size, seq_len=seq_length).to(device)
criterion = F.mse_loss
# criterion = F.binary_cross_entropy
optimizer = optim.RMSprop(model.parameters(), lr=.01)
# optimizer = optim.SGD(model.parameters(), lr=.01, momentum=.2, weight_decay=0)

model.train()
for index in range(1000):
    optimizer.zero_grad()

    output = model.forward(video_tensor)
    # print(video_tensor.dtype)
    # print(output.dtype)
    loss = criterion(output, video_tensor)

    #frame_one = output[0].detach().numpy()

    #frame_one = np.transpose(frame_one, )

    print("Step: {}, Loss: {}".format(index, loss))
    loss.backward()
    optimizer.step()

model.eval()
output = model.forward(video_tensor)
array_video = output.detach().cpu().numpy()
for index in range(len(array_video)):
    data.vid_array_to_video("test_me_" + str(index), array_video[index])


