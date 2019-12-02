import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from data import DataSet
from encoder import Encoder
from decoder_single import DecoderSingle
import sys

class EncoderDecoder(nn.Module):
    """
    Decoder for the auto-encoder/decoder network
    ref https://zo7.github.io/blog/2016/09/25/generating-faces.html
    """
    def __init__(self, seq_length):  # , input_dim, hidden_dim, num_layers, linear_out, de_conv_stuff, batch_size):
        super(EncoderDecoder, self).__init__()

        #cnn_model, input_dim, hidden_dim, lstm_layers, embedding_dim
        self.encoder = Encoder(cnn_model="vgg", input_dim=512, hidden_dim=256, lstm_layers=2, embedding_dim=256, sequence_len=seq_length)
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


# Check for cuda
if torch.cuda.is_available():
    device = 'cuda'
else:
    device = 'cpu'

torch.manual_seed(sys.argv[1])

# Set values for DataSet object.
seq_length = 5
class_limit = 1  # Number of classes to extract. Can be 1-101 or None for all.
video_limit = 1  # Number of videos allowed per class.  None for no limit
data = DataSet(seq_length=seq_length, class_limit=class_limit, video_limit=video_limit)

video_array = None
for video in data.data:
    video_array = data.video_to_vid_array(video)  # Get numpy array of sequence
    break  # Only need one video to begin with.

data.vid_array_to_video("ground_truth", video_array)

video_tensor = torch.from_numpy(video_array).type(torch.float32).to(device)

model = EncoderDecoder(seq_length).to(device)
criterion = F.mse_loss
# criterion = F.binary_cross_entropy
optimizer = optim.Adam(model.parameters(), lr=.15, weight_decay=0)
# optimizer = optim.SGD(model.parameters(), lr=.01, momentum=.2, weight_decay=0)

model.train()
for index in range(1500):
    optimizer.zero_grad()

    output = model.forward(video_tensor)
    # print(video_tensor.dtype)
    # print(output.dtype)
    loss = criterion(output, video_tensor)

    print (output.shape)

    #frame_one = output[0].detach().numpy()

    #frame_one = np.transpose(frame_one, )

    print("Step: {}, Loss: {}".format(index, loss))
    loss.backward()
    optimizer.step()

model.eval()
output = model.forward(video_tensor)
array_video = output.detach().cpu().numpy()
data.vid_array_to_video("test_me", array_video)


