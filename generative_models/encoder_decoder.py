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
        self.encoder = Encoder(cnn_model="vgg", input_dim=4096, hidden_dim=1024, lstm_layers=5, embedding_dim=400, sequence_len=seq_length)
        self.decoder = DecoderSingle(400, seq_length)

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
seq_length = 7
class_limit = 101  # Number of classes to extract. Can be 1-101 or None for all.
video_limit = 1  # Number of videos allowed per class.  None for no limit
data = DataSet(seq_length=seq_length, class_limit=class_limit, video_limit=video_limit)

video_array = None
i = 0
for video in data.data:
    video_array = data.video_to_vid_array(video)  # Get numpy array of sequence
    i += 1
    #print (video_array)
    if (i == 92):
        break

data.vid_array_to_video("ground_truth", video_array)

video_tensor = torch.from_numpy(video_array).type(torch.float32).to(device)

model = EncoderDecoder(seq_length).to(device)
criterion = F.mse_loss
# criterion = F.binary_cross_entropy
optimizer = optim.RMSprop(model.parameters(), lr=.01)
# optimizer = optim.SGD(model.parameters(), lr=.01, momentum=.2, weight_decay=0)

model.train()
for index in range(250):
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
data.vid_array_to_video("test_me", array_video)


