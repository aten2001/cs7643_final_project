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
    def __init__(self, sequence_length=1, batch_size=1, lstm_input_size=25, lstm_hidden_size=25, lstm_num_layers=2):  # , input_dim, hidden_dim, num_layers, linear_out, de_conv_stuff, batch_size):
        super(Decoder, self).__init__()
        self.sequence_length = sequence_length
        self.batch_size = batch_size
        self.lstm_1 = nn.LSTM(input_size=lstm_input_size, hidden_size=lstm_hidden_size, num_layers=lstm_num_layers,
                              batch_first=True)
        self.unpool_1 = nn.Upsample(scale_factor=5, mode='bilinear')
        self.deconv_1 = nn.ConvTranspose2d(in_channels=1, out_channels=3, kernel_size=200, stride=1)

        # TODO: Consider dropout


    def forward(self, input_vec):
        output = input_vec
        output, (_, _) = self.lstm_1(output)  # input of shape (batch, seq_len, input_size)
        output = output.view(self.sequence_length * self.batch_size, 1, 5, 5)  # Desired shape (seq_len, batch_size, H, W)
        output = self.unpool_1(output)  # (N*B, C, H_{in}, W_{in})    C
        output = self.deconv_1(output)  # (N*B, C_{in}, H_{in}, W_{in}) ->  (N*B, C_{out}, H_{out}, W_{out})
        output = output.view(self.batch_size, self.sequence_length, )


        return output




tensor_1 = torch.ones(1, 1, 25)
tensor_1 = tensor_1.repeat(2, 1, 1)

# Set values for DataSet object.
seq_length = 2
class_limit = 1  # Number of classes to extract. Can be 1-101 or None for all.
video_limit = 2  # Number of videos allowed per class.  None for no limit
data = DataSet(seq_length=seq_length, class_limit=class_limit, video_limit=video_limit)
batch = class_limit * video_limit

model = Decoder(sequence_length=seq_length, batch_size=batch)
criterion = F.mse_loss
# criterion = F.binary_cross_entropy
optimizer = optim.SGD(model.parameters(), lr=.001, momentum=.2, weight_decay=0)

video_batch = np.array((batch, seq_length, data.image_shape[2], data.image_shape[0], data.image_shape[1]))
for i, video in enumerate(data.data):
    this_video = data.video_to_vid_array(video)  # Get numpy array of sequence
    video_batch[i] = np.expand_dims(this_video, 0)  # Add batch dimension
    # break  # Only need one video to begin with.

video_batch = np.transpose(video_batch, )  # (batch, sequence, C, H, W) -> (sequence, batch, C, H, W)

video_tensor = torch.from_numpy(video_batch).type(torch.float32)

model.train()
for index in range(5):
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


