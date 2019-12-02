import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from data import DataSet

class DecoderSingle(nn.Module):
    """
    Decoder for the auto-encoder/decoder network
    ref https://zo7.github.io/blog/2016/09/25/generating-faces.html
    """
    def __init__(self):  # , input_dim, hidden_dim, num_layers, linear_out, de_conv_stuff, batch_size):
        super(DecoderSingle, self).__init__()

        self.lstm_1 = nn.LSTM(input_size=100, hidden_size=100, num_layers=2, batch_first=True)
        self.unpool_1 = nn.Upsample(scale_factor=5, mode='bilinear')
        self.deconv_1 = nn.ConvTranspose2d(in_channels=4, out_channels=3, kernel_size=200, stride=1)

    def forward(self, input_vec):
        output = input_vec
        output, (_, _) = self.lstm_1(output)  # input of shape (batch, seq_len, input_size)
        output = output.view(8, 4, 5, 5)
        output = self.unpool_1(output)
        output = self.deconv_1(output)

        return output


if __name__ == "__main__":

    # Check for cuda
    if torch.cuda.is_available():
        device = 'cuda'
    else:
        device = 'cpu'

    model = DecoderSingle().to(device)
    criterion = F.mse_loss
    # criterion = F.binary_cross_entropy
    optimizer = optim.SGD(model.parameters(), lr=.01, momentum=.2, weight_decay=0)

    tensor_1 = torch.ones(1, 1, 100)
    tensor_1 = tensor_1.repeat(8, 1, 1)

    # Set values for DataSet object.
    seq_length = 8
    class_limit = 1  # Number of classes to extract. Can be 1-101 or None for all.
    video_limit = 1  # Number of videos allowed per class.  None for no limit
    data = DataSet(seq_length=seq_length, class_limit=class_limit, video_limit=video_limit)

    video_array = None
    for video in data.data:
        video_array = data.video_to_vid_array(video)  # Get numpy array of sequence
        break  # Only need one video to begin with.

    data.vid_array_to_video("ground_truth", video_array)

    video_tensor = torch.from_numpy(video_array).type(torch.float32).to(device)

    model.train()
    for index in range(200):
        optimizer.zero_grad()

        output = model.forward(tensor_1)
        # print(video_tensor.dtype)
        # print(output.dtype)
        loss = criterion(output, video_tensor)
        print("Step: {}, Loss: {}".format(index, loss))
        loss.backward()
        optimizer.step()

    model.eval()
    output = model(tensor_1)
    array_video = output.detach().cpu().numpy()
    data.vid_array_to_video("test_me", array_video)


