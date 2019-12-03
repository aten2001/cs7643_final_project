import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from data import DataSet
from encoder_next import Encoder
from decoder_single import DecoderSingle
import sys
import cv2

#model_checker_1 = torchvision.models.vgg16(pretrained=True).features[:9].to('cuda')
#model_checker_2 = torchvision.models.vgg16(pretrained=True).features[:16].to('cuda')
model_checker = torchvision.models.vgg16(pretrained=True).features[:23].to('cuda')
#model_checker_4 = torchvision.models.vgg16(pretrained=True).features[:30].to('cuda')

#print (model_checker)

#exit()

criterion = F.mse_loss

def perceptual_loss(x, x_):
    """
    Compare feature activations of ground truth with output
    """

    feats_x = model_checker(x)
    feats_x_ = model_checker(x_)

    #print (feats_x.shape)

    loss = criterion(feats_x_, feats_x) / (2 * 512*25**2)
    #loss2 = criterion(x_, x)
    #loss3 = criterion(x_, x)
    #loss4 = criterion(x_, x)

    return loss 

class EncoderDecoder(nn.Module):
    """
    Decoder for the auto-encoder/decoder network
    ref https://zo7.github.io/blog/2016/09/25/generating-faces.html
    """
    def __init__(self, seq_length):  # , input_dim, hidden_dim, num_layers, linear_out, de_conv_stuff, batch_size):
        super(EncoderDecoder, self).__init__()

        #cnn_model, input_dim, hidden_dim, lstm_layers, embedding_dim
        self.encoder = Encoder(cnn_model="vgg", input_dim=128, hidden_dim=512, lstm_layers=3, embedding_dim=200, sequence_len=seq_length)

    def forward(self, x):
        """

        :param x: (S,C,H,W)
        :return:
        """

        out = self.encoder.forward(x)
        return out


# Check for cuda
if torch.cuda.is_available():
    device = 'cuda'
else:
    device = 'cpu'

torch.manual_seed(sys.argv[1])

# Set values for DataSet object.
seq_length = 25
class_limit = 101  # Number of classes to extract. Can be 1-101 or None for all.
video_limit = 1  # Number of videos allowed per class.  None for no limit
data = DataSet(seq_length=seq_length, class_limit=class_limit, video_limit=video_limit)

video_array = None
i = 0
for video in data.data:
    video_array = data.video_to_vid_array(video)  # Get numpy array of sequence
    i += 1
    #print (video_array)
    if (i == 21):
        break



print (len(video_array))

video_tensor = torch.from_numpy(video_array).type(torch.float32).to(device)[:-1]

vid_array = video_tensor.detach().cpu().numpy()

data.vid_array_to_video("ground_truth", vid_array)

print (len(video_array))
label = torch.from_numpy(video_array).type(torch.float32).to(device)[1:]

array_video = label.detach().cpu().numpy()
data.vid_array_to_video("test_label", array_video)

model = EncoderDecoder(seq_length - 1).to(device)

# criterion = F.binary_cross_entropy
optimizer = optim.Adam(model.parameters(), lr=.01)
# optimizer = optim.SGD(model.parameters(), lr=.01, momentum=.2, weight_decay=0)

model.train()
for index in range(550):
    optimizer.zero_grad()

    output = model.forward(video_tensor)
    # print(video_tensor.dtype)
    # print(output.dtype)
    loss = criterion(output, label)

    #frame_one = output[0].detach().numpy()

    #frame_one = np.transpose(frame_one, )

    print("Step: {}, Loss: {}".format(index, loss))
    loss.backward()
    optimizer.step()

    

    #print (label.shape)

    #array_video = label.detach().cpu().numpy()
    #data.vid_array_to_video("test_label", array_video)

    #cv2.imshow("next frame", array_video)
    #cv2.waitKey(10)

model.eval()
output = model.forward(video_tensor)
array_video = output.detach().cpu().numpy()
data.vid_array_to_video("test_me", array_video)
   

