"""
Class to define GAN discriminator
"""

import torch
from torchvision import models
import torch.nn as nn
import numpy

class Discriminator(nn.Module):
    def __init__(self, cnn_model, h_lstm, lstm_layers):
        super(Discriminator, self).__init__()

        # Convolutions (pre-trained)
        cnn_embedding_dim = None

        if (cnn_model == "vgg"):
            self.cnn = models.vgg16(pretrained=True).features
            cnn_embedding_dim = 25088

        elif (cnn_model == "resnet"):
            self.cnn = models.resnet(pretrained=True).features
            cnn_embedding_dim = 1024

        self.fc1 = nn.Linear(cnn_embedding_dim, 512)
        self.fc2 = nn.Linear(h_lstm, 256)
        self.fc3 = nn.Linear(256, 1)

        #Activations
        self.relu = nn.ReLU()
        self.leaky_relu = nn.LeakyReLU(0.2)

        #LSTM
        self.LSTM = nn.LSTM(
            input_size=512,
            hidden_size=h_lstm,        
            num_layers=lstm_layers,       
            batch_first=True,       # input & output will has batch size as 1s dimension. e.g. (batch, time_step, input_size)
        )

        # Lock conv layers
        self.cnn.eval()

    def forward(self, x):
        """
        Unroll video tensor and pass through cnn feature extractor
        """

        x = x.view(-1, 3, 250, 250)

        #print (x.shape)

        conv_feats = self.cnn(x).view(20,-1)
        embedding = self.fc1(conv_feats)
        embedding = self.relu(embedding)

        hidden = None

        for i in range(embedding.shape[0]):
            #print (embedding[i].view(1, 1, -1).shape)
            out, hidden = self.LSTM(embedding[i].view(1, 1, -1), hidden)
            
        out = self.fc2(out)
        out = self.relu(out)
        out = self.fc3(out)
        out = torch.sigmoid(out)

        return out