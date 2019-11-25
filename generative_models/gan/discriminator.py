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
            cnn_embedding_dim = 1024

        elif (cnn_model == "resnet"):
            self.cnn = models.resnet(pretrained=True).features
            cnn_embedding_dim = 1024

        #Activations
        self.relu = nn.ReLU()
        self.leaky_relu = nn.LeakyReLU(0.2)

        #LSTM
        self.LSTM = nn.LSTM(
            input_size=cnn_embedding_dim,
            hidden_size=h_lstm,        
            num_layers=lstm_layers,       
            batch_first=True,       # input & output will has batch size as 1s dimension. e.g. (batch, time_step, input_size)
        )

    def forward(self, x):
        """
        Unroll video tensor and pass through cnn feature extractor
        """

        x = x.view(-1, 3, 250, 250)

        #print (x.shape)

        print (self.cnn(x).shape)
