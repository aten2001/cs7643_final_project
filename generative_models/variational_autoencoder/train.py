"""
Script to train a model
"""

from vae import VariationalAutoencoder as vae
from utils import train_model
import torch
import torchvision
import argparse
import torch.optim as optim

model = vae(latent_dim, use_dfc=False)

dataloader = None
opt = optim.Adam(model.parameters(), lr=lr)

train_model(model, dataloader, opt, 50000)