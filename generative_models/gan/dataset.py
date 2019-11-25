"""
Class to load video dataset
"""
from torch.utils.data import Dataset, DataLoader
import os
import numpy as np
import torch

class VideoDataSet(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        len_horiz = len(os.listdir(self.root_dir + "/horiz"))
        len_vert = len(os.listdir(self.root_dir + "/vert"))

        return len_horiz + len_vert

    def __getitem__(self, idx):

        sample = None

        if (idx < 200):
            sample = np.load(self.root_dir + "/horiz/" + "video_" + str(idx) + ".npy")
            sample = torch.Tensor(sample)
            label = torch.zeros(1)
            
        else:
            sample = np.load(self.root_dir + "/vert/" + "video_" + str(idx-200) + ".npy")
            sample = torch.Tensor(sample)
            label = torch.ones(1)

        return sample, label
            



