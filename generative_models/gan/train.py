from discriminator import Discriminator
from dataset import VideoDataSet
import cv2
import numpy as np

def show_video(sequence, delay):
    """
    Function to display sequence of image frames
    """

    for i in range(len(sequence)):
        image = sequence[i]
        cv2.imshow("image", image)
        cv2.waitKey(delay)

d = Discriminator("vgg", 16, 3)

vds = VideoDataSet("data")


data, _ = vds[309]

d(data)

#show_video(data,100)