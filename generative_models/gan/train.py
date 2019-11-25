from discriminator import Discriminator
from dataset import VideoDataSet
import cv2
import numpy as np
from utils import train_model
import torch.optim as optim
import torch

def draw_circle(center, color):
    """
    Function to create image with circle at center
    """
    image_size = 250

    # Draw a blank canvas
    image = np.ones((image_size, image_size, 3))

    # Draw a circle
    cv2.circle(image, center, 10, color, thickness=30)

    return image

def show_video(sequence, delay):
    """
    Function to display sequence of image frames
    """

    for i in range(len(sequence)):
        image = sequence[i]
        cv2.imshow("image", image)
        cv2.waitKey(delay)

d = Discriminator("vgg", 512, 1).to('cuda')

opt = optim.Adam(d.parameters(), lr=1e-4)

vds = VideoDataSet("data")

train_model(d, vds, opt, 150, 'cuda')

# Test model

for j in range(200):
	way = None

	if (np.random.randint(10) < 6):
		way = False
	
	else:
		way = True
			
	video = []
	y_init = np.random.randint(10, 200)
	x_init = np.random.randint(10, 200)
	color = (0, 100, 0)

	for i in range (0, 200, 10):
		if (way):
			image = draw_circle(((30 + i),y_init), color)
			
		else:
			image = draw_circle((x_init,(30 + i)), color)

		video.append(image)

	show_video(video, 100)

	video = torch.Tensor(video).to('cuda')

	if (d(video).item() < 0.5):
		print ("horizontal")

	else:
		print("vertical")

	print()

	# 0 - horiz
	# 1 - vert