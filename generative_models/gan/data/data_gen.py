"""
Script to generate trivial image sequences for model verfication
"""
import cv2
import numpy as np

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


video = []
color = tuple(np.random.randint(0, 10, 3))
color = (0, 0, 0)

for j in range(200):
    video = []
    y_init = np.random.randint(10, 200)
    x_init = np.random.randint(10, 200)

    for i in range (0, 200, 10):
        image = draw_circle((x_init, 30 + i), color)
        video.append(image)
        
    #show_video(video, 50)

    np.save("vert/video_" + str(j), video)