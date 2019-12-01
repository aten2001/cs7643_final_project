"""
This script demonstrates how to used data.py to load and save
video (series of images) files.

You can change you sequence length of the vidoes, limit to a set number
of classes, and limit the amount of videos per class below.

class_limit is an integer that denotes the first N classes you want to
extract features from. This is useful is you don't want to wait to
extract all 101 classes. For instance, set class_limit = 8 to just
extract features for the first 8 (alphabetical) classes in the dataset.
Then set the same number when training models.
"""
import numpy as np
import os.path
from data import DataSet
# from extractor import Extractor
# from tqdm import tqdm  # Decent loader bar.  Uncomment all tqdm stuff to use.

# Set defaults.
seq_length = 40
class_limit = 1  # Number of classes to extract. Can be 1-101 or None for all.
video_limit = 1  # Number of videos allowed per class.  None for no limit

# Get the dataset.
data = DataSet(seq_length=seq_length, class_limit=class_limit, video_limit=video_limit)

# Loop through data.
# pbar = tqdm(total=len(data.data))
for video in data.data:

    video_array = data.video_to_vid_array(video)  # Get numpy array of sequence
    data.vid_array_to_video(video[2], video_array)  # Save numpy array to picture files

    # pbar.update(1)

# pbar.close()
