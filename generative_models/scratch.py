import torchvision
import torch
import os

print(os.path.exists('data/ucfTrainTestlist/trainlist01.txt'))

clip_length = 5

ucf101_data = torchvision.datasets.UCF101(root="data/",  annotation_path="data/ucfTrainTestlist",
                      frames_per_clip=clip_length, frame_rate=None, step_between_clips=clip_length, fold=2, train=True)
train_loader = torch.utils.data.DataLoader(ucf101_data,
                                           batch_size=1,
                                           shuffle=True)


for batch_idx, (video, audio, label) in enumerate(train_loader):
    print("hello world")