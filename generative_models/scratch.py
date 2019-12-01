import torchvision
import torch
import os
import torchvision.models as models

stride_1 = 1.0
padding_1 = 0.0
dilation_1 = 1.0
kernel_size_1 = 200.0
outpadding_1 = 0.0
H_out = 224.0

# Deconv
H_in = - (-stride_1 - 2 * padding_1 + dilation_1 * (kernel_size_1 - 1) + outpadding_1 + 1 - H_out) / stride_1
print(H_in)
print(H_in * H_in)

scale_factor_2 = 5

H_out = H_in
H_in = H_out / scale_factor_2
print(H_in)
print(H_in * H_in)


# vgg16 = models.vgg16()
#
# print(os.path.exists('data/ucfTrainTestlist/trainlist01.txt'))
#
# clip_length = 5



# ucf101_data = torchvision.datasets.UCF101(root="data/",  annotation_path="data/ucfTrainTestlist",
#                       frames_per_clip=clip_length, frame_rate=None, step_between_clips=clip_length, fold=2, train=True)
# train_loader = torch.utils.data.DataLoader(ucf101_data,
#                                            batch_size=1,
#                                            shuffle=True)
#
#
# for batch_idx, (video, audio, label) in enumerate(train_loader):
#     print("hello world")