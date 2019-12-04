import torchvision
import torch
import os
import torchvision.models as models

"""
########  DEDCODER_01  ###################
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
"""

########  DEDCODER_02  ###################
print("DEDCODER_02")
# CONVOLUTION
print("Convolution")
stride_1 = 1.0
padding_1 = 0.0
dilation_1 = 1.0
kernel_size_1 = 5.0
H_out_u2 = 224.0

H_in_1 = stride_1 * (H_out_u2 - 1) - 2 * padding_1 + dilation_1 * (kernel_size_1 - 1) + 1
print("Side {}".format(H_in_1))
print("Area {}".format(H_in_1 * H_in_1))

# DECONVOLUTION
print("Deconvolution Large")
stride_dl = 1.0
padding_dl = 0.0
dilation_dl = 1.0
kernel_size_dl = 190.0
outpadding_dl = 0.0
H_out_dl = H_in_1

# Deconv
H_in_dl = - (-stride_dl - 2 * padding_dl + dilation_dl * (kernel_size_dl - 1) + outpadding_dl + 1 - H_out_dl) / stride_dl
print(H_in_dl)
print(H_in_dl * H_in_dl)

print("Unpooling Large")
scale_factor_2 = 3

H_out_u2 = H_in_dl
H_in_u2 = H_out_u2 / scale_factor_2
print(H_in_u2)
print(H_in_u2 * H_in_u2)

print("Deconvolution Medium")
stride_dm = 1.0
padding_dm = 0.0
dilation_dm = 1.0
kernel_size_dm = 59.0  # 59 64
outpadding_dm = 0.0
H_out_dm = H_in_1

# Deconv
H_in_dm = - (-stride_dm - 2 * padding_dm + dilation_dm * (kernel_size_dm - 1) + outpadding_dm + 1 - H_out_dm) / stride_dm
print(H_in_dm)
print(H_in_dm * H_in_dm)

print("Unpooling Medium")
scale_factor_2 = 10  # 10

H_out_um = H_in_dm
H_in_um = H_out_um / scale_factor_2
print(H_in_um)
print(H_in_um * H_in_um)

print("Deconvolution Small")
stride_ds = 2.0
padding_ds = 0.0
dilation_ds = 1.0
kernel_size_ds = 10.0
outpadding_ds = 0.0
H_out_ds = H_in_1

# Deconv
H_in_ds = - (-stride_ds - 2 * padding_ds + dilation_ds * (kernel_size_ds - 1) + outpadding_ds + 1 - H_out_ds) / stride_ds
print(H_in_ds)
print(H_in_ds * H_in_ds)

print("Unpooling Small")
scale_factor_2 = 5

H_out_us = H_in_ds
H_in_us = H_out_us / scale_factor_2
print(H_in_us)
print(H_in_us * H_in_us)

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