"""
Utilies for training
"""

import numpy as np

def dfc_loss():
	pass

def KLD_loss(mu, logvar):
	"""
	Calculate KL-divergence loss
	"""
	kld = mu.pow(2).add_(logvar.exp()).mul_(-1).add_(logvar)
	kld = torch.sum(kld).mul_(-0.5)

	return kld

def train_model(model, training_dataloader, optimizer, epochs):
	pass

def plot_loss(loss):
	"""
	Plots loss to tensorboardX
	"""
	pass

"""
Utility function for computing output of convolutions
takes a tuple of (h,w) and returns a tuple of (h,w)
"""
def conv_output_shape(h_w, kernel_size=1, stride=1, pad=0, dilation=1):
    from math import floor
    if type(kernel_size) is not tuple:
        kernel_size = (kernel_size, kernel_size)
    h = floor( ((h_w[0] + (2 * pad) - ( dilation * (kernel_size[0] - 1) ) - 1 )/ stride) + 1)
    w = floor( ((h_w[1] + (2 * pad) - ( dilation * (kernel_size[1] - 1) ) - 1 )/ stride) + 1)
    return h, w