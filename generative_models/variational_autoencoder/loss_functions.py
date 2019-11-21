"""
Script defining loss functions
"""
import torch

# Define reconstr loss
reconstruction_loss = torch.nn.BCELoss()

def dfc_loss():
	"""
	This loss function compares intermediate activations 
	of the encoder against a pretrained net
	"""
	pass

def KLD_loss(mu, log_var):
	"""
	Calculate KL-divergence loss
	"""
	kld = mu.pow(2).add_(log_var.exp()).mul_(-1).add_(log_var)
	kld = torch.sum(kld).mul_(-0.5)

	return kld

def loss_function(x_, x, mu, log_var, beta=0.8):
    """
    Calculate loss for batch. Loss is reconstruction + beta * KLD
    """

    loss = reconstruction_loss(x_, x) + beta * KLD_loss(mu, log_var)
    return loss