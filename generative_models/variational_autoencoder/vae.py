import torch
import torch.nn as nn

class VariationalAutoencoder(nn.Module):
	"""
	
	Class to implement variational autoencoder
	"""
	def __init__(self, arg, latent_dim, use_dfc):
		super(VariationalAutoencoder, self).__init__()
		self.latent_dim = latent_dim
		self.use_dfc = use_dfc

		# Encoder

		self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=2)
		self.conv1 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=2)
		self.conv1 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=2)

		self.fc1 = nn.Linear(1000, 512)
		self.fc2 = nn.Linear(512, self.latent_dim)

		# Decoder

		self.fc3 = nn.Linear(self.latent_dim, 128)
		self.fc3 = nn.Linear(128, 512)

	def forward(self):
		"""
		Forward pass
		"""
		pass

	def reparameterize(self):
		"""
		Perform reparam trick to preserve gradient
		"""
		pass


	def calculate_latent_variable(self):
		"""
		Calculate the latent distribution vector
		"""
		pass