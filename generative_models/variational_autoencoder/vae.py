import torch
import torch.nn as nn

class VariationalAutoencoder(nn.Module):
	"""
	Lightly based on https://github.com/bhpfelix/Variational-Autoencoder-PyTorch/blob/master/src/vanila_vae.py
	Class to implement variational autoencoder
	"""
	def __init__(self, arg, latent_dim, use_dfc, input_channels=3):
		super(VariationalAutoencoder, self).__init__()
		self.latent_dim = latent_dim
		self.use_dfc = use_dfc
		self.input_channels = input_channels

		# Encoder

		# Convolutions
		self.conv1 = nn.Conv2d(in_channels=input_channels, out_channels=32, kernel_size=3, stride=2)
		self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=2)
		self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=2)

		# Fully connected
		self.fc1 = nn.Linear(512, self.latent_dim)

		#Activations
		self.relu = nn.ReLU()
		self.leaky_relu = nn.LeakyReLU(0.2)

		# Decoder
		
		# Fully Connected
		self.fc_mu = nn.Linear(128, self.latent_dim)
		self.fc_sig = nn.Linear(128, self.latent_dim)

		self.deconv1 = nn.ConvTranspose2d(input_channels=1, out_channels=32, kernel_size=3, stride=2)
		self.deconv2 = nn.ConvTranspose2d(input_channels=32, out_channels=16, kernel_size=3, stride=2)
		self.deconv3 = nn.ConvTranspose2d(input_channels=16, out_channels=3, kernel_size=3, stride=2)

	def encode(self, x, leaky=False):
		"""
		Encode image
		"""

		x = self.conv1(x)
		if (leaky):
			x = self.leaky_relu(x)
		else:
			x = self.relu(x)

		x = self.conv2(x)
		if (leaky):
			x = self.leaky_relu(x)
		else:
			x = self.relu(x)

		x = self.conv3(x)
		if (leaky):
			x = self.leaky_relu(x)
		else:
			x = self.relu(x)

		return self.fc_mu(x), self.fc_sig(x)


	def decode(self, z):
		"""
		Decode sampled latent vector and return image
		"""
		pass

	def forward(self, x):
		"""
		Forward pass
		"""
		# Encode the image
		mu, log_var = self.encode(x)

		# Reparameterize latent vector
		z = self.reparameterize(mu, log_var)

		# Decode sampled latent vector
		out = self.Decode(z)

		return out, mu, log_var

	def reparameterize(self, mu, log_var):
		"""
		Perform reparam trick to preserve gradient
		"""
		std_dev = torch.exp(0.5 * log_var)
		eps = torch.randn_like(std_dev)

		return mu + eps*std_dev


	def calculate_latent_variable(self):
		"""
		Calculate the latent distribution vector
		"""
		pass