import torch.nn as nn
import torch as t
from typing import Tuple

from torch.nn import Conv2d, Flatten, Linear, ConvTranspose2d, ReLU, Unflatten

class VAE(nn.Module):

    def __init__(self, latent_dim_size = 5):
        super().__init__()
        self.latent_dim_size = latent_dim_size
        self.encoder_base = nn.Sequential(
            Conv2d(1, 16, kernel_size=4, stride=2, padding=1),
            Conv2d(16, 32, kernel_size=4, stride=2, padding=1),
            Flatten(),
            Linear(32*7*7, 100),
        )

        self.encoder_mu = nn.Linear(100,latent_dim_size)
        self.encoder_sigma = nn.Linear(100,latent_dim_size) # outputs log(sigma)

        self.decoder = nn.Sequential(
            Linear(latent_dim_size, 100),
            Linear(100, 32*7*7),
            nn.ReLU(),
            Unflatten(1, (32, 7, 7)),
            ConvTranspose2d(32, 16, kernel_size=4, stride=2, padding=1),
            ConvTranspose2d(16, 1, kernel_size=4, stride=2, padding=1)
        )
    
    def encoder(self, x):
        '''
        A utility method that returns samples from the latent vector z
        '''
        x = self.encoder_base(x)
        mu = self.encoder_mu(x)
        logsigma = self.encoder_sigma(x)
        z = mu + t.exp(logsigma) * t.randn_like(mu)
        return z


    def forward(self, x: t.Tensor) -> Tuple[t.Tensor, t.Tensor, t.Tensor]:
        '''
        Returns a tuple of (mu, logsigma, z), where:
            mu and logsigma are the outputs of your encoder module
            z is the sampled latent vector taken from distribution N(mu, sigma**2)
        '''
        x = self.encoder_base(x)
        mu = self.encoder_mu(x)
        logsigma = self.encoder_sigma(x)
        z = mu + t.exp(logsigma) * t.randn_like(mu)
        output = self.decoder(z)
        return mu, logsigma, output