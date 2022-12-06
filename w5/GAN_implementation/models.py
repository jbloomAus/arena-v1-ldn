import torch as t
import torch.nn as nn
from .layers import ConvTranspose2d
from einops.layers.torch import Rearrange
class Generator(nn.Module):

    def __init__(
        self,
        latent_dim_size: int,           # size of the random vector we use for generating outputs
        img_size = int,                 # size of the images we're generating
        img_channels = int,             # indicates RGB images
        generator_num_features = int,   # number of channels after first projection and reshaping
        n_layers = int,                 # number of CONV_n layers
    ):
        super().__init__()
        self.latent_dim_size = latent_dim_size
        self.img_size = img_size
        self.img_channels = img_channels
        self.generator_num_features = generator_num_features
        self.n_layers = n_layers

        # self.projection = nn.Linear(self.latent_dim_size, self.generator_num_features*4*4)
        self.activation = nn.ReLU()

        # start by project and reshape
        self.project_and_reshape = nn.Sequential(
            nn.Linear(latent_dim_size, self.generator_num_features * 4 * 4, bias=False),
            Rearrange("b (ic h w) -> b ic h w", h=4, w=4),
            nn.BatchNorm2d(generator_num_features),
            nn.ReLU(),
        )

        def get_layer_i(i):
            return ConvTranspose2d(
                in_channels = self.generator_num_features//(2**i),
                out_channels = self.generator_num_features//(2*2**i) if i < self.n_layers-1 else self.img_channels,
                kernel_size = 4,
                stride = 2,
                padding = 1,
            )
        # use an nn.Sequential to stack the layers
        self.layers = [
            nn.Sequential(*[get_layer_i(i), 
            nn.BatchNorm2d(self.generator_num_features//(2*2**i)) if i < self.n_layers-1 else nn.Identity(),
            nn.ReLU()]) 
            for i in range(self.n_layers)]
        self.layers = nn.Sequential(*self.layers)
        self.tanh = nn.Tanh()

    def forward(self, x: t.Tensor) -> t.Tensor:
        x = self.project_and_reshape(x)
        x = self.layers(x)
        x = self.tanh(x)
        return x

class Discriminator(nn.Module):

    def __init__(
        self,
        img_size = 64,
        img_channels = 3,
        generator_num_features = 1024,
        n_layers = 4,
    ):
        super().__init__()
        self.img_size = img_size
        self.img_channels = img_channels
        self.generator_num_features = generator_num_features
        self.n_layers = n_layers

        self.activation = nn.LeakyReLU(0.2)

        def get_layer_i(i):
            return nn.Conv2d(
                in_channels = self.img_channels if i == 0 else self.generator_num_features//(2**(self.n_layers-i)),
                out_channels = self.generator_num_features//(2**(self.n_layers-i-1)),
                kernel_size = 4,
                stride = 2,
                padding = 1,
            )
        # use an nn.Sequential to stack the layers
        self.layers = [
            nn.Sequential(*[get_layer_i(i),
            nn.BatchNorm2d(self.generator_num_features//(2**(self.n_layers-i-1))) if i != 0  else nn.Identity(),
            nn.LeakyReLU(0.2)])
            for i in range(self.n_layers)]
        
        self.layers = nn.Sequential(*self.layers)
        self.linear = nn.Linear(self.generator_num_features*4*4, 1, bias=False)
        self.sigmoid = nn.Sigmoid()


    def forward(self, x: t.Tensor):
        x = self.layers(x)
        x = x.view(x.shape[0], -1)
        x = self.linear(x)
        x = self.sigmoid(x)
        
        return x
