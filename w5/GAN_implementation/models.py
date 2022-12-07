import torch as t
import torch.nn as nn
# from .layers import ConvTranspose2d
from einops.layers.torch import Rearrange
from collections import OrderedDict
# from GAN_implementation.layers import Conv2d, ConvTranspose2d
from torch.nn import Conv2d, ConvTranspose2d
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

        first_height = img_size//(2**n_layers)
        first_width = img_size//(2**n_layers)
        first_size = first_height*first_width*generator_num_features

        # start by project and reshape
        self.project_and_reshape = nn.Sequential(
            nn.Linear(latent_dim_size, first_size, bias=False),
            Rearrange("b (ic h w) -> b ic h w", h=first_height, w=first_width),
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
                bias=   False,
            )

        # use an nn.Sequential to stack the layers
        self.layers = []
        for i in range(self.n_layers):
            layer = nn.Sequential(OrderedDict([
                ("conv", get_layer_i(i)),
                ("bn", nn.BatchNorm2d(self.generator_num_features//(2*2**i)) if i < self.n_layers-1 else nn.Identity()),
                ("relu", nn.ReLU() if i < self.n_layers-1 else nn.Identity())
            ]))
            self.layers.append(layer)

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
        self.rearrange = Rearrange("b ic h w -> b (ic h w)")

        def get_layer_i(i):
            return nn.Conv2d(
                in_channels = self.img_channels if i == 0 else self.generator_num_features//(2**(self.n_layers-i)),
                out_channels = self.generator_num_features//(2**(self.n_layers-i-1)),
                kernel_size = 4,
                stride = 2,
                padding = 1,
            )
        # use an nn.Sequential to stack the layers
        self.layers = []
        for i in range(self.n_layers):
            layer = nn.Sequential(OrderedDict(
                [
                    ("conv", get_layer_i(i)),
                    ("bn", nn.BatchNorm2d(self.generator_num_features//(2**(self.n_layers-i-1))) if i != 0 else nn.Identity()),
                    ("relu", nn.LeakyReLU(0.2)),
                ]
            ))
            self.layers.append(layer)
        
        self.layers = nn.Sequential(*self.layers)
        self.linear = nn.Linear(self.generator_num_features*4*4, 1, bias=False)
        self.sigmoid = nn.Sigmoid()


    def forward(self, x: t.Tensor):
        x = self.layers(x)
        x = self.rearrange(x)
        x = self.linear(x)
        x = self.sigmoid(x)
        
        return x



def initialize_weights_discriminator(model: nn.Module) -> None:
    for i in model.modules():
        if isinstance(i, Conv2d):
            nn.init.normal_(i.weight, 0.0, 0.02)
            if i.bias is not None:
                nn.init.constant_(i.bias, 0)
        elif isinstance(i, nn.BatchNorm2d):
            nn.init.normal_(i.weight, 1.0, 0.02)
            nn.init.constant_(i.bias, 0)
        

import torch.nn as nn
def initialize_weights_generator(model) -> None:
    for i in model.modules():
        if isinstance(i, ConvTranspose2d):
            nn.init.normal_(i.weight, 0.0, 0.02)
        elif isinstance(i, nn.BatchNorm2d):
            nn.init.normal_(i.weight, 1.0, 0.02)
            nn.init.constant_(i.bias, 0)
