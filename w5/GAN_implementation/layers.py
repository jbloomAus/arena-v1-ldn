import numpy as np
import torch as t
from typing import Union
import torch.nn as nn

IntOrPair = Union[int, tuple[int, int]]
Pair = tuple[int, int]

from .utils import force_pair
from .functional import conv_transpose2d, conv2d

class ConvTranspose2d(nn.Module):
    def __init__(
        self, in_channels: int, out_channels: int, kernel_size: IntOrPair, stride: IntOrPair = 1, padding: IntOrPair = 0
    ):
        '''
        Same as torch.nn.ConvTranspose2d with bias=False.

        Name your weight field `self.weight` for compatibility with the tests.
        '''
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = force_pair(kernel_size)
        self.stride = force_pair(stride)
        self.padding = force_pair(padding)
        k = 1/(out_channels * np.prod(self.kernel_size))
        self.weight = nn.Parameter((t.rand(in_channels, out_channels, *self.kernel_size)*2-1) * k**0.5, requires_grad=True)

    def forward(self, x: t.Tensor) -> t.Tensor:
        return conv_transpose2d(x, self.weight, stride=self.stride, padding=self.padding)

class Conv2d(nn.Module):
    def __init__(
        self, in_channels: int, out_channels: int, kernel_size: IntOrPair, stride: IntOrPair = 1, padding: IntOrPair = 0
    ):
        '''
        Same as torch.nn.Conv2d with bias=False.

        Name your weight field `self.weight` for compatibility with the PyTorch version.
        '''
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_height, self.kernel_width = force_pair(kernel_size)
        self.n_features = self.kernel_height*self.kernel_width*in_channels
        self.stride = stride
        self.padding = padding
        weight = nn.Parameter((2*t.rand((out_channels, in_channels, self.kernel_height, self.kernel_width))-1))
        weight = weight/np.sqrt(self.n_features)
        self.weight = nn.Parameter(weight)

        
    def forward(self, x: t.Tensor) -> t.Tensor:
        '''Apply the functional conv2d you wrote earlier.'''
        return conv2d(x, self.weight, self.stride, self.padding)

    def extra_repr(self) -> str:
        return f'in_channels {self.in_channels} out_channels {self.out_channels} kernel_size {self.kernel_height, self.kernel_width} padding {self.padding}'
