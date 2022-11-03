from typing import Optional, Union

import numpy as np
import torch as t
import torch.nn as nn
from fancy_einsum import einsum

IntOrPair = Union[int, tuple[int, int]]
Pair = tuple[int, int]

def force_pair(v: IntOrPair) -> Pair:
    '''Convert v to a pair of int, if it isn't already.'''
    if isinstance(v, tuple):
        if len(v) != 2:
            raise ValueError(v)
        return (int(v[0]), int(v[1]))
    elif isinstance(v, int):
        return (v, v)
    raise ValueError(v)

def maxpool2d(x: t.Tensor, kernel_size: IntOrPair, stride: Optional[IntOrPair] = None, padding: IntOrPair = 0
) -> t.Tensor:
    '''Like PyTorch's maxpool2d.

    x: shape (batch, channels, height, width)
    stride: if None, should be equal to the kernel size

    Return: (batch, channels, out_height, output_width)
    '''

    if stride is None:
        stride = kernel_size

    stride_height, stride_width = force_pair(stride)
    padding_height, padding_width = force_pair(padding)
    kernel_height, kernel_width = force_pair(kernel_size)

    x_padded = pad2d(x, left = padding_width, right=padding_width, top=padding_height, bottom=padding_height, pad_value=-t.inf)

    batch, in_channels, height, width = x_padded.shape
    
    output_width = 1 + (width - kernel_width) // stride_width
    output_height = 1 + (height - kernel_height) // stride_height

    xsB, xsI, xsHe, xsWi = x_padded.stride()
    x_new_stride = (xsB, xsI, xsHe*stride_height, xsWi*stride_width, xsHe, xsWi)


    x_strided = x_padded.as_strided(
        (batch, in_channels, output_height, output_width, kernel_height, kernel_width),
        x_new_stride
    )

    return t.amax(x_strided, dim = (-1,-2))

def pad2d(x: t.Tensor, left: int, right: int, top: int, bottom: int, pad_value: float) -> t.Tensor:
    '''Return a new tensor with padding applied to the edges.

    x: shape (batch, in_channels, height, width), dtype float32

    Return: shape (batch, in_channels, top + height + bottom, left + width + right)
    '''

    batch, in_channels, height, width = x.shape
    n_x_shape = (batch, in_channels, height+top+bottom, width+left+right)
    tmp = x.new_full(n_x_shape, fill_value = pad_value)
    tmp[..., top : top + height, left: left + width] = x
    return tmp 

def conv2d(x, weights, stride: IntOrPair = 1, padding: IntOrPair = 0) -> t.Tensor:
    '''Like torch's conv2d using bias=False

    x: shape (batch, in_channels, height, width)
    weights: shape (out_channels, in_channels, kernel_height, kernel_width)


    Returns: shape (batch, out_channels, output_height, output_width)
    '''

    stride_height, stride_width = force_pair(stride)
    padding_height, padding_width = force_pair(padding)

    x_padded = pad2d(x, left = padding_width, right=padding_width, top=padding_height, bottom=padding_height, pad_value=0)

    batch, in_channels, height, width = x.shape
    out_channels, in_channels, kernel_height, kernel_width = weights.shape

    output_width = int(np.floor((width +2*padding_width- kernel_width)/stride_width) +1 )
    output_height = int(np.floor((height +2*padding_height- kernel_height)/stride_height) +1 )

    xsB, xsI, xsHe, xsWi = x_padded.stride()
    x_new_stride = (xsB, xsI, xsHe*stride_height, xsWi*stride_width, xsHe, xsWi)

    wsO, wsI, wsKh, wsKw = weights.stride()

    x_strided = x_padded.as_strided(
        (batch, in_channels, output_height, output_width, kernel_height, kernel_width),
        x_new_stride
    )

    return einsum("batch in_channels output_height output_width kernel_height kernel_width, \
                    out_channels in_channels kernel_height kernel_width -> batch out_channels output_height output_width", 
        x_strided, 
        weights
        )

class MaxPool2d(nn.Module):
    def __init__(self, kernel_size: IntOrPair, stride: Optional[IntOrPair] = None, padding: IntOrPair = 1):
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride 
        self.padding = padding 

    def forward(self, x: t.Tensor) -> t.Tensor:
        '''Call the functional version of maxpool2d.'''
        return maxpool2d(x, self.kernel_size, self.stride, self.padding)

    def extra_repr(self) -> str:
        '''Add additional information to the string representation of this class.'''
        return f'Stride {self.stride}, kernel_size {self.kernel_size}, padding {self.padding}'



class ReLU(nn.Module):
    def forward(self, x: t.Tensor) -> t.Tensor:
        return nn.ReLU()(x)


class Flatten(nn.Module):
    def __init__(self, start_dim: int = 1, end_dim: int = -1) -> None:
        super().__init__()
        self.start_dim = start_dim
        self.end_dim = end_dim

    def forward(self, input: t.Tensor) -> t.Tensor:
        '''Flatten out dimensions from start_dim to end_dim, inclusive of both.
        '''

        if self.end_dim == -1:
            self.end_dim = len(input.shape)
        old_shape = input.shape
        new_shape = old_shape[:self.start_dim] + (-1,) + old_shape[self.end_dim+1:]
        return t.reshape(input, new_shape)

    def extra_repr(self) -> str:
        return f'Start Dim {self.start_dim}, End Dim {self.end_dim}'

class Linear(nn.Module):
    def __init__(self, in_features: int, out_features: int, bias=True):
        '''A simple linear (technically, affine) transformation.

        The fields should be named `weight` and `bias` for compatibility with PyTorch.
        If `bias` is False, set `self.bias` to None.
        '''
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter((2*t.rand((out_features, in_features))-1)/np.sqrt(in_features))
        self.bias = nn.Parameter((2*t.rand((out_features))-1)/np.sqrt(in_features)) if bias else None

    def forward(self, x: t.Tensor) -> t.Tensor:
        '''
        x: shape (*, in_features)
        Return: shape (*, out_features)
        '''
        if self.bias is not None:
            return t.matmul(x, self.weight.T) + self.bias
        else:
            return t.matmul(x, self.weight.T)

    def extra_repr(self) -> str:
        return f"in_features {self.in_features}, out_features {self.out_features}. bias {self.bias is not None}"


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
