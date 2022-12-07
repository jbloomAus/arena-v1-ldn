import numpy as np
import torch as t
from typing import Union
from fancy_einsum import einsum
from einops import rearrange, reduce, repeat

IntOrPair = Union[int, tuple[int, int]]
Pair = tuple[int, int]

from .utils import force_pair

def conv1d_minimal(x: t.Tensor, weights: t.Tensor) -> t.Tensor:
    '''Like torch's conv1d using bias=False and all other keyword arguments left at their default values.

    x: shape (batch, in_channels, width)
    weights: shape (out_channels, in_channels, kernel_width)

    Returns: shape (batch, out_channels, output_width)
    '''

    stride = 1
    batch = x.shape[0]
    in_channels = x.shape[1]
    out_channels = x.shape[0]
    width = x.shape[2]
    kernel_width = weights.shape[2]
    output_width = width - kernel_width +1 

    xsB, xsI, xsWi = x.stride()
    x_new_stride = (xsB, xsI, xsWi, xsWi)

    wsO, wsI, wsKw = weights.stride()

    x_strided = x.as_strided(
        (batch, in_channels, output_width, kernel_width),
        x_new_stride
    )

    return einsum("batch in_channels output_width kernel_width, out_channels in_channels kernel_width -> batch out_channels output_width", 
        x_strided, 
        weights
        )

def pad1d(x: t.Tensor, left: int, right: int, pad_value: float) -> t.Tensor:
    '''Return a new tensor with padding applied to the edges.

    x: shape (batch, in_channels, width), dtype float32

    Return: shape (batch, in_channels, left + right + width)
    '''
    batch, in_channels, width = x.shape
    n_x_shape = (batch, in_channels, width+left+right)
    tmp = x.new_full(n_x_shape,fill_value=pad_value)
    tmp[...,left:width+left] = x
    return tmp 

def fractional_stride_1d(x, stride: int = 1):
    '''Returns a version of x suitable for transposed convolutions, i.e. "spaced out" with zeros between its values.
    This spacing only happens along the last dimension.

    x: shape (batch, in_channels, width)

    Example: 
        x = [[[1, 2, 3], [4, 5, 6]]]
        stride = 2
        output = [[[1, 0, 2, 0, 3], [4, 0, 5, 0, 6]]]
    '''
    b,i,w = x.shape
    new_shape = (b,i,w+(w-1)*(stride-1))
    out = t.zeros(new_shape, dtype = x.dtype, device = x.device)

    out[...,::(stride)] = x
    return out

def conv_transpose1d(x, weights, stride: int = 1, padding: int = 0) -> t.Tensor:
    '''Like torch's conv_transpose1d using bias=False and all other keyword arguments left at their default values.

    x: shape (batch, in_channels, width)
    weights: shape (in_channels, out_channels, kernel_width)

    Returns: shape (batch, out_channels, output_width)
    '''
    kernel_width = weights.shape[-1]

    # get x padded 
    pad_length = kernel_width-1-padding
    x_strided = fractional_stride_1d(x, stride)
    x_padded = pad1d(x_strided, left = pad_length, right=pad_length, pad_value=0)
    weights = rearrange(weights,'i o w -> o i w')

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

def conv2d_minimal(x: t.Tensor, weights: t.Tensor) -> t.Tensor:
    '''Like torch's conv2d using bias=False and all other keyword arguments left at their default values.

    x: shape (batch, in_channels, height, width)
    weights: shape (out_channels, in_channels, kernel_height, kernel_width)

    Returns: shape (batch, out_channels, output_height, output_width)
    '''
    
    batch, in_channels, height, width = x.shape
    out_channels, in_channels, kernel_height, kernel_width = weights.shape

    output_width = width - kernel_width +1 
    output_height = height - kernel_height +1 

    xsB, xsI, xsHe, xsWi = x.stride()
    x_new_stride = (xsB, xsI, xsHe, xsWi, xsHe, xsWi)

    wsO, wsI, wsKh, wsKw = weights.stride()

    x_strided = x.as_strided(
        (batch, in_channels, output_height, output_width, kernel_height, kernel_width),
        x_new_stride
    )

    return einsum(
        "batch in_channels output_height output_width kernel_height kernel_width, \
out_channels in_channels kernel_height kernel_width \
-> batch out_channels output_height output_width",
        x_strided, weights
    )
    
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

def fractional_stride_2d(x, stride_h: int, stride_w: int):
    '''
    Same as fractional_stride_1d, except we apply it along the last 2 dims of x (width and height).
    '''
    b,i,h,w = x.shape
    new_shape = (b,i,h+(h-1)*(stride_h-1),w+(w-1)*(stride_w-1))
    out = t.zeros(new_shape, dtype = x.dtype, device = x.device)

    out[...,::(stride_h), ::(stride_w)] = x
    return out

def conv_transpose2d(x, weights, stride: IntOrPair = 1, padding: IntOrPair = 0) -> t.Tensor:
    '''Like torch's conv_transpose2d using bias=False

    x: shape (batch, in_channels, height, width)
    weights: shape (in_channels, out_channels, kernel_height, kernel_width)


    Returns: shape (batch, out_channels, output_height, output_width)
    '''
    in_channeles, out_channels, kernel_height, kernel_width = weights.shape
    stride_h, stride_w = force_pair(stride)
    padding_h, padding_w = force_pair(padding)
    # get x padded 
    
    pad_height = kernel_height-1-padding_h
    pad_width = kernel_width-1-padding_w

    x_strided = fractional_stride_2d(x, stride_h, stride_w)
    x_padded = pad2d(x_strided, left = pad_width, right=pad_width, top=pad_height, bottom=pad_height, pad_value=0)

    weights = rearrange(weights,'i o h w -> o i h w')

    return conv2d(x_padded, weights=weights.flip(-1,-2))

# w5d1_utils.test_conv_transpose2d(conv_transpose2d)