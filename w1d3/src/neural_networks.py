import torch as t
import torch.nn as nn
from .activations import GELU, SWISH
from typing import Union, List

class Embedding(nn.Module):

    def __init__(self, num_embeddings: int, embedding_dim: int):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embeddding_dim = embedding_dim
        self.weight = nn.Parameter(t.randn(num_embeddings,embedding_dim))

    def forward(self, x: t.LongTensor) -> t.Tensor:
        '''For each integer in the input, return that row of the embedding.
        '''
        return self.weight[x]

    def __repr__(self) -> str:
        return f'Embedding({self.num_embeddings}, {self.embeddding_dim})'

class LayerNorm(nn.Module):

    def __init__(self, normalized_shape: Union[int, List[int]], eps: float = 1e-05, elementwise_affine: bool = True):
        super().__init__()
        self.normalized_shape = normalized_shape
        self.eps = eps
        self.elementwise_affine = elementwise_affine
        self.weight = t.nn.Parameter(t.ones(normalized_shape))
        self.bias = t.nn.Parameter(t.zeros(normalized_shape))

    def forward(self, x: t.Tensor) -> t.Tensor:
        '''
        x: (batch, sequence, embedding)
        
        '''

        mean = t.mean(x, dim = -1, keepdim=True)
        var = t.var(x, dim = -1, keepdim=True, unbiased=False)
        x = (x - mean)/(var + self.eps)**0.5
        if self.elementwise_affine:
            x = x*self.weight+self.bias
        return x 
        
    def extra_repr(self) -> str:
        pass

class Dropout(nn.Module):

    def __init__(self, p: float):
        super().__init__()
        self.p = p 

    def forward(self, x: t.Tensor) -> t.Tensor:
        device = x.device
        if self.training:
            mask = (t.rand(x.shape) > self.p).to(device)
            return (x*mask)/(1-self.p)
        else:
            return x 

    def extra_repr(self) -> str:
        return f'Dropout(p = {self.p}'

class Linear(nn.Module):
    def __init__(self, in_features: int, out_features: int, bias=True):
        '''A simple linear (technically, affine) transformation.

        The fields should be named `weight` and `bias` for compatibility with PyTorch.
        If `bias` is False, set `self.bias` to None.
        '''
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter((2*t.rand((out_features, in_features))-1)/(in_features**0.5))
        self.bias = nn.Parameter((2*t.rand((out_features))-1)/(in_features**0.5)) if bias else None

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

class Sequential(nn.Module):
    def __init__(self, *modules: nn.Module):
        super().__init__()
        for i, mod in enumerate(modules):
            self.add_module(str(i), mod)

    def forward(self, x: t.Tensor) -> t.Tensor:
        '''Chain each module together, with the output from one feeding into the next one.'''
        for mod in self._modules.values():
            x = mod(x)
        return x

class MLP(nn.Module):
    def __init__(self, hidden_size: int, dropout: float = 0.1):
        super().__init__()
        self.hidden_size = hidden_size
        self.dropout = dropout

        self.linear_1 = Linear(self.hidden_size, 4*self.hidden_size)
        self.linear_2 = Linear(4*self.hidden_size, self.hidden_size)
        self.gelu = GELU()
        self.dropout = Dropout(p = self.dropout)

    def forward(self, x: t.Tensor) -> t.Tensor:
        x = self.gelu(self.linear_1(x))
        x = self.dropout(self.linear_2(x))
        return x