import torch as t
from torch import nn 
from typing import Union, List

class Embedding(nn.Module):

    def __init__(self, num_embeddings: int, embedding_dim: int):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embeddding_dim = embedding_dim
        self.weight = t.randn(num_embeddings,embedding_dim)

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
        if self.training:
            mask = (t.rand(x.shape) > self.p)
            return (x*mask)/(1-self.p)
        else:
            return x 

    def extra_repr(self) -> str:
        return f'Dropout(p = {self.p}'

class GELU(nn.Module):

    def forward(self, x: t.Tensor) -> t.Tensor:
        return x*0.5*(1+t.erf(x/(2**0.5)))

class SWISH(nn.Module):

    def forward(self, x: t.Tensor) -> t.Tensor:
        return x*t.sigmoid(x)
