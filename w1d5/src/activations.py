import torch as t
import torch.nn as nn

class GELU(nn.Module):

    def forward(self, x: t.Tensor) -> t.Tensor:
        return x*0.5*(1+t.erf(x/(2**0.5)))

class SWISH(nn.Module):

    def forward(self, x: t.Tensor) -> t.Tensor:
        return x*t.sigmoid(x)
