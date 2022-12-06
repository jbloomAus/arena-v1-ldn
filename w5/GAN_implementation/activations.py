import torch as t
import torch.nn as nn

class Tanh(nn.Module):
    def forward(self, x: t.Tensor) -> t.Tensor:
        return t.tanh(x)

class LeakyReLU(nn.Module):
    def __init__(self, negative_slope: float = 0.01):
        super().__init__()
        self.negative_slope = negative_slope

    def forward(self, x: t.Tensor) -> t.Tensor:
        return t.where(x > 0, x, x * self.negative_slope)
        
    def extra_repr(self) -> str:
        return f"negative_slope={self.negative_slope}"

# w5d1_tests.test_LeakyReLU(LeakyReLU)

class Sigmoid(nn.Module):
    def forward(self, x: t.Tensor) -> t.Tensor:
        return t.sigmoid(x)

# w5d1_tests.test_Sigmoid(Sigmoid)