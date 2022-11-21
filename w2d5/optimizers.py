# %%



import torch as t
from torch import nn, optim
import numpy as np

import utils

# %% 

def rosenbrocks_banana(x: t.Tensor, y: t.Tensor, a=1, b=100) -> t.Tensor:
    return (a - x) ** 2 + b * (y - x**2) ** 2 + 1

x_range = [-2, 2]
y_range = [-1, 3]
fig = utils.plot_fn(rosenbrocks_apple, x_range, y_range, log_scale=True)
fig.show()