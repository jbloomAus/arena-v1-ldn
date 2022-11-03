import torchvision
import torch as t
import torch.nn as nn 
from d2 import Conv2d, Flatten, ReLU, MaxPool2d, Linear, MaxPool2d

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

class BatchNorm2d(nn.Module):
    running_mean: t.Tensor         # shape: (num_features,)
    running_var: t.Tensor          # shape: (num_features,)
    num_batches_tracked: t.Tensor  # shape: ()

    def __init__(self, num_features: int, eps=1e-05, momentum=0.1):
        '''Like nn.BatchNorm2d with track_running_stats=True and affine=True.

        Name the learnable affine parameters `weight` and `bias` in that order.
        '''
        super().__init__()
        self.momentum = momentum
        self.eps = eps
        self.num_features = num_features
        self.weight= t.nn.Parameter(t.ones(num_features))
        self.bias = t.nn.Parameter(t.zeros(num_features))
        self.register_buffer('running_mean', t.zeros(num_features))
        self.register_buffer('running_var', t.ones(num_features))
        self.register_buffer('num_batches_tracked', t.tensor(0))


    def forward(self, x: t.Tensor) -> t.Tensor:
        '''Normalize each channel.

        Compute the variance using `torch.var(x, unbiased=False)`
        Hint: you may also find it helpful to use the argument `keepdim`.

        x: shape (batch, channels, height, width)
        Return: shape (batch, channels, height, width)
        '''

        if self.training:
            mean = t.mean(x, (0,2,3), keepdim=True)
            var = t.var(x, (0,2,3), unbiased=False, keepdim=True)
            self.running_mean = (1-self.momentum)*self.running_mean+self.momentum*mean.squeeze()
            self.running_var =(1-self.momentum)*self.running_var+self.momentum*var.squeeze()
            self.num_batches_tracked += 1
        else: 
            mean = self.running_mean.as_strided(x.shape[1:], (1, 0, 0))
            var = self.running_var.as_strided(x.shape[1:], (1, 0, 0))
        
        weight = self.weight.as_strided(x.shape[1:], (1, 0, 0))
        bias = self.bias.as_strided(x.shape[1:], (1, 0, 0))

        return self.scale(self.normalize(x, mean, var), weight, bias)
        #return (x - mean)/t.sqrt(var +self.eps) * weight + bias
    
    def extra_repr(self) -> str:
        return f'num_features {self.num_features}, num batches tracked {self.num_batches_tracked}'

    def normalize(self, x, mean, var):
        return (x - mean) / t.sqrt(self.eps + var)
    
    def scale(self, x, weight, bias):
        return x * weight + bias

class AveragePool(nn.Module):
    def forward(self, x: t.Tensor) -> t.Tensor:
        '''
        x: shape (batch, channels, height, width)
        Return: shape (batch, channels)
        '''
        return t.mean(x, (2,3))

class ResidualBlock(nn.Module):
    def __init__(self, in_feats: int, out_feats: int, first_stride=1):
        '''A single residual block with optional downsampling.

        For compatibility with the pretrained model, declare the left side branch first using a `Sequential`.

        If first_stride is > 1, this means the optional (conv + bn) should be present on the right branch. Declare it second using another `Sequential`.
        '''
        super().__init__()
        self.in_feats = in_feats
        self.out_feats = out_feats
        self.first_stride = 1

        self.left_side = Sequential(
            Conv2d(in_feats, out_feats, kernel_size=(3,3), stride=first_stride, padding=1),
            BatchNorm2d(out_feats),
            ReLU(),
            Conv2d(out_feats, out_feats, kernel_size=(3,3), stride=1, padding=1),
            BatchNorm2d(out_feats),
        )

        if first_stride == 1:
            self.right_side = nn.Identity()
        else:
            self.right_side = Sequential(
                Conv2d(in_feats, out_feats, kernel_size=(1,1), stride=first_stride, padding=0),
                BatchNorm2d(out_feats),
            )

        self.relu = ReLU()

    def forward(self, x: t.Tensor) -> t.Tensor:
        '''Compute the forward pass.

        x: shape (batch, in_feats, height, width)

        Return: shape (batch, out_feats, height / first_stride, width / first_stride)
        '''
        x = self.left_side(x) + self.right_side(x)
        x = self.relu(x)
        return x

class BlockGroup(nn.Module):
    def __init__(self, n_blocks: int, in_feats: int, out_feats: int, first_stride=1):
        '''An n_blocks-long sequence of ResidualBlock where only the first block uses the provided stride.'''
        super().__init__()
        self.n_blocks = n_blocks
        self.in_feats = in_feats
        self.out_feats = out_feats 
        self.first_stride = first_stride

        self.block_one = ResidualBlock(in_feats, out_feats, first_stride)
        self.remaining_blocks = Sequential(*[ResidualBlock(out_feats, out_feats) for i in range(n_blocks-1)])

    def forward(self, x: t.Tensor) -> t.Tensor:
        '''Compute the forward pass.
        x: shape (batch, in_feats, height, width)

        Return: shape (batch, out_feats, height / first_stride, width / first_stride)
        '''
        return self.remaining_blocks(self.block_one(x)) 

class ResNet34(nn.Module):
    def __init__(
        self,
        n_blocks_per_group=[3, 4, 6, 3],
        out_features_per_group=[64, 128, 256, 512],
        first_strides_per_group=[1, 2, 2, 2],
        n_classes=1000,
    ):
        super().__init__()
        self.c1 = Conv2d(3,64,(7,7), stride = 2, padding = 3)
        self.b1 = BatchNorm2d(64)
        self.r1 = ReLU()
        self.m1 = MaxPool2d(3,2)

        self.blocks = Sequential(
            *[BlockGroup(n, i, o, s) for n,i,o,s in zip(
                n_blocks_per_group, 
                [64] + out_features_per_group[:-1],
                out_features_per_group, 
                first_strides_per_group)]
        )

        self.average_pool = AveragePool()
        self.flatten = Flatten()
        self.linear = Linear(
            in_features=out_features_per_group[-1], 
            out_features=n_classes,
            )

    def forward(self, x: t.Tensor) -> t.Tensor:
        '''
        x: shape (batch, channels, height, width)

        Return: shape (batch, n_classes)
        '''
        x = self.m1(self.r1(self.b1(self.c1(x))))
        x = self.blocks(x)
        x = self.linear(self.flatten(self.average_pool(x)))
        return x


def copy_weights(myresnet: ResNet34, pretrained_resnet: torchvision.models.resnet.ResNet) -> ResNet34:
    '''Copy over the weights of `pretrained_resnet` to your resnet.'''

    mydict = myresnet.state_dict()
    pretraineddict = pretrained_resnet.state_dict()

    # Check the number of params/buffers is correct
    assert len(mydict) == len(pretraineddict), "Number of layers is wrong. Have you done the prev step correctly?"

    # Initialise an empty dictionary to store the correct key-value pairs
    state_dict_to_load = {}

    for (mykey, myvalue), (pretrainedkey, pretrainedvalue) in zip(mydict.items(), pretraineddict.items()):
        state_dict_to_load[mykey] = pretrainedvalue

    myresnet.load_state_dict(state_dict_to_load)

    return myresnet