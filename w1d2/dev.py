
# %%
import torch as t
from d1 import Embedding, LayerNorm, Dropout, GELU, SWISH
from fancy_einsum import einsum

# Let's start by implementing self attention
# let's start with a function

# %% 


# def single_head_attention(Q: t.Tensor, K: t.Tensor, V: t.Tensor) -> t.Tensor:
#     '''
#     Should return the results of self-attention (see the "Self-Attention in Detail" section of the Illustrated Transformer).

#     With this function, you can ignore masking.

#     Q: shape (FILL THIS IN!)
#     K: shape (FILL THIS IN!)
#     V: shape (FILL THIS IN!)

#     Return: shape (FILL THIS IN!)
#     '''
#     embedding_size = Q.size(-1)
#     scores = einsum('batch i j, batch j k -> batch k i', Q, K.transpose(-2,-1))
#     scores = scores / (embedding_size**.5)
#     scores = t.softmax(scores, dim = -1)
#     attention_values = einsum("batch sq sk, batch sk h -> batch sq h" , scores, V)

#     return attention_values

# query = t.zeros((1,10,100))
# key = t.zeros((1,10,100))
# value = t.ones((1,10,100))

# assert single_head_attention(query, key, value).shape == (1,10,100)

# import plotly.express as px 
# px.imshow(value.squeeze()).show()
# px.imshow(single_head_attention(query, key, value).squeeze())

# # %%

# a = t.randn((5,10,100))
# b = t.randn((5,10,100))
# c = t.randn((5,10,100))
# print(b.shape)
# print(b.transpose(-2,-1).shape)
# print(b.T.shape)
# import numpy as np
# # np.einsum('bij, bjk -> bki', a, b)

# scores = einsum('batch i j, batch j k -> batch k i', a, b.transpose(-2,-1))
# scores.shape # seq len to seq len
# scores = scores / (100**.5)
# scores = t.softmax(scores, dim = -1)
# scores.shape


# attention_values = einsum("batch sq sk, batch sk h -> batch sq h" , scores,c)
# attention_values.shape

# # %%
# %%
import torch as t
import torch.nn as nn
from fancy_einsum import einsum
import einops
import utils as utils
import plotly.express as px

# %%
class PositionalEncoding(nn.Module):

    def __init__(self, max_sequence_length: int = 32, hidden_dim: int = 128):
        super().__init__()
        self.max_sequence_length = max_sequence_length
        self.hidden_dim = hidden_dim
        self.maximal_pe = self.get_maximal_pe()
        #self.d = 10000

    def get_maximal_pe(self):
        
        def PE(delta):
            hidden_dim = self.hidden_dim
            
            sin_vec = t.sin( delta / 10000**(2*t.arange(hidden_dim//2) / hidden_dim ))
            cos_vec = t.cos( delta/ 10000**(2*t.arange(hidden_dim//2) / hidden_dim ))

            pe = t.zeros(hidden_dim)
            pe[::2] = sin_vec
            pe[1::2] = cos_vec

            return pe 
            
        pe = t.stack([PE(i) for i in range(self.max_sequence_length)])
        return pe

    def forward(self, x):
        '''
        x has shape (n, seq_len, hidden_dim)
        '''
        return x + self.maximal_pe[:x.size(1),:]



# %%


def single_head_attention(Q: t.Tensor, K: t.Tensor, V: t.Tensor) -> t.Tensor:
    '''
    Should return the results of self-attention (see the "Self-Attention in Detail" section of the Illustrated Transformer).
    With this function, you can ignore masking.
    Q: shape (batch, seq_len, emb_len)
    K: shape (batch, seq_len, emb_len)
    V: shape (batch, seq_len, emb_len)
    Return: shape (batch, seq_len, emb_len)
    '''

    emb_len = Q.shape[-1]
    QKT = einsum('b s_q h, b h s_k -> b s_q s_k',
                 Q, K.transpose(dim0=-1, dim1=-2))
    QKT = QKT / (emb_len ** 0.5)
    attention_probs = t.softmax(QKT, dim=-1)

    attention_values = einsum(
        'b s_q s_k, b s_k h -> b s_q h', attention_probs, V)
    #attention_values = einsum(' b i j, b k j-> b i k', attention_probs, V)
    return attention_values


b = 64
seq = 12
emb = 8
Q, K, V = t.rand((b, seq, emb)), t.rand((b, seq, emb)), t.rand((b, seq, emb))

av = single_head_attention(Q, K, V)
assert av.shape == t.Size([64, 12, 8])

# %%
Q = t.tensor([[1, 1, 1, 1], [0, 0, 0, 0], [0, 0, 0, 0]],
             dtype=float).unsqueeze(dim=0)
K = t.tensor([[1, 1, 1, 0], [0, 1, 0, 0], [1, 0, 0, 0]],
             dtype=float).unsqueeze(dim=0)
V = t.tensor([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0]],
             dtype=float).unsqueeze(dim=0)

av = single_head_attention(Q, K, V)
# px.imshow(Q.squeeze())
# px.imshow(K.squeeze())

px.imshow(av.squeeze())
#assert av.shape == t.Size([64, 12, 8])

# %%


def single_head_masked_attention(Q: t.Tensor, K: t.Tensor, V: t.Tensor) -> t.Tensor:
    '''
    Should return the results of masked self-attention.
    See "The Decoder Side" section of the Illustrated Transformer for an explanation of masking.
    Q: shape (batch, seq_len, emb_len)
    K: shape (batch, seq_len, emb_len)
    V: shape (batch, seq_len, emb_len)
    Return: shape (batch, seq_len, emb_len)
    '''

    emb_len = Q.shape[-1]
    seq_len = Q.shape[-2]
    QKT = einsum('b s_q h, b h s_k -> b s_q s_k',
                 Q, K.transpose(dim0=-1, dim1=-2))
    QKT = QKT / (emb_len ** 0.5)
    tri = t.tril(t.ones((seq_len, seq_len)), diagonal=0)*(-10 ** 4)
    QKT_masked = (QKT - tri)
    attention_probs = t.softmax(QKT_masked, dim=-1)

    attention_values = einsum(
        'b s_q s_k, b s_k h -> b s_q h', attention_probs, V)
    #attention_values = einsum(' b i j, b k j-> b i k', attention_probs, V)
    return attention_values


# %%
Q = t.tensor([[1, 1, 1, 1], [0, 0, 0, 0], [0, 0, 0, 0]],
             dtype=float).unsqueeze(dim=0)
K = t.tensor([[1, 1, 1, 0], [0, 1, 0, 0], [1, 0, 0, 0]],
             dtype=float).unsqueeze(dim=0)
V = t.tensor([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0]],
             dtype=float).unsqueeze(dim=0)

av = single_head_masked_attention(Q, K, V)
px.imshow(av.squeeze())
# %%
def multihead_masked_attention(Q: t.Tensor, K: t.Tensor, V: t.Tensor, num_heads: int):
    '''
    Implements multihead masked attention on the matrices Q, K and V.
    Q: shape (batch, seq, nheads*headsize)
    K: shape (batch, seq, nheads*headsize)
    V: shape (batch, seq, nheads*headsize)
    Return: shape (batch, seq_len, nheads*headsize)
    '''
    
    emb_len = Q.shape[-1]
    seq_len = Q.shape[-2]
    headsize = emb_len // num_heads

    Q_ = einops.rearrange(Q, 'b s (nh h) -> b nh s h', nh=num_heads)
    K_ = einops.rearrange(K, 'b s (nh h) -> b nh s h', nh=num_heads)
    V_ = einops.rearrange(V, 'b s (nh h) -> b nh s h', nh=num_heads)

    QKT = einsum('b nh s_q h, b nh s_k h -> b nh s_q s_k', Q_, K_)
    QKT = QKT / (headsize ** 0.5)
    #tri = t.triu(t.ones((seq_len, seq_len)), diagonal=1)*(-10 ** 4)
    #QKT_masked = (QKT + tri)
    attention_probs = t.softmax(QKT, dim=-1)

    attention_values_ = einsum(
        'b nh s_q s_k, b nh s_k h -> b nh s_q h', attention_probs, V_)
    # b hn s_q h -->e = n*h --> b s_q e
    attention_values = einops.rearrange(attention_values_, ' b hn s_q h ->  b s_q (hn h)')

    return attention_values

# %%
b = 64
n_h = 2
seq = 12
emb = 8
Q, K, V = t.rand((b, seq, n_h*emb)), t.rand((b, seq, n_h*emb)), t.rand((b, seq, n_h*emb))

av = multihead_masked_attention(Q, K, V, num_heads=n_h)
assert av.shape == t.Size([64, 12, 16])


# %%
class MultiheadMaskedAttention(nn.Module):
    # Hydra-Attention 
    # W_QKV: nn.Linear 
    # W_O: nn.Linear

    def __init__(self, hidden_size: int, num_heads: int):
        super().__init__()

        assert (hidden_size % num_heads) == 0

        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.W_QKV = nn.Linear(hidden_size, 3*hidden_size)
        self.W_O = nn.Linear(hidden_size, hidden_size)

        

    def forward(self, x: t.Tensor) -> t.Tensor:
        '''
        x: shape (batch, seq, hidden_size)
        Return: shape (batch, seq, hidden_size)
        '''
        # QKV = einsum('b s hs, hs h2 -> b (s h2) hs',x, self.W_QKV) # h2 = 3*emb_len 
        QKV = self.W_QKV(x)
        Q, K, V = einops.rearrange(QKV, 'b hs (n es) -> n b hs es', n=3)
        av = multihead_masked_attention(Q, K, V, self.num_heads)
        # av shape: b s_q emb
        # W_O shape: emb, embs
        out = self.W_O(av)
        return out

# %%
b = 128
n_h = 4
seq = 16
emb = 48
hydra = MultiheadMaskedAttention(hidden_size=emb, num_heads=n_h)

x = t.rand((b, seq, emb))
assert hydra(x).shape == t.Size([128, 16, 48])


# Assembling your transformer

# %%
from dataclasses import dataclass


@dataclass(frozen=True)
class TransformerConfig:
    '''Constants used throughout your decoder-only transformer model.'''

    num_layers: int
    num_heads: int
    vocab_size: int
    hidden_size: int
    max_seq_len: int
    dropout: float = 0.1
    layer_norm_epsilon: float = 1e-05


# %% 
import numpy as np 
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

# %%
class DecoderBlock(nn.Module):

    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.config = config 

        self.MLP = MLP(self.config.hidden_size, self.config.dropout)
        self.multiheaded_self_attention = MultiheadMaskedAttention(config.hidden_size, config.num_heads)
        self.layer_norm_1 = LayerNorm(self.config.hidden_size, eps= config.layer_norm_epsilon)
        self.layer_norm_2 = LayerNorm(self.config.hidden_size, eps= config.layer_norm_epsilon)

    def forward(self, x: t.Tensor) -> t.Tensor:

        x = x + self.layer_norm_1(self.multiheaded_self_attention(x))
        x = x + self.layer_norm_2(self.MLP(x))

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

test = MLP(48)

class DecoderOnlyTransformer(nn.Module):

    def __init__(self, config: TransformerConfig):
        super().__init__()
        
        self.config = config 
        self.token_embedding = Embedding(config.vocab_size, embedding_dim= config.hidden_size)
        self.positional_embedding = PositionalEncoding(config.max_seq_len, hidden_dim= config.hidden_size)
        self.dropout = Dropout(p=config.dropout)
        self.decoder_blocks = Sequential(*[DecoderBlock(config) for _ in range(config.num_layers)])
        self.layer_norm_final = LayerNorm(config.hidden_size, config.layer_norm_epsilon)


    def forward(self, x: t.Tensor) -> t.Tensor:
        
        x = self.token_embedding(x)
        x = self.positional_embedding(x)
        x = self.dropout(x)
        x = self.decoder_blocks(x)
        x = self.layer_norm_final(x)

        x = einsum('b seq emb, word emd -> b seq word', x, self.token_embedding.weight)

        return x

# %%

device = t.device("mps" if t.backends.mps.is_available() else "cpu")
print(device)
# %%

import numpy as np 
from torch.utils.data import Dataset
from copy import deepcopy
class RevSequenceDataset(Dataset):
    def __init__(self, seq_length = 5, data_set_size = 10000):
        self.data_set_size = data_set_size
        self.seq_length = seq_length

    def __len__(self):
        return self.data_set_size

    def __getitem__(self, idx):
        sequence = t.tensor(np.random.choice(10,self.seq_length, replace=False))
        rev_sequence = t.flip(sequence, dims = (0,))
        return (sequence, rev_sequence)

dataset = RevSequenceDataset()
dataset[0]

# %%

from torch.utils.data import DataLoader


# %%

batch_size = 32
transformer_config = TransformerConfig(
    num_layers=2,
    num_heads=4,
    vocab_size=10,
    hidden_size=128,
    max_seq_len=5,
    dropout=0.1,
    layer_norm_epsilon=1e-5
)

model = DecoderOnlyTransformer(config=transformer_config)
print(dataset[103][0].unsqueeze(0).shape)
model(dataset[103][0].unsqueeze(0)).shape

# %%

import time

def train_model(model, optimizer, trainloader, testloader, criterion, num_epochs=10, device = "mps"):

    since = time.time()

    print("Beginning Training")
    data = next(iter(testloader))
    example_output = model(data[0])[:3]
    print(data[1][:3])
    print(t.argmax(example_output, dim = -1))
    print("="*30)

    for epoch in range(num_epochs):
        
        model.to(device)
        model.train()
        
        running_loss = 0.0

        for batch, (x, y)  in enumerate(trainloader):
            
            x = x.to(device)
            y = y.to(device)
            
            preds = model(x)
            #print(preds.shape)
            preds_rearranged = einops.rearrange(preds, "b s v -> (b s) v")
            #print(preds_rearranged.shape)
            y_rearranged = einops.rearrange(y, "b s -> (b s)")
            #print(y_rearranged.shape)

            training_loss = criterion(preds_rearranged, y_rearranged)

            training_loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        
            running_loss += training_loss.item() * x.size(0) # scale to n in batch

        epoch_loss = running_loss / len(trainloader.dataset)
        print('Epoch {} Loss: {:.4f}'.format(epoch, epoch_loss))

        data = next(iter(testloader))
        example_output = model(data[0])[:3]
        print(data[1][:3])
        print(t.argmax(example_output, dim = -1))
        print("="*30)

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    #print('Best val Acc: {:4f}'.format(best_acc))   

    return model 


train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
criterion = nn.CrossEntropyLoss()
optimizer = t.optim.Adam(model.parameters(), lr = 0.01)

model = train_model(model, optimizer,  train_loader, test_loader, criterion, num_epochs=10, device="cpu")

# %%

data = next(iter(train_loader))
#example_output = model(next(iter(train_loader))[0])
data[0][0]


# %%
example_output = model(data)
# %%
t.argmax(example_output[0], dim = -1)
# %%
data[0][0]
# %%
data = next(iter(test_loader))
example_output = model(data[0])[:3]
print(data[1][:3])
print(t.argmax(example_output, dim = -1))
# %%


# %% 
px.imshow(example_output[0].detach().numpy()).show()
px.imshow(example_output[1].detach().numpy()).show()
px.imshow(example_output[2].detach().numpy()).show()


# %%
print(t.argmax(example_output[0], dim = -1))
print(t.argmax(example_output[1], dim = -1))
print(t.argmax(example_output[2], dim = -1))


# %%
from torchinfo import torchinfo
torchinfo.summary(model, input_data=t.tensor([[1,2,3,4,5]]))
# %%
