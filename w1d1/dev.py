
# %%
import transformers


# %%
tokenizer = transformers.AutoTokenizer.from_pretrained("gpt2")
encoding = tokenizer.encode("I am legend.")
print(encoding)

original_text = tokenizer.decode(encoding)
print(original_text)
# %%
tokens = tokenizer.tokenize(original_text)
tokens
# %%
encoding = [tokenizer.vocab[token] for token in tokens]
original_text = tokenizer.decode(encoding)
print(original_text)

# Let's do some experiments with cosine similarity in high dimensions
# %%

import torch 
from torch.nn.functional import cosine_similarity
import plotly.express as px 

def get_average_cosine(len_vector, samples = 20):
    a = torch.randn((samples,len_vector))
    a = a / torch.norm(a, dim = 0)
    b = torch.randn((samples,len_vector))
    b = b / torch.norm(b, dim = 0)
    return torch.mean(torch.abs(cosine_similarity(a,b, dim = 1))).item()

go_to_2_to_the = 8
average_cosines = [get_average_cosine(i**2) for i in range(go_to_2_to_the)]
px.line(x = [i**2 for i in range(go_to_2_to_the)], y = average_cosines, log_y=True, 
    labels={"x":"length of vector", "y": "cosine similarity"}, 
    title = "Average Absolute Cosine Similarity over Vector Length",
    template="plotly_dark")


# Looking at embeddings
# https://medium.com/mlearning-ai/load-pre-trained-glove-embeddings-in-torch-nn-embedding-layer-in-under-2-minutes-f5af8f57416a
# https://nlp.stanford.edu/projects/glove/

# %%
# !wget http://nlp.stanford.edu/data/glove.6B.zip
# !unzip glove.6B.zip
# !ls -lat

# %%
vocab,embeddings = [],[]
with open('glove.6B.300d.txt','rt') as fi:
    full_content = fi.read().strip().split('\n')
for i in range(len(full_content)):
    i_word = full_content[i].split(' ')[0]
    i_embeddings = [float(val) for val in full_content[i].split(' ')[1:]]
    vocab.append(i_word)
    embeddings.append(i_embeddings)
# %%

import numpy as np
vocab_npa = np.array(vocab)
embs_npa = np.array(embeddings)


#insert '<pad>' and '<unk>' tokens at start of vocab_npa.
vocab_npa = np.insert(vocab_npa, 0, '<pad>')
vocab_npa = np.insert(vocab_npa, 1, '<unk>')
print(vocab_npa[:10])

pad_emb_npa = np.zeros((1,embs_npa.shape[1]))   #embedding for '<pad>' token.
unk_emb_npa = np.mean(embs_npa,axis=0,keepdims=True)    #embedding for '<unk>' token.

#insert embeddings for pad and unk tokens at top of embs_npa.
embs_npa = np.vstack((pad_emb_npa,unk_emb_npa,embs_npa))
print(embs_npa.shape)


# %% 
import torch
my_embedding_layer = torch.nn.Embedding.from_pretrained(torch.from_numpy(embs_npa).float())

assert my_embedding_layer.weight.shape == embs_npa.shape
print(my_embedding_layer.weight.shape)

# %% 
with open('vocab_npa.npy','wb') as f:
    np.save(f,vocab_npa)

with open('embs_npa.npy','wb') as f:
    np.save(f,embs_npa)

# %%

my_embedding_layer
# %%
vocab_dict_i_to_w = {i:vocab_npa[i] for i in range(len(vocab_npa))}
vocab_dict_w_to_i = {vocab_npa[i]:i for i in range(len(vocab_npa))}
vocab_dict_w_to_i["against"]
# %%
words = ["tall","short","big","small"]
indices = torch.tensor([vocab_dict_w_to_i[w] for w in words])
print(indices.shape)
indices

# %%
embeddings = my_embedding_layer.forward(indices)
embeddings.shape
# %%
type(my_embedding_layer)

# %%
difference = embeddings[0] - embeddings[1]
difference.shape
# %%
my_embedding_layer.weight.T.shape
# %%
difference_output = difference @ my_embedding_layer.weight.T
difference_output.shape

biggest_term = torch.argmax(difference_output)

vocab_dict_i_to_w[biggest_term.item()]

# %%
words = ["doctor","nurse"]
indices = torch.tensor([vocab_dict_w_to_i[w] for w in words])
embeddings = my_embedding_layer.forward(indices)
difference = embeddings[0] - embeddings[1]
distance = torch.norm(my_embedding_layer.weight.data - difference, dim=1)
nearest = torch.argmin(distance)
vocab_dict_i_to_w[nearest.item()]

# %%
import torch as t
from torch import nn 
import utils
class Embedding(nn.Module):

    def __init__(self, num_embeddings: int, embedding_dim: int):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embeddding_dim = embedding_dim
        self.weight = torch.randn(num_embeddings,embedding_dim)

    def forward(self, x: t.LongTensor) -> t.Tensor:
        '''For each integer in the input, return that row of the embedding.
        '''
        return [self.weight[i] for i in x]
        

    def __repr__(self) -> str:
        return f'Embedding({self.num_embeddings}, {self.embeddding_dim})'


assert repr(Embedding(10, 20)) == repr(t.nn.Embedding(10, 20))
utils.test_embedding(Embedding)

# %%

class PositionalEncoding(nn.Module):

    def __init__(self, max_sequence_length: int = 32, hidden_dim: int = 128):
        super().__init__()
        self.max_sequence_length = max_sequence_length
        self.hidden_dim = 128
        self.maximal_pe = self.get_maximal_pe()

    def get_maximal_pe(self):
 
        def PE(delta):
            hidden_dim = self.hidden_dim
            max_seq_length
            sin_vec = torch.sin( delta / d **(2*torch.arange(hidden_dim//2) / hidden_dim ))
            cos_vec = torch.cos( delta/ d **(2*torch.arange(hidden_dim//2) / hidden_dim ))

            pe = torch.zeros(hidden_dim)
            pe[::2] = sin_vec
            pe[1::2] = cos_vec

            return pe 
            
        pe = torch.stack([PE(i) for i in range(self.max_seq_length)])
        return pe

    def forward(self, x):
        '''
        x has shape (n, seq_len, hidden_dim)
        '''
        return x + self.pe[:,x.size(0), x.size(1)]



# %%
max_seq_length = 32
hidden_dim = 128
d = 10000

def PE(delta):
    sin_vec = torch.sin( delta / d **(2*torch.arange(hidden_dim//2) / hidden_dim ))
    cos_vec = torch.cos( delta/ d **(2*torch.arange(hidden_dim//2) / hidden_dim ))

    pe = torch.zeros(hidden_dim)
    pe[::2] = sin_vec
    pe[1::2] = cos_vec

    return pe 
    
pe = torch.stack([PE(i) for i in range(max_seq_length)])
pe


px.imshow(pe, color_continuous_scale="RdBu")


# show that the pe dot product decays for over distance in sequence
# %%

from torch.nn.functional import cosine_similarity

vec = torch.Tensor([cosine_similarity(pe[i], pe[j], dim = 0) for i in range(max_seq_length) for j in range(max_seq_length)])
px.imshow(vec.reshape(32,32))


# %%
from torch.nn import LayerNorm

layer_norm = LayerNorm(10)
batches = 10
sequence = 100
embedding = 10
a = torch.randn((batches, sequence, embedding))*100000000
print(torch.mean(torch.mean(a, dim = 2)))
print(torch.mean(torch.var(a, dim = 2)))
a_norm = layer_norm(a)
print(torch.mean(torch.mean(a_norm, dim = 2)))
print(torch.mean(torch.var(a_norm, dim = 2)))


# %%
from typing import Union, List
from torch import nn 

class LayerNorm(nn.Module):

    def __init__(self, normalized_shape: Union[int, List[int]], eps: float = 1e-05, elementwise_affine: bool = True):
        super().__init__()
        self.normalized_shape = normalized_shape
        self.eps = eps
        self.elementwise_affine = elementwise_affine
        self.weight = torch.nn.Parameter(torch.ones(normalized_shape))
        self.bias = torch.nn.Parameter(torch.zeros(normalized_shape))

    def forward(self, x: t.Tensor) -> t.Tensor:
        '''
        x: (batch, sequence, embedding)
        
        '''

        mean = torch.mean(x, dim = -1, keepdim=True)
        var = torch.var(x, dim = -1, keepdim=True, unbiased=False)
        x = (x - mean)/(var + self.eps)**0.5
        if self.elementwise_affine:
            x = x*self.weight+self.bias
        return x 
        
    def extra_repr(self) -> str:
        pass

utils.test_layernorm_mean_1d(LayerNorm)
utils.test_layernorm_mean_2d(LayerNorm)
utils.test_layernorm_std(LayerNorm)
utils.test_layernorm_exact(LayerNorm)
utils.test_layernorm_backward(LayerNorm)
# %%
class Dropout(nn.Module):

    def __init__(self, p: float):
        super().__init__()
        self.p = p 

    def forward(self, x: t.Tensor) -> t.Tensor:
        if self.training:
            mask = (torch.rand(x.shape) > self.p)
            return (x*mask)/(1-self.p)
        else:
            return x 

    def extra_repr(self) -> str:
        return f'Dropout(p = {self.p}'

utils.test_dropout_eval(Dropout)
utils.test_dropout_training(Dropout)
# %%
class GELU(nn.Module):

    def forward(self, x: t.Tensor) -> t.Tensor:
        return x*0.5*(1+torch.erf(x/(2**0.5)))

utils.plot_gelu(GELU)



# %% 
class SWISH(nn.Module):

    def forward(self, x: t.Tensor) -> t.Tensor:
        return x*torch.sigmoid(x)

utils.plot_gelu(SWISH)

