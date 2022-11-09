import torch as t
import torch.nn as nn
from fancy_einsum import einsum
import einops
import utils as utils
from dataclasses import dataclass

from src.neural_networks import MLP, LayerNorm, Dropout, Sequential

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

            sin_vec = t.sin(
                delta / 10000**(2 * t.arange(hidden_dim // 2) / hidden_dim))
            cos_vec = t.cos(
                delta / 10000**(2 * t.arange(hidden_dim // 2) / hidden_dim))

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
        return x + self.maximal_pe[:x.size(1), :]

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

def multihead_masked_attention(Q: t.Tensor, K: t.Tensor, V: t.Tensor,
                               num_heads: int):
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
    QKT = QKT / (headsize**0.5)
    # tri = t.triu(t.ones((seq_len, seq_len)), diagonal=1)*(-10 ** 4)
    # QKT_masked = (QKT + tri)
    attention_probs = t.softmax(QKT, dim=-1)

    attention_values_ = einsum('b nh s_q s_k, b nh s_k h -> b nh s_q h',
                               attention_probs, V_)
    # b hn s_q h -->e = n*h --> b s_q e
    attention_values = einops.rearrange(attention_values_,
                                        ' b hn s_q h ->  b s_q (hn h)')

    return attention_values

class MultiheadMaskedAttention(nn.Module):
    # Hydra-Attention
    # W_QKV: nn.Linear
    # W_O: nn.Linear

    def __init__(self, hidden_size: int, num_heads: int):
        super().__init__()

        assert (hidden_size % num_heads) == 0

        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.W_QKV = nn.Linear(hidden_size, 3 * hidden_size)
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

class DecoderOnlyTransformer(nn.Module):

    def __init__(self, config: TransformerConfig):
        super().__init__()
        
        self.config = config 
        self.token_embedding = nn.Embedding(config.vocab_size, embedding_dim= config.hidden_size)
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

        x = einsum('word emd, b seq emb -> b seq word', self.token_embedding.weight, x)

        return x
