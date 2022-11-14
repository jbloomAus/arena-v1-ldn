# %% [markdown]
# # Building and Fine Tuning BERT
# Today my goals are to build and fine-tune BERT. 

# %% 
from typing import Optional, List
import einops
import transformers
import numpy as np
import pandas as pd
import torch as t
import torch.nn as nn
import utils
from fancy_einsum import einsum
from IPython.display import display
from src.activations import GELU
from src.neural_networks import Dropout, Embedding, LayerNorm
from src.transformers import Linear, TransformerConfig

config = TransformerConfig(
    hidden_size=768, 
    num_heads=12,
    num_layers=12,
    layer_norm_epsilon=1e-12,
    max_seq_len=512,
    dropout=0.1,
    vocab_size=28996
)

# %%

def make_additive_attention_mask(one_zero_attention_mask: t.Tensor, big_negative_number: float = -10000) -> t.Tensor:
    '''
    one_zero_attention_mask: 
        shape (batch, seq)
        Contains 1 if this is a valid token and 0 if it is a padding token.

    big_negative_number:
        Any negative number large enough in magnitude that exp(big_negative_number) is 0.0 for the floating point precision used.

    Out: 
        shape (batch, nheads=1, seqQ=1, seqK)
        Contains 0 if attention is allowed, big_negative_number if not.
    '''
    # let's assume the padding token is 0. 
    out = t.where(one_zero_attention_mask == 1, 0, big_negative_number).unsqueeze(1).unsqueeze(1) # shape batch, seq
    return out

utils.test_make_additive_attention_mask(make_additive_attention_mask)

class MultiheadAttention(nn.Module):
    # Hydra-Attention
    # W_QKV: nn.Linear
    # W_O: nn.Linear
    
    def __init__(self, hidden_size: int, num_heads: int, layer_norm_epsilon: float):
        super().__init__()

        assert (hidden_size % num_heads) == 0

        self.hidden_size = hidden_size
        self.num_heads = num_heads

        self.query = Linear(hidden_size, hidden_size)
        self.key = Linear(hidden_size, hidden_size)
        self.value = Linear(hidden_size, hidden_size)

        self.output = nn.ModuleDict({
            "dense":Linear(hidden_size, hidden_size),
            "LayerNorm":LayerNorm(hidden_size, eps = layer_norm_epsilon)
            })


    def forward(self, x: t.Tensor, additive_attention_mask: Optional[t.Tensor]) -> t.Tensor:
        '''
        x: shape (batch, seq, hidden_size)
        Return: shape (batch, seq, hidden_size)
        '''
        # QKV = einsum('b s hs, hs h2 -> b (s h2) hs',x, self.W_QKV) # h2 = 3*emb_len
        input = x 
        Q = self.query(x)
        K = self.key(x)
        V = self.value(x)
        av = multihead_masked_attention(Q, K, V, additive_attention_mask, self.num_heads)
        # av shape: b s_q emb
        # W_O shape: emb, embs
        out = self.output["dense"](x)
        out = self.output["LayerNorm"](x + input)
        return out

def multihead_masked_attention(Q: t.Tensor, K: t.Tensor, V: t.Tensor, mask: t.Tensor, num_heads: int):
    '''
    Implements multihead masked attention on the matrices Q, K and V.
    Q: shape (batch, seq, nheads*headsize)
    K: shape (batch, seq, nheads*headsize)
    V: shape (batch, seq, nheads*headsize)
    mask: shape (batch, 1, 1, seq)
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
    if mask is not None:
        QKT = (QKT + mask)
    attention_probs = t.softmax(QKT, dim=-1)
    attention_values_ = einsum('b nh s_q s_k, b nh s_k h -> b nh s_q h',
                               attention_probs, V_)
    # b hn s_q h -->e = n*h --> b s_q e
    attention_values = einops.rearrange(attention_values_,
                                        ' b hn s_q h ->  b s_q (hn h)')

    return attention_values

class BERTBlock(nn.Module):

    def __init__(self, hidden_size, num_heads, dropout, layer_norm_epsilon):
        super().__init__()
        self.attention = MultiheadAttention(
            hidden_size = hidden_size, 
            num_heads = num_heads,
            layer_norm_epsilon=layer_norm_epsilon
            )
        self.intermediate = nn.ModuleDict({
            "dense": Linear(hidden_size,4*hidden_size)
        })
        self.GELU = GELU()
        self.output = nn.ModuleDict({
            "dense": Linear(4*hidden_size, hidden_size),
            "LayerNorm": LayerNorm(hidden_size, layer_norm_epsilon)
        })
        self.dropout = Dropout(p = dropout)
            
    def forward(self, x: t.Tensor, additive_attention_mask: Optional[t.Tensor] = None) -> t.Tensor:
        '''
        x: shape (batch, seq, hidden_size)
        additive_attention_mask: shape (batch, nheads=1, seqQ=1, seqK)
        '''
        x = self.attention(x, additive_attention_mask)
        x_res = x 
        x = self.GELU(self.intermediate["dense"](x))
        x = self.dropout(self.output["dense"](x))
        x = self.output["LayerNorm"](x + x_res)
        return x

class BERTCommon(nn.Module):

    def __init__(self, config: TransformerConfig):
        super().__init__()
        
        self.config = config 
        self.embeddings = nn.ModuleDict({
            "word_embeddings": Embedding(config.vocab_size, embedding_dim= config.hidden_size),
            "position_embeddings": Embedding(config.max_seq_len, embedding_dim= config.hidden_size),
            "token_type_embeddings": Embedding(2, embedding_dim= config.hidden_size),
            "LayerNorm": LayerNorm(config.hidden_size)
        })

        self.dropout = Dropout(p=config.dropout)
        
        self.encoder = nn.ModuleDict({"layer": nn.ModuleList([BERTBlock(
            config.hidden_size,
            config.num_heads,
            config.dropout,
            config.layer_norm_epsilon
        ) for _ in range(config.num_layers)])})
        
    def forward(
            self,
            x: t.Tensor,
            one_zero_attention_mask: Optional[t.Tensor] = None,
            token_type_ids: Optional[t.Tensor] = None,
        ) -> t.Tensor:
        """
        input_ids: (batch, seq) - the token ids
        token_type_ids: (batch, seq) - only used for next sentence prediction.
        one_zero_attention_mask: (batch, seq) - only used in training. See make_additive_attention_mask.
        """
        if one_zero_attention_mask is None:
            additive_attention_mask = None
        else:
            additive_attention_mask = make_additive_attention_mask(one_zero_attention_mask)
        
        if token_type_ids is None:
            token_type_ids = t.zeros_like(x, dtype=t.int64)

        pos = t.arange(x.shape[1], device=x.device)
        x = self.embeddings["word_embeddings"](x) + self.embeddings["position_embeddings"](pos)
        x = x + self.embeddings["token_type_embeddings"](token_type_ids)
        x = self.embeddings["LayerNorm"](x)
        x = self.dropout(x)

        for block in self.encoder["layer"]:
            x = block(x, additive_attention_mask)
  
        # # project to vocab size
        # x = einsum('word emb, b seq emb -> b seq word', self.wte.weight, x)

        return x

class BERTLanguageMODEL(nn.Module):

    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.bert = BERTCommon(config)
        self.cls = Linear(config.hidden_size, config.hidden_size)
        self.GELU = GELU()
        self.LayerNorm = LayerNorm(config.hidden_size, config.layer_norm_epsilon)
        self.prediction_bias = nn.Parameter(t.randn(config.vocab_size))

    def forward(
        self,
        x: t.Tensor,
        one_zero_attention_mask: Optional[t.Tensor] = None,
        token_type_ids: Optional[t.Tensor] = None,
    ) -> t.Tensor:
        """Compute logits for each token in the vocabulary.

        Return: shape (batch, seq, vocab_size)
        """
        x = self.bert(x, one_zero_attention_mask, token_type_ids)
        x = self.GELU(self.cls(x))
        x = self.LayerNorm(x)
        x = einsum('word emb, b seq emb -> b seq word', self.bert.embeddings["word_embeddings"].weight, x)
        x = x + self.prediction_bias
        return x

def print_param_count(*models, display_df=True, use_state_dict=False):
    """
    display_df: bool
        If true, displays styled dataframe
        if false, returns dataframe

    use_state_dict: bool
        If true, uses model.state_dict() to construct dataframe
            This will include buffers, not just params
        If false, uses model.named_parameters() to construct dataframe
            This misses out buffers (more useful for bert)
    """
    df_list = []
    gmap_list = []
    for i, model in enumerate(models, start=1):
        print(f"Model {i}, total params = {sum([param.numel() for name, param in model.named_parameters()])}")
        iterator = model.state_dict().items() if use_state_dict else model.named_parameters()
        df = pd.DataFrame([
            {f"name_{i}": name, f"shape_{i}": tuple(param.shape), f"num_params_{i}": param.numel()}
            for name, param in iterator
        ]) if (i == 1) else pd.DataFrame([
            {f"num_params_{i}": param.numel(), f"shape_{i}": tuple(param.shape), f"name_{i}": name}
            for name, param in iterator
        ])
        display(df)
        df_list.append(df)
        gmap_list.append(np.log(df[f"num_params_{i}"]))
    df = df_list[0] if len(df_list) == 1 else pd.concat(df_list, axis=1).fillna(0)
    for i in range(1, len(models) + 1):
        df[f"num_params_{i}"] = df[f"num_params_{i}"].astype(int)
    if len(models) > 1:
        param_counts = [df[f"num_params_{i}"].values.tolist() for i in range(1, len(models) + 1)]
        if all([param_counts[0] == param_counts[i] for i in range(1, len(param_counts))]):
            print("All parameter counts match!")
        else:
            print("Parameter counts don't match up exactly.")
    if display_df:
        s = df.style
        for i in range(1, len(models) + 1):
            s = s.background_gradient(cmap="viridis", subset=[f"num_params_{i}"], gmap=gmap_list[i-1])
        with pd.option_context("display.max_rows", 1000):
            display(s)
    else:
        return df

#print_param_count(bert, my_bert, use_state_dict=False)
# %%

def copy_weights_from_bert(my_bert: nn.Module, bert) -> nn.Module:
    '''
    Copy over the weights from bert to your implementation of bert.

    bert should be imported using: 
        bert = transformers.BertForMaskedLM.from_pretrained("bert-base-cased")

    Returns your bert model, with weights loaded in.
    '''

    my_parameters = dict(my_bert.named_parameters())
    bert_parameters = dict(bert.named_parameters())

    print(f'Number of parameters in original bert: {len(bert_parameters)}')
    print(f'Number of parameters in       my bert: {len(my_parameters)}')

    state_dict = dict()
    my_par_keys = my_parameters.keys()
    for k, v in bert_parameters.items():
        new_k = k.replace("self.", "").replace(".predictions.transform.dense","")
        new_k = new_k.replace("cls.predictions.bias","prediction_bias")
        new_k = new_k.replace("cls.predictions.transform.","")
        present = new_k in my_par_keys
        #print(k, new_k, present)
        if not present:
            print("Parameters don't line up")
            print(f"{k} has no match in my_bert")
            return False

        # check shape match
        bert_shape = v.shape
        my_shape = my_parameters[new_k].shape
        if bert_shape == my_shape:
            #print("Shapes different, however transposition will fix it")
            state_dict[new_k] = v
        else:
            print("Parameter shapes don't line up")
            print(f"{k} has shape: {bert_shape}")
            print(f"{new_k} has shape: {my_shape}")
            return False

    my_bert.load_state_dict(state_dict)
    return my_bert


tokenizer = transformers.AutoTokenizer.from_pretrained("bert-base-cased")
bert = transformers.BertForMaskedLM.from_pretrained("bert-base-cased")
my_bert = BERTLanguageMODEL(config)
my_bert = copy_weights_from_bert(my_bert, bert)


# %%
import plotly.express as px

def predict(model, tokenizer, text: str, k=15) -> List[List[str]]:
    '''
    Return a list of k strings for each [MASK] in the input.
    '''
    input_ids = tokenizer.encode(text=text, return_tensors="pt")

    print(input_ids)
    with t.inference_mode():
        model.eval()
        results = model(input_ids)
        logits = results if isinstance(results, t.Tensor) else results.logits
    mask_positons = (input_ids.squeeze() == 103).nonzero(as_tuple=True)[0]
    print(mask_positons)
    mask_token_predictions = []
    for position in mask_positons: #  t.arange(input_ids.size(1)):#
        position_completion = []
        position_logits = logits.squeeze()[position]
        output_ids = t.topk(position_logits, k= 15).indices
        for id in output_ids:
            position_completion.append(tokenizer.decode(id))
        mask_token_predictions.append(sorted(position_completion))
    
    return mask_token_predictions

def test_bert_prediction(predict, model, tokenizer):
    '''Your Bert should know some names of American presidents.'''
    text = "Former President of the United States of America, George[MASK][MASK]"
    predictions = predict(model, tokenizer, text)
    print(f"Prompt: {text}")
    print("Model predicted: \n", "\n".join(map(str, predictions)))
    assert "Washington" in predictions[0]
    assert "Bush" in predictions[0]

test_bert_prediction(predict, my_bert, tokenizer)
#test_bert_prediction(predict, bert, tokenizer)

# %%

# %%
# import transformers

# bert = transformers.BertForMaskedLM.from_pretrained("bert-base-cased")
# tokenizer = transformers.AutoTokenizer.from_pretrained("bert-base-cased")

# # The senetence to be encoded
# sent = "[MASK][MASK][MASK]"

# # Encode the sentence
# encoded_plus = tokenizer.encode_plus(
#     text=sent,  # the sentence to be encoded
#     add_special_tokens=True,  # Add [CLS] and [SEP]
#     max_length = 64,  # maximum length of a sentence
#     pad_to_max_length=True,  # Add [PAD]s
#     return_attention_mask = True,  # Generate the attention mask
#     return_tensors = 'pt',  # ask the function to return PyTorch tensors
# )
# encoded = tokenizer.encode(text=sent, return_tensors="pt")

# print(encoded)
# print(encoded_plus.input_ids)
# # %%
