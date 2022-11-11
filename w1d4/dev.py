# %% [markdown]

#  ## Reproducing GPT Architecture so we can load the weights.

# First, we will start by going through the differences between GPT and your implementation (i.e. the diagram from W1D3).

# * The order of the LayerNorms in the decoder block have changed: they now come before the attention and MLP blocks, rather than after.
# * The attention block has two dropout layers: one immediately after the softmax (i.e. before multiplying by V), and one immediately after multiplying with W_O at the very end of the attention block. Note that the dropout layers won't actually affect weight-loading or performance in eval mode (and you should still be able to train your model without them), but all the same it's nice to be able to exactly match GPT's architecture!
# * All your linear layers should have biases - even though some of them are called projections (which would seem to suggest not having a bias), this is often how they are implemented.
# * GPT-2 uses a learned positional embedding (i.e. nn.Embedding) rather than a sinusoidal one.

# %%
#from shakespeare_utils import WordsTokenizer, WordDataset
import torch as t
from src.transformers import TransformerConfig, DecoderOnlyTransformer
from torch.utils.data import DataLoader
# Get my model

from src.transformers import DecoderOnlyTransformer, TransformerConfig

config = {
    'batch_size': 16,
    'hidden_size': 768,
    'lr': 0.00125,
    'seq_len': 1024,
    'num_layers': 12,
    'num_heads': 12,
    'vocab_size': 50257,
    'num_epochs': 1,
    'device': 'cpu',
    'dropout': 0.1,
    'layer_norm_epsilon': 1e-5,
    'train_set_size': 4 * 10**4,
    'test_set_size': 1000,
    'num_workers': 2,
}

batch_size = config["batch_size"]
seq_len = config["seq_len"]
transformer_config = TransformerConfig(
    num_layers=config["num_layers"],
    num_heads=config["num_heads"],
    vocab_size=config["vocab_size"],
    hidden_size=config["hidden_size"],
    max_seq_len=config["seq_len"],
    dropout=config["dropout"],
    layer_norm_epsilon=config["layer_norm_epsilon"])

my_gpt = DecoderOnlyTransformer(config=transformer_config).train()

from torchinfo import summary

summary(my_gpt, input_data=t.arange(config["seq_len"]).unsqueeze(0))

# %%
import transformers
import utils
import torch as t

#my_gpt = GPT(config).train()
gpt = transformers.AutoModelForCausalLM.from_pretrained("gpt2").train()

utils.print_param_count(my_gpt, gpt, use_state_dict=False)
# %%
import torch.nn as nn


def copy_weights_from_gpt(my_gpt: nn.Module, gpt) -> nn.Module:
    '''
    Copy over the weights from gpt to your implementation of gpt.

    gpt should be imported using: 
        gpt = transformers.AutoModelForCausalLM.from_pretrained("gpt2")

    Returns your gpt model, with weights loaded in.

    You might find the function `copy_weights` from w0d3 helpful as a template.
    '''

    # FILL IN CODE: define a state dict from my_gpt.named_parameters() and gpt.named_parameters()
    my_parameters = dict(my_gpt.named_parameters())
    gpt_parameters = dict(gpt.named_parameters())

    print(f'Number of parameters in original gpt: {len(gpt_parameters)}')
    print(f'Number of parameters in       my gpt: {len(my_parameters)}')

    state_dict = dict()
    my_par_keys = my_parameters.keys()
    for k, v in gpt_parameters.items():
        new_k = k.replace("transformer.", "")
        present = new_k in my_par_keys
        #print(k, new_k, present)
        if not present:
            print("Parameters don't line up")
            print(f"{k} has no match in my_gpt")
            return False

        # check shape match
        gpt_shape = v.shape
        gpt_shape_transpose = v.T.shape if len(v.shape) > 1 else v.shape
        my_shape = my_parameters[new_k].shape
        if (gpt_shape_transpose != my_shape):
            if gpt_shape == my_shape:
                #print("Shapes different, however transposition will fix it")
                state_dict[new_k] = v
            else:
                # print("Parameter shapes don't line up")
                # print(f"{k} has shape: {gpt_shape}")
                # print(f"{new_k} has shape: {my_shape}")
                return False
        else:
            #print("Shapes match, all good")
            state_dict[new_k] = v.T

    my_gpt.load_state_dict(state_dict)
    return my_gpt


gpt = transformers.AutoModelForCausalLM.from_pretrained("gpt2")
my_gpt = copy_weights_from_gpt(my_gpt, gpt)

# %%
from sampling_methods import sample_tokens

tokenizer = transformers.AutoTokenizer.from_pretrained("gpt2")
initial_text = "turn down for what"
text_output = sample_tokens(gpt,
                            tokenizer,
                            initial_text,
                            max_tokens_generated=100,
                            temperature=1.0,
                            top_p=0.3,
                            freq_penalty=2)

print(text_output)

# %%
tokenizer = transformers.AutoTokenizer.from_pretrained("gpt2")
initial_text = "turn down for what"
text_output = sample_tokens(my_gpt,
                            tokenizer,
                            initial_text,
                            max_tokens_generated=100,
                            temperature=1.0,
                            top_p=0.3,
                            freq_penalty=2)

print(text_output)

# %% [markdown]

## Notes

# When this model works well, it is evidence that all modules used are working

# We have controlled for initialisation, training and the data itself.

# I will sequentially add back by self-written modules and see where the errors occur.
# When the output starts looking crap, I will assume that module has issues.

# Note that the positional encoding module isn't used in GPT so that module isn't validated by this method.
# I've also got the module producing good ouput with the rest of the structure so DecoderBlock and DecoderOnlyModels are good too.

# I'm predicting ahead of time that the issue is in the linear layer.

# List of modules :

# - Transformers: LayerNorm (works fine)
# - Transformers & MLP: Dropout (works fine)
# - Transformers: Sequential (works fine)
# - Transformers: Embedding ! Failed ! -> My embedding weights weren't updateable. This explains some degree of poor performance. -> Fixed
# - MLP: Linear (works fine)
# - MLP (rest, dropout, GELU)

# %%
import plotly.express as px

positional_embeddings = dict(
    gpt.named_parameters())["transformer.wpe.weight"].detach().numpy().T
fig = px.imshow(positional_embeddings,
                color_continuous_scale=[[0, 'blue'], [0.5, 'white'],
                                        [1.0, 'red']])
fig.write_image("positional_embeddings_gpt2.png")
# %%
# import numpy as np
# fig = px.imshow(np.log(max(positional_embeddings,10**-6)), color_continuous_scale=[[0, 'blue'], [0.5, 'white'], [1.0, 'red']])
# fig.write_image("log_positional_embeddings_gpt2.png")

# %%

# from torch.nn.functional import cosine_similarity
# max_seq_length = 1024
# pe = t.Tensor(positional_embeddings).T
# vec = t.Tensor([cosine_similarity(pe[i], pe[j], dim = 0) for i in range(max_seq_length) for j in range(max_seq_length)])
# fig = px.imshow(vec.reshape(1024,1024))
# fig.write_image("positional_embeddings_gpt2_dot_product.png")
# # %%
# from src.transformers import PositionalEncoding
# from torch.nn.functional import cosine_similarity
# import torch as t
# import plotly.express as px
# max_seq_length = 1024
# fixed_pe = PositionalEncoding(max_seq_length, 768).get_maximal_pe()
# vec = t.Tensor([cosine_similarity(fixed_pe[i], fixed_pe[j], dim = 0) for i in range(max_seq_length) for j in range(max_seq_length)])
# fig = px.imshow(vec.reshape(1024,1024))
# fig.write_image("positional_embeddings_fixed_dot_product.png")
# # %%

import transformers
import torch as t


def beam_search(model,
                input_ids: t.Tensor,
                num_return_sequences: int,
                num_beams: int,
                max_new_tokens: int,
                tokenizer,
                verbose=False) -> list[tuple[float, t.Tensor]]:
    '''
    input_ids: (seq, ) - the prompt
    max_new_tokens: stop after this many new tokens are generated, even if no EOS is generated. In this case, the best incomplete sequences should also be returned.
    verbose: if True, print the current (unfinished) completions after each iteration for debugging purposes

    Return list of length num_return_sequences. Each element is a tuple of (logprob, tokens) where the tokens include both prompt and completion, sorted by descending logprob.
    '''
    assert num_return_sequences <= num_beams

    model.eval()

    def get_continuations(input_ids, upstream_log_prob=0):
        '''
        Sample from model, either return top k or everything
        as a list of (tokens, sum log prob of path) tuples. 
        '''
        # get model outputs
        output = model(input_ids if isinstance(input_ids, t.Tensor) else t.
                       LongTensor(input_ids))
        all_logits = output if isinstance(output, t.Tensor) else output.logits

        # store output with each path prob
        log_probs = t.log_softmax(all_logits[-1], dim=0)
        next_search = list(
            zip(log_probs.detach().numpy() + upstream_log_prob,
                [list(input_ids) + [i] for i in t.arange(len(log_probs))]))
        return next_search

    beams = [(0, input_ids)]
    for i in range(max_new_tokens):
        if verbose:
            print(f"Iteration {i}")
            print([(i[0], tokenizer.decode(i[1])) for i in beams])
            print("="*89)
        new_beams = []
        [new_beams.extend(get_continuations(i[1], i[0])) for i in beams]
        sorted_beams = sorted(new_beams, key=lambda x: x[0], reverse=True)
        beams = sorted_beams[:num_beams]

    return beams[:num_return_sequences]


device = "cpu"
#tokenizer = transformers.AutoTokenizer.from_pretrained("gpt2")
#gpt = transformers.AutoModelForCausalLM.from_pretrained("gpt2").to(device).train()

your_prompt = "I'm the prince of Machine Learning. My catchphrase is "
input_ids = tokenizer(your_prompt,
                      return_tensors="pt",
                      return_attention_mask=False)["input_ids"][0]

num_return_sequences = 3
num_beams = 6
max_new_tokens = 10

final_logitsums_and_completions = beam_search(gpt,
                                              input_ids,
                                              num_return_sequences,
                                              num_beams,
                                              max_new_tokens,
                                              tokenizer,
                                              verbose=True)
# %%
