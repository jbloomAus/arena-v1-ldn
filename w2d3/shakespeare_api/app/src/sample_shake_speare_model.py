# %% 
from .sampling_methods import sample_tokens
import torch as t
from src.transformers import DecoderOnlyTransformer, TransformerConfig
from src.shakespeare_utils import WordsTokenizer, WordDataset

config = {
    'batch_size': 16,
    'hidden_size': 512,
    'lr': 0.00125,
    'seq_len': 128,
    'num_layers': 8,
    'num_heads': 8,
    'vocab_size': 34543,
    'num_epochs': 1,
    'device': 'mps',
    'dropout': 0.1,
    'layer_norm_epsilon': 1e-4,
    'train_set_size': 10 * 10**4,
    'test_set_size': 10000,
    'num_workers': 0,
    }

transformer_config = TransformerConfig(
    num_layers=config['num_layers'],
    num_heads=config['num_heads'],
    vocab_size=config['vocab_size'],
    hidden_size=config['hidden_size'],
    max_seq_len=config['seq_len'],
    dropout=config['dropout'],
    layer_norm_epsilon=config['layer_norm_epsilon'])


shakespeare_text = open('/data/shakespeare.txt', 'r').read()
train_dataset = WordDataset(shakespeare_text,
                        block_size=config['seq_len'],
                        overwrite_length=config['train_set_size'])

tokenizer = WordsTokenizer(train_dataset)
my_gpt = DecoderOnlyTransformer(config=transformer_config)

PATH = "/models/language_model.pt"
my_gpt.load_state_dict(t.load(PATH))
my_gpt.eval()

# initial_text = "turn down for what"
# text_output = sample_tokens(my_gpt,
#                             tokenizer,
#                             initial_text,
#                             max_tokens_generated=100,
#                             temperature=1.0,
#                             top_p=0.3,
#                             freq_penalty=2)

# print(text_output)
# %%
