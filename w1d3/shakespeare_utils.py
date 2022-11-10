# %% 
import torch as t 
import re 
from torch.utils.data import Dataset
from typing import Optional, Union

class WordDataset(Dataset):

    def __init__(self, data, block_size, overwrite_length: int = None):
        self.overwrite_length = overwrite_length

        # get the words and the vocab
        words = re.split(r"\b", data)
        vocab = sorted(list(set(words)))

        # get word and vocab size
        data_size, vocab_size = len(words), len(vocab)

        # get i to s and s to i dictionary
        self.stoi = {s:i for i,s in enumerate(vocab)}
        self.itos = {i:s for i,s in enumerate(vocab)}

        # store block, vocab size and data
        self.block_size = block_size
        self.vocab_size = vocab_size
        self.data = words
        self.size = self.determine_size()

    def __len__(self):
        return self.size

    def __getitem__(self, idx):

        start = idx
        end = idx + self.block_size + 1

        chunk = self.data[start:end]
        chunk_in_tokens = [self.stoi[s] for s in chunk]

        x = t.tensor(chunk_in_tokens[:-1], dtype = t.long)
        y = t.tensor(chunk_in_tokens[1:], dtype = t.long)
        return x, y 

    def determine_size(self):
        default_size = len(self.data) - self.block_size
        print(f"Default text size {default_size}")

        if (self.overwrite_length is not None) and (self.overwrite_length < default_size):
            print("Using {:.2f}% of the data".format(100*self.overwrite_length /default_size))
            return self.overwrite_length
        else:
            print(f"Using 100% of the data")
            return default_size


class WordsTokenizer():
    model_max_length: int

    def __init__(self, wordsdataset: WordDataset):
        self.stoi = wordsdataset.stoi
        self.itos = wordsdataset.itos
        self.model_max_length = 48 # arbitrary

    def encode(self, initial_text: str, return_tensors: Optional[str] = None) -> Union[list, t.Tensor]:
        '''
        Tokenizes initial_text, then returns the token ids.

        Return type is list by default, but if return_tensors="pt" then it is returned as a tensor.
        '''
        words = re.split(r"\b", initial_text)
        tokens =  [self.stoi[s] for s in words if s != '']
        if return_tensors:
            return t.tensor(tokens)
        else:
            return tokens

    def decode(self, list_of_ids: Union[t.Tensor, list]) -> str:
        '''
        Converts ids to a list of tokens, then joins them into a single string.
        '''
        words =  [self.itos[i] if type(i) is int else self.itos[i.item()] for i in list_of_ids]
        return  "".join(words)

# shakespeare_text  = open('w1d3/shakespeare.txt', 'r').read()
# dataset = WordDataset(shakespeare_text, block_size=48, overwrite_length= 30000)
# len(dataset)
# word_tokenizer = WordsTokenizer(dataset)

# input_text = "The lady doth protest too much."
# ids =word_tokenizer.encode(input_text)
# print(ids)
# print(word_tokenizer.decode(ids))

# # %% 
# print(dataset.itos[113])
# print(dataset.stoi[' '])
# # %%
# word_tokenizer.decode(dataset[2][0])
# word_tokenizer.decode(dataset[2][1])
# # %%
# dataset[2][0].shape
# # %%
# t.arange(48).shape
# # %%