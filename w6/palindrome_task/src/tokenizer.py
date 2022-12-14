import torch 
from tokenizers import models, Tokenizer, processors
from transformers import PreTrainedTokenizerFast
from typing import List, Union, Optional


class SimpleTokenizer:
    '''
    tokenizer = SimpleTokenizer(string.ascii_lowercase)
    tokens = tokenizer.tokenize(["racecar", "dkfgsdkngkseng"])
    print(tokens.shape)
    
    '''
    START_TOKEN = 0
    PAD_TOKEN = 1
    END_TOKEN = 2
    base_d = {"[start]": START_TOKEN, "[pad]": PAD_TOKEN, "[end]": END_TOKEN}
    # lower_case_map = {c:i+3 for i, c in enumerate(string.ascii_lowercase)}
    # all_tokens_map

    def __init__(self, alphabet: str):
        self.alphabet = alphabet
        # the 3 is because there are 3 special tokens (defined just above)
        self.t_to_i = {**{c: i + 3 for i, c in enumerate(alphabet)}, **self.base_d}
        self.i_to_t = {i: c for c, i in self.t_to_i.items()}

    def tokenize(self, strs: list[str], 
                max_len: Optional[int] = None) -> torch.Tensor:
        def c_to_int(c: str) -> int:
            if c in self.t_to_i:
                return self.t_to_i[c]
            else:
                raise ValueError(c)

        if max_len is None:
            max_len = max((max(len(s) for s in strs), 1))

        ints = [
            [self.START_TOKEN] + [c_to_int(c) for c in s] + [self.END_TOKEN] + [self.PAD_TOKEN] * (max_len - len(s))
            for s in strs
        ]
        return torch.tensor(ints, dtype=torch.long)

    def decode(self, tokens) -> list[str]:
        def int_to_c(c: int) -> str:
            if c < len(self.i_to_t):
                return self.i_to_t[c]
            else:
                raise ValueError(c)

        return [
            "".join(int_to_c(i.item()) for i in seq[1:] if i != self.PAD_TOKEN and i != self.END_TOKEN)
            for seq in tokens
        ]


# provided by Callum McDougal
class ToyTokenizer(PreTrainedTokenizerFast):

    def __init__(
        self,
        vocab: Union[str, List[str], int, List[int]],
        pad: bool = True,
        cls: bool = True,
        sep: bool = True,
        unk: bool = False,
        mask: bool = False,
    ):
        """Creates a tokenizer from a list of characters, and a list of special tokens.

        Args:
            vocab_list (Optional[str, List[str]]):
                if str, then it is a string of characters to use as the vocabulary.
                if List[str], then it is a list of characters to use as the vocabulary.
                if List[int], then it is a list of integers to use as the vocabulary.
                if int, then it is the size of the vocabulary to generate (vocab assumed to be integers 0, 1, ..., vocab-1).

            pad, cls, sep, unk, mask (bool):
                whether to include these special tokens in the vocabulary.
                recall that [CLS] and [SEP] play the role of start and end tokens, respectively.
        """

        self.uses_pad = pad
        self.uses_cls = cls
        self.uses_sep = sep
        self.uses_unk = unk
        self.uses_mask = mask

        if isinstance(vocab, str):
            vocab = list(vocab)
        elif isinstance(vocab, int):
            vocab = list(range(vocab))
        self.originally_int = isinstance(vocab[0], int)
        if self.originally_int:
            vocab = map(str, vocab)

        special_tokens = {}
        for token_name in ["pad", "cls", "sep", "unk", "mask"]:
            if getattr(self, f"uses_{token_name}"):
                special_tokens[f"{token_name}_token"] = f"[{token_name.upper()}]"

        # trainer = trainers.WordPieceTrainer(vocab_size=100, special_tokens=list(special_tokens.values()))
        tokenizer_raw = Tokenizer(models.Unigram([(char, 0) for char in vocab]))
        tokenizer_raw.add_special_tokens(list(special_tokens.values()))
        tokenizer_raw.post_processor = processors.TemplateProcessing(
            single="[CLS] $A [SEP]",
            pair="[CLS] $A [SEP] $B:1 [SEP]:1",
            special_tokens=[(k, v) for k, v in tokenizer_raw.get_vocab().items() if k in ["[CLS]", "[SEP]"]],
        )

        self.tokenizer = tokenizer_raw
        super().__init__(tokenizer_object=tokenizer_raw, **special_tokens)
