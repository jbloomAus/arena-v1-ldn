import torch as t
import numpy as np 

from torch.utils.data import Dataset


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

class CustomTextDataset(Dataset):
    def __init__(self, texts, labels):
        self.labels = labels
        self.texts = texts

    @staticmethod
    def from_config(config, samples):
        texts = [t.randint(high=config.vocab_size, size=(config.max_seq_len,)) for _ in range(samples)]
        labels = [t.flip(text, (0,)) for text in texts]
        return CustomTextDataset(texts, labels)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        label = self.labels[idx]
        text = self.texts[idx]
        sample = (text, label)
        return sample