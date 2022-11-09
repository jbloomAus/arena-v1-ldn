import torch as t
import torch.nn as nn
from src.train import train
from src.datasets import RevSequenceDataset
from src.transformers import TransformerConfig, DecoderOnlyTransformer
from torch.utils.data import DataLoader
from multiprocessing import cpu_count
import wandb


config =  {
    'batch_size': 256,
    'hidden_size': 64,
    'lr': 0.00125,
    'seq_len': 6,
    'num_layers': 2,
    'num_heads': 4,
    'vocab_size': 10,
    'num_epochs': 10,
    'device': 'cpu',
    'dropout': 0.1,
    'layer_norm_epsilon': 1e-5,
    'train_set_size': 10**5,
    'test_set_size': 10**3
}

wandb.init(
    project="W1D1 Transformer Tilman and Joseph",
    name = "second try",
    entity="arena-ldn",
    config = config
    )





batch_size = 1024
seq_len = 6
transformer_config = TransformerConfig(num_layers=wandb.config.num_layers,
                                       num_heads=wandb.config.num_heads,
                                       vocab_size=wandb.config.vocab_size,
                                       hidden_size=wandb.config.hidden_size,
                                       max_seq_len=wandb.config.seq_len,
                                       dropout=wandb.config.dropout,
                                       layer_norm_epsilon=wandb.config.layer_norm_epsilon)

model = DecoderOnlyTransformer(config=transformer_config)

from torchinfo import torchinfo

torchinfo.summary(model, input_data=t.tensor([[1, 2, 3, 4, 5, 6]]))

# get data
train_set_size = wandb.config.train_set_size
test_set_size = wandb.config.test_set_size
num_workers = 0#cpu_count()

train_dataset = RevSequenceDataset(seq_length = seq_len, data_set_size=train_set_size)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
test_dataset = RevSequenceDataset(seq_length = seq_len, data_set_size=test_set_size)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

# run model
num_epochs = wandb.config.num_epochs
device = "cpu" if t.backends.mps.is_available() else "cpu"
criterion = nn.CrossEntropyLoss()
optimizer = t.optim.Adam(model.parameters(), lr=wandb.config.lr)
model = train(model,
              optimizer,
              train_loader,
              test_loader,
              criterion,
              num_epochs=num_epochs,
              device=device)
