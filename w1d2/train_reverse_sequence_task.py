

import torch as t
import torch.nn as nn
from src.train import train
from src.datasets import RevSequenceDataset
from src.transformers import TransformerConfig, DecoderOnlyTransformer
from torch.utils.data import DataLoader


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

from torchinfo import torchinfo
torchinfo.summary(model, input_data=t.tensor([[1,2,3,4,5]]))


dataset = RevSequenceDataset()
train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
criterion = nn.CrossEntropyLoss()
optimizer = t.optim.Adam(model.parameters(), lr = 0.01)

model = train(model, optimizer,  train_loader, test_loader, criterion, num_epochs=10, device="cpu")
