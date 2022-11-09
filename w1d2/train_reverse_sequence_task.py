import torch as t
import torch.nn as nn
from src.train import train
from src.datasets import RevSequenceDataset
from src.transformers import TransformerConfig, DecoderOnlyTransformer
from torch.utils.data import DataLoader

batch_size = 64
transformer_config = TransformerConfig(num_layers=2,
                                       num_heads=4,
                                       vocab_size=10,
                                       hidden_size=128,
                                       max_seq_len=4,
                                       dropout=0.1,
                                       layer_norm_epsilon=1e-5)

model = DecoderOnlyTransformer(config=transformer_config)

from torchinfo import torchinfo

torchinfo.summary(model, input_data=t.tensor([[1, 2, 3, 4]]))

# get data
dataset = RevSequenceDataset(seq_length = 4, data_set_size=10**5)
train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

# Test forward pass
data = next(iter(test_loader))
example_output = model(data[0])[:3]
print(data[1][:3])
print(t.argmax(example_output, dim = -1))

# run model
num_epochs = 2
device = "mps" if t.backends.mps.is_available() else "cpu"
criterion = nn.CrossEntropyLoss()
optimizer = t.optim.Adam(model.parameters(), lr=0.01)
model = train(model,
              optimizer,
              train_loader,
              test_loader,
              criterion,
              num_epochs=num_epochs,
              device=device)
