import torch as t
import torch.nn as nn
from src.train import train
from src.datasets import RevSequenceDataset
from src.transformers import TransformerConfig, DecoderOnlyTransformer
from torch.utils.data import DataLoader
from multiprocessing import cpu_count



batch_size = 256
seq_len = 6
transformer_config = TransformerConfig(num_layers=2,
                                       num_heads=4,
                                       vocab_size=10,
                                       hidden_size=512,
                                       max_seq_len=seq_len,
                                       dropout=0.1,
                                       layer_norm_epsilon=1e-5)

model = DecoderOnlyTransformer(config=transformer_config)

from torchinfo import torchinfo

torchinfo.summary(model, input_data=t.tensor([[1, 2, 3, 4, 5, 6]]))

# get data
train_set_size = 10**4#5*10**6
test_set_size = 10**3
num_workers = 0#cpu_count()

train_dataset = RevSequenceDataset(seq_length = seq_len, data_set_size=train_set_size)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
test_dataset = RevSequenceDataset(seq_length = seq_len, data_set_size=test_set_size)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

# Test forward pass
data = next(iter(test_loader))
example_output = model(data[0])[:3]
print(data[1][:3])
print(t.argmax(example_output, dim = -1))

# run model
num_epochs = 10
device = "cpu" if t.backends.mps.is_available() else "cpu"
criterion = nn.CrossEntropyLoss()
optimizer = t.optim.Adam(model.parameters(), lr=0.00125)
model = train(model,
              optimizer,
              train_loader,
              test_loader,
              criterion,
              num_epochs=num_epochs,
              device=device)
