# # %%
# %%
import torch as t
# import torch.nn as nn
# from fancy_einsum import einsum
# import einops
# import utils as utils
import plotly.express as px
from src.transformers import single_head_attention, single_head_masked_attention, multihead_masked_attention, MultiheadMaskedAttention

b = 64
seq = 12
emb = 8
Q, K, V = t.rand((b, seq, emb)), t.rand((b, seq, emb)), t.rand((b, seq, emb))

av = single_head_attention(Q, K, V)
assert av.shape == t.Size([64, 12, 8])

Q = t.tensor([[1, 1, 1, 1], [0, 0, 0, 0], [0, 0, 0, 0]],
             dtype=float).unsqueeze(dim=0)
K = t.tensor([[1, 1, 1, 0], [0, 1, 0, 0], [1, 0, 0, 0]],
             dtype=float).unsqueeze(dim=0)
V = t.tensor([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0]],
             dtype=float).unsqueeze(dim=0)

av = single_head_attention(Q, K, V)
# px.imshow(Q.squeeze())
# px.imshow(K.squeeze())

px.imshow(av.squeeze())
#assert av.shape == t.Size([64, 12, 8])

Q = t.tensor([[1, 1, 1, 1], [0, 0, 0, 0], [0, 0, 0, 0]],
             dtype=float).unsqueeze(dim=0)
K = t.tensor([[1, 1, 1, 0], [0, 1, 0, 0], [1, 0, 0, 0]],
             dtype=float).unsqueeze(dim=0)
V = t.tensor([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0]],
             dtype=float).unsqueeze(dim=0)

av = single_head_masked_attention(Q, K, V)
px.imshow(av.squeeze()).show()

from src.transformers import multihead_masked_attention
b = 64
n_h = 2
seq = 12
emb = 8
Q, K, V = t.rand((b, seq, n_h*emb)), t.rand((b, seq, n_h*emb)), t.rand((b, seq, n_h*emb))

av = multihead_masked_attention(Q, K, V, num_heads=n_h)
assert av.shape == t.Size([64, 12, 16])

b = 128
n_h = 4
seq = 16
emb = 48
hydra = MultiheadMaskedAttention(hidden_size=emb, num_heads=n_h)

x = t.rand((b, seq, emb))
assert hydra(x).shape == t.Size([128, 16, 48])

# %%

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
print(dataset[103][0].unsqueeze(0).shape)
model(dataset[103][0].unsqueeze(0)).shape

# %%
import time
import torch.nn as nn

device = t.device("mps" if t.backends.mps.is_available() else "cpu")

from src.train import train
from src.datasets import RevSequenceDataset
from torch.utils.data import DataLoader

dataset = RevSequenceDataset()
train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
criterion = nn.CrossEntropyLoss()
optimizer = t.optim.Adam(model.parameters(), lr = 0.01)

model = train(model, optimizer,  train_loader, test_loader, criterion, num_epochs=10, device="cpu")

# %%

data = next(iter(train_loader))
#example_output = model(next(iter(train_loader))[0])
data[0][0]


# %%
example_output = model(data)
# %%
t.argmax(example_output[0], dim = -1)
# %%
data[0][0]
# %%
data = next(iter(test_loader))
example_output = model(data[0])[:3]
print(data[1][:3])
print(t.argmax(example_output, dim = -1))
# %%


# %% 
px.imshow(example_output[0].detach().numpy()).show()
px.imshow(example_output[1].detach().numpy()).show()
px.imshow(example_output[2].detach().numpy()).show()

# %%
print(t.argmax(example_output[0], dim = -1))
print(t.argmax(example_output[1], dim = -1))
print(t.argmax(example_output[2], dim = -1))

# %%
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


# get model trained from script
model.load_state_dict(t.load("best_decoder_only_rev_sequence_model.pt"))

# Test forward pass
data = next(iter(test_loader))
example_output = model(data[0])[:3]
print(data[1][:3])
print(t.argmax(example_output, dim = -1))



# %%
