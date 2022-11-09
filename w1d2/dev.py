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
from torchinfo import torchinfo
torchinfo.summary(model, input_data=t.tensor([[1,2,3,4,5]]))
# %%





# %%
import wandb
import os
from einops import rearrange
from tqdm import tqdm
device = t.device('cpu')
os.environ['WANDB_NOTEBOOK_NAME'] = 'my_solutions.py'
def train():

    # wandb_config_dict = {
    #     'batch_size': 256,
    #     'hidden_size': 64,
    #     'lr': 0.00125
    # }
    
    # wandb.init(project='w1d1_transformer', config=wandb_config_dict)

    config = TransformerConfig(
        num_layers=2, #N=6
        num_heads=4, #h=8
        vocab_size=10,
        hidden_size=64,#wandb.config.hidden_size, #d_model = 64 x 8 = 512
        max_seq_len=6,
        dropout=0.0 #p=0.1
    )

    epochs = 10
    batch_size = 1024
    lr = 10e-4 #wandb.config.lr
    train_samples = 50000
    test_samples = 1000

    model = DecoderOnlyTransformer(config).to(device).train()
    optimizer = t.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()

    examples_seen = 0
    start_time = time.time()

    trainset = CustomTextDataset.from_config(config, train_samples)
    testset = CustomTextDataset.from_config(config, test_samples)

    trainloader = DataLoader(trainset, shuffle=True, batch_size=batch_size)
    testloader = DataLoader(testset, shuffle=True, batch_size=batch_size)
    #wandb.watch(model, criterion=loss_fn, log="all", log_freq=10, log_graph=True)

    for epoch in range(epochs):
        progress_bar = tqdm(trainloader)

        for (x, y) in progress_bar:
            x = x.to(device)
            y = y.to(device)
            optimizer.zero_grad()
            y_hat = model(x)
            loss = loss_fn(rearrange(y_hat, "batch seq vocab_size -> (batch seq) vocab_size"), rearrange(y, "batch seq -> (batch seq)"))
            loss.backward()
            optimizer.step()
            progress_bar.set_description(f"Epoch = {epoch}, Loss = {loss.item():.4f}")
            examples_seen += len(x)
            #wandb.log({"train_loss": loss, "elapsed": time.time() - start_time}, step=examples_seen)

        with t.inference_mode():
            accuracy = 0
            total = 0
            for (x, y) in testloader:
                x = x.to(device)
                y = y.to(device)
                y_hat = model(x)
                y_flat = rearrange(y, "batch seq -> (batch seq)")
                y_pred_flat = rearrange(y_hat, "batch seq vocab_size -> (batch seq) vocab_size")
                y_predictions = y_pred_flat.argmax(-1)
                accuracy += (y_predictions == y_flat).sum().item()
                total += y_flat.size(0)

            #wandb.log({"test_accuracy": accuracy/total}, step=examples_seen)

        print(f"Epoch {epoch+1}/{epochs}, train loss is {loss:.6f}, accuracy is {accuracy}/{total}")

    #filename = f"{wandb.run.dir}/model_state_dict.pt"
    # print(f"Saving model to: {filename}")
    # t.save(model.state_dict(), filename)
    #wandb.save(filename)
    return model

model = train()

# %% 
data = next(iter(test_loader))
example_output = model(data[0])[:3]
print(data[1][:3])
print(t.argmax(example_output, dim = -1))
# %%