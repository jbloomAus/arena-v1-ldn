
# %%
import torch as t
from torch import nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from fancy_einsum import einsum
from typing import Union, Optional, Callable
import numpy as np
from einops import rearrange
from tqdm.notebook import tqdm_notebook
import plotly.express as px
import plotly.graph_objs as go
from plotly.subplots import make_subplots
import time
import wandb
import utils

device = "mps" if t.backends.mps.is_available() else "cpu"
print(device)

# %%

# %%
cifar_mean = [0.485, 0.456, 0.406]
cifar_std = [0.229, 0.224, 0.225]

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=cifar_mean, std=cifar_std)
])

trainset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
testset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

utils.show_cifar_images(trainset, rows=3, cols=5)


# %%

from d3 import ResNet34

def train(trainset, testset, epochs: int, loss_fn: Callable, batch_size: int, lr: float) -> tuple[list, list]:

    model = ResNet34().to(device).train()
    optimizer = t.optim.Adam(model.parameters(), lr=lr)

    loss_list = []
    accuracy_list = []

    trainloader = DataLoader(trainset, shuffle=True, batch_size=batch_size)
    testloader = DataLoader(testset, shuffle=True, batch_size=batch_size)

    for epoch in range(epochs):

        progress_bar = tqdm_notebook(trainloader)

        for (x, y) in progress_bar:

            x = x.to(device)
            y = y.to(device)

            optimizer.zero_grad()
            y_hat = model(x)
            loss = loss_fn(y_hat, y)
            loss.backward()
            optimizer.step()

            loss_list.append(loss.item())

            progress_bar.set_description(f"Epoch = {epoch}, Loss = {loss.item():.4f}")

        with t.inference_mode():

            accuracy = 0
            total = 0

            for (x, y) in testloader:

                x = x.to(device)
                y = y.to(device)

                y_hat = model(x)
                y_predictions = y_hat.argmax(1)
                accuracy += (y_predictions == y).sum().item()
                total += y.size(0)

            accuracy_list.append(accuracy/total)

        print(f"Epoch {epoch+1}/{epochs}, train loss is {loss:.6f}, accuracy is {accuracy}/{total}")

    filename = "./w0d3_resnet.pt"
    print(f"Saving model to: {filename}")
    t.save(model.state_dict(), filename)

    utils.plot_results(loss_list, accuracy_list)
    return loss_list, accuracy_list

# epochs = 1
# loss_fn = nn.CrossEntropyLoss()
# batch_size = 128
# lr = 0.001

# loss_list, accuracy_list = train(trainset, testset, epochs, loss_fn, batch_size, lr)

# %%
from torchvision.models.resnet import resnet34

def train(trainset, testset, epochs: int, loss_fn: Callable, batch_size: int, lr: float) -> None:

    config_dict = {
        "batch_size": batch_size,
        "epochs": epochs,
        "lr": lr,
    }
    wandb.init(project="w2d1_resnet", config=config_dict)

    model = resnet34(weights="DEFAULT").to(device).train()
    optimizer = t.optim.Adam(model.parameters(), lr=lr)

    examples_seen = 0
    start_time = time.time()



    trainloader = DataLoader(trainset, shuffle=True, batch_size=batch_size)
    testloader = DataLoader(testset, shuffle=True, batch_size=batch_size)

    wandb.watch(model, criterion=loss_fn, log="all", log_freq=10, log_graph=True)

    for epoch in range(epochs):

        progress_bar = tqdm_notebook(trainloader)

        model.train()
        for (x, y) in progress_bar:

            x = x.to(device)
            y = y.to(device)

            optimizer.zero_grad()
            y_hat = model(x)
            loss = loss_fn(y_hat, y)
            loss.backward()
            optimizer.step()

            progress_bar.set_description(f"Epoch = {epoch}, Loss = {loss.item():.4f}")

            examples_seen += len(x)
            wandb.log({"train_loss": loss, "elapsed": time.time() - start_time}, step=examples_seen)

        model.eval()
        with t.inference_mode():

            accuracy = 0
            total = 0

            for (x, y) in testloader:

                x = x.to(device)
                y = y.to(device)

                y_hat = model(x)
                y_predictions = y_hat.argmax(1)
                accuracy += (y_predictions == y).sum().item()
                total += y.size(0)

            wandb.log({"test_accuracy": accuracy/total}, step=examples_seen)

    filename = f"{wandb.run.dir}/model_state_dict.pt"
    print(f"Saving model to: {filename}")
    t.save(model.state_dict(), filename)
    wandb.save(filename)

# %%

epochs = 1
loss_fn = nn.CrossEntropyLoss()
batch_size = 128
lr = 0.001
# num_classes = 10
# use_imagenet_weight = True 

def train() -> None:

    wandb.init()

    epochs = wandb.config.epochs
    batch_size = wandb.config.batch_size
    lr = wandb.config.lr
    finetune_final_layer = wandb.config.finetune_final_layer
    use_imagenet_weight = wandb.config.use_imagenet_weight
    data_augmentation = wandb.config.data_augmentation
    num_classes = 10


    cifar_mean = [0.485, 0.456, 0.406]
    cifar_std = [0.229, 0.224, 0.225]

    if not data_augmentation:
        transform_train = transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomCrop(32, padding=4),
            transforms.ToTensor(),
            transforms.Normalize(mean=cifar_mean, std=cifar_std)
        ])
        transform_val = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=cifar_mean, std=cifar_std)
        ])
    else: 
        transform_val = transform_train = transforms.Compose([

            transforms.ToTensor(),
            transforms.Normalize(mean=cifar_mean, std=cifar_std)
        ])

    trainset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
    testset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_val)




    if use_imagenet_weight:
        model = resnet34(weights="DEFAULT").to(device).train()
    else: 
        model = resnet34().to(device).train()

    if finetune_final_layer:

        model.fc = nn.Linear(512, num_classes).to(device)
        for name, param in model.named_parameters():
            if name.startswith("fc"):
                param.requires_grad = True
            else:
                param.requires_grad = False

        print("Params to learn:")
        params_to_update = []
        for name, param in model.named_parameters():
            if param.requires_grad == True:
                params_to_update.append(param)
                print("\t", name)

        optimizer = t.optim.Adam(params_to_update, lr=lr)
    else:
        optimizer = t.optim.Adam(model.parameters(), lr=lr)

    examples_seen = 0
    start_time = time.time()

    trainloader = DataLoader(trainset, shuffle=True, batch_size=batch_size)
    testloader = DataLoader(testset, shuffle=True, batch_size=batch_size)

    wandb.watch(model, criterion=loss_fn, log="all", log_freq=10, log_graph=True)

    for epoch in range(epochs):

        progress_bar = tqdm_notebook(trainloader)

        model.train()
        for (x, y) in progress_bar:

            x = x.to(device)
            y = y.to(device)

            optimizer.zero_grad()
            y_hat = model(x)
            loss = loss_fn(y_hat, y)
            loss.backward()
            optimizer.step()

            progress_bar.set_description(f"Epoch = {epoch}, Loss = {loss.item():.4f}")

            examples_seen += len(x)
            wandb.log({"train_loss": loss, "elapsed": time.time() - start_time}, step=examples_seen)

        model.eval()
        with t.inference_mode():

            accuracy = 0
            total = 0

            for (x, y) in testloader:

                x = x.to(device)
                y = y.to(device)

                y_hat = model(x)
                y_predictions = y_hat.argmax(1)
                accuracy += (y_predictions == y).sum().item()
                total += y.size(0)

            wandb.log({"test_accuracy": accuracy/total}, step=examples_seen)

        print(f"Epoch {epoch+1}/{epochs}, train loss is {loss:.6f}, accuracy is {accuracy}/{total}")

    filename = f"{wandb.run.dir}/model_state_dict.pt"
    print(f"Saving model to: {filename}")
    t.save(model.state_dict(), filename)
    wandb.save(filename)

sweep_config = {
    'method': 'random',
    'name': 'w2d1_resnet_sweep_2',
    'metric': {'name': 'test_accuracy', 'goal': 'maximize'},
    'parameters': 
    {
        'batch_size': {'values': [64, 128]},
        'epochs': {'min': 5, 'max': 10},
        'lr': {'max': 0.0001, 'min': 0.00001, 'distribution': 'log_uniform_values'},
        'finetune_final_layer': {'values': [False]},
        'use_imagenet_weight': {'values': [True]},
        'data_augmentation': {'value':[True]}
     }
}

sweep_id = wandb.sweep(sweep=sweep_config, project='w2d1_resnet')

wandb.agent(sweep_id=sweep_id, function=train, count=10)
# %%
