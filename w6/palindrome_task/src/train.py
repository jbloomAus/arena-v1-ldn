import torch
import wandb
from tqdm import tqdm


from src.model import Classifier
from src.tokenizer import ToyTokenizer
from src.utils import save_classifier, save_checkpoint

from torch.optim import Optimizer
from torch.utils.data import DataLoader


def train(classifier: Classifier,
           train_loader: DataLoader,
           test_loader: DataLoader,
           optimizer: Optimizer,
           loss_fn,
           tokenizer: ToyTokenizer,
           device: str = 'cpu',
           test_interval: int = 1000,
           save_interval: int = 1000,
           save_path: str = "models/",
           use_wandb: bool = False):

    if use_wandb:
        wandb.watch(classifier, log="all")

    classifier.train()
    pbar = tqdm(enumerate(train_loader), total=len(train_loader))

    for i, (x, y) in pbar:
        x_tokens = tokenizer(x)
        x_tokens = torch.tensor(x_tokens["input_ids"])
        y = y.to(device).long()

        logits = classifier.forward(x_tokens)
        loss = loss_fn(logits, y)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if i % 10 == 0:
            pbar.set_description(f"Loss: {loss.item():.3f}")
        
        if use_wandb:
            wandb.log({"loss": loss.item()})

        if i % test_interval == 0:
            accuracy = test(classifier, test_loader, tokenizer, device)
            if use_wandb:
                wandb.log({"accuracy": accuracy})

        if i % save_interval == 0:
            save_checkpoint(
                model_path=save_path, 
                classifier=classifier, 
                tokenizer=tokenizer, 
                cfg = classifier.transformer.cfg,
                num_examples=i)

    return classifier


def test(classifier,
          test_loader: DataLoader,
          tokenizer: ToyTokenizer,
          device: str = 'cpu'):

    classifier.eval()
    correct = 0
    total = 0
    for x,y in test_loader:
        x_tokens = tokenizer(x)
        x_tokens = torch.tensor(x_tokens["input_ids"])
        y = y.to(device).long()
        logits = classifier.forward(x_tokens)
        preds = torch.argmax(logits, dim=1)
        correct += (preds == y).sum().item()
        total += len(y)


    return correct/total