import torch
import wandb
import tqdm.notebook as tqdm


from src.tokenizer import ToyTokenizer
from torch.optim import Optimizer
from torch.utils.data import DataLoader

def train(classifier,
           train_loader: DataLoader,
           test_loader: DataLoader,
           optimizer: Optimizer,
           loss_fn,
           tokenizer: ToyTokenizer,
           total_examples: int = 4 * 10**6,
           device: str = 'cpu',
           test_interval: int = 1000,
           wandb: wandb = None):

    if wandb is not None:
        wandb.watch(classifier)

    classifier.train()
    batch_size = train_loader.batch_size
    pbar = tqdm.tqdm(enumerate(train_loader),
                     total=total_examples // batch_size)

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
        
        if wandb is not None:
            wandb.log({"loss": loss.item()})

        if i % test_interval == 0:
            accuracy = test(classifier, test_loader, tokenizer, device)
            if wandb is not None:
                wandb.log({"accuracy": accuracy})

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