import json
import random
import torch as t
import torch.nn as nn
import transformers
from build_bert import BERTLanguageMODEL as BertLanguageModel
from einops import rearrange
from src.transformers import TransformerConfig
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
import wandb

def random_mask(
    input_ids: t.Tensor, mask_token_id: int, vocab_size: int, select_frac=0.15, mask_frac=0.8, random_frac=0.1
) -> tuple[t.Tensor, t.Tensor]:
    '''Given a batch of tokens, return a copy with tokens replaced according to Section 3.1 of the paper.

    input_ids: (batch, seq)

    Return: (model_input, was_selected) where:

    model_input: (batch, seq) - a new Tensor with the replacements made, suitable for passing to the BertLanguageModel. Don't modify the original tensor!

    was_selected: (batch, seq) - 1 if the token at this index will contribute to the MLM loss, 0 otherwise
    '''
    device = input_ids.device
    input_ids = input_ids.clone()
    seq_len= input_ids.shape[-1]
    input_ids = rearrange(input_ids, 'b s -> (b s)')
    n = len(input_ids)

    # choose which positions to affect
    masked_positions = random.sample(range(n), k = int(select_frac*n))
    mask_token_positions = masked_positions[:int(mask_frac*len(masked_positions))]
    random_token_positions = masked_positions[int(mask_frac*len(masked_positions)):int((mask_frac+random_frac)*len(masked_positions))]
    leave_token_positions = masked_positions[int((mask_frac+random_frac)*len(masked_positions)):]

    # mask each
    input_ids[mask_token_positions] = mask_token_id
    input_ids[random_token_positions] = t.tensor(random.sample(range(vocab_size), k = len(random_token_positions))).to(device).to(t.int)

    # get was selected 
    mask = t.zeros(n)
    mask[masked_positions] = 1

    # rearrange before returning
    input_ids = rearrange(input_ids, '(b s) -> b s', s = seq_len)
    was_selected = rearrange(mask, '(b s) -> b s', s = seq_len)

    return input_ids.to(t.long), was_selected.to(t.long)

def cross_entropy_selected(pred: t.Tensor, target: t.Tensor, was_selected: t.Tensor) -> t.Tensor:
    '''
    pred: (batch, seq, vocab_size) - predictions from the model
    target: (batch, seq, ) - the original (not masked) input ids
    was_selected: (batch, seq) - 1 if the token at this index will contribute to the MLM loss, 0 otherwise

    Out: the mean loss per predicted token
    '''
    device = pred.device
    target = target.to(device)
    was_selected = was_selected.to(device)
    target = t.where(was_selected.to(t.bool), target, -100).long()
    entropy = nn.functional.cross_entropy(
        rearrange(pred, "b s ... -> (b s) ..."), 
        rearrange(target, "b s ... -> (b s) ...")
    )
    return entropy

def lr_for_step(step: int, max_step: int, max_lr: float, warmup_step_frac: float):
    '''Return the learning rate for use at this step of training.'''
    delta =  max_step*warmup_step_frac
    if step < delta:
        return max_lr*(step/delta) # when step == delta, reach max_lr
    else: 
        return max_lr - max_lr*(step/(max_step-delta)) # when step = max_step-delta, lr = 0

def make_optimizer(model: BertLanguageModel, config_dict: dict) -> t.optim.AdamW:
    '''
    Loop over model parameters and form two parameter groups:

    - The first group includes the weights of each Linear layer and uses the weight decay in config_dict
    - The second has all other parameters and uses weight decay of 0
    '''



    weights = [v for k,v in model.named_parameters() if ("bias" not in k) and ("LayerNorm" not in k) and ("embeddings" not in k)]
    biases = [v for k,v in model.named_parameters() if ("bias" in k) or ("LayerNorm" in k) or ("embeddings" in k)]
    parameter_groups = [{"params": weights, **config_dict}, {"params": biases, **config_dict, "weight_decay":0}]
    
    return t.optim.AdamW(parameter_groups)

def bert_mlm_pretrain(model: BertLanguageModel, config_dict: dict, train_loader: DataLoader, device = "cpu") -> None:
    '''Train using masked language modelling.'''
    wandb.init(
        project="BERT - Scratch - Joseph", 
        config=config_dict, 
        entity="arena-ldn"
        )
    config = wandb.config
    wandb.watch(model, log="all")
    optimizer = make_optimizer(model, config_dict)
    max_step = int(len(train_loader) * config.epochs)
    step = 0
    tokens_processed = 0
    model = model.to(device)
    for epoch in range(config.epochs):
        pbar = tqdm(enumerate(train_loader), desc=f"Epoch {epoch}")
        for i, (batch,) in pbar:
            model.train()
            optimizer.zero_grad()
            batch = batch.to(device)
            (masked, was_selected) = random_mask(batch, config.mask_token_id, model.bert.config.vocab_size)
            pred = model(masked)
            loss = cross_entropy_selected(pred, batch, was_selected)
            loss.backward()
            if config.max_grad_norm is not None:
                t.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
            optimizer.step()
            lr = lr_for_step(step, max_step, config.lr, config.warmup_step_frac)
            step += 1
            for param_group in optimizer.param_groups:
                param_group["lr"] = lr
            tokens_processed += batch.shape[0] * batch.shape[1]
            wandb.log({"loss": loss.item(), "lr": lr, "tokens": tokens_processed})
            pbar.set_description(f'Epoch {epoch} - Loss: {loss.item():.4f} - LR: {lr:.4f}')
            
    # save model
    t.save(model.state_dict(), f"./models/bert_mlm_{config_dict['hidden_size']}.pt")

    # save config
    with open(f"./models/bert_mlm_{config_dict['hidden_size']}.json", "w") as f:
        json.dump(config_dict, f)

tokenizer = transformers.AutoTokenizer.from_pretrained("bert-base-cased")

hidden_size = 512
bert_config_tiny = TransformerConfig(
    num_layers = 8,
    num_heads = hidden_size // 64,
    vocab_size = 28996,
    hidden_size = hidden_size,
    max_seq_len = 128,
    dropout = 0.1,
    layer_norm_epsilon = 1e-12
)

config_dict = dict(
    lr=0.0002,
    epochs=40,
    batch_size=64,
    weight_decay=0.01,
    mask_token_id=tokenizer.mask_token_id,
    warmup_step_frac=0.01,
    eps=1e-06,
    max_grad_norm=None,
    device = "cpu",
)


if __name__ == "__main__":

    (train_data, val_data, test_data) = t.load("w2d5/data/wikitext_tokens_2.pt")
    print("Training data size: ", train_data.shape)

    train_loader = DataLoader(
        TensorDataset(train_data), shuffle=True, batch_size=config_dict["batch_size"], drop_last=True
    )

    model = BertLanguageModel(bert_config_tiny)
    num_params = sum((p.nelement() for p in model.parameters()))
    print("Number of model parameters: ", num_params)
    bert_mlm_pretrain(model, config_dict, train_loader, device = config_dict["device"])