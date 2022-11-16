import torch as t
import torch.nn as nn
from src.train import train
from shakespeare_utils import WordsTokenizer, WordDataset
from src.transformers import TransformerConfig, DecoderOnlyTransformer
from torch.utils.data import DataLoader
from multiprocessing import cpu_count
import wandb

print(f"pytorch version {t.__version__}")
print(f"default data type: {t.get_default_dtype()}")

if __name__ == '__main__':

    config = {
    'batch_size': 64,
    'hidden_size': 512,
    'lr': 6e-4,
    'seq_len': 128,
    'num_layers': 8,
    'num_heads': 8,
    'vocab_size': 34543,
    'num_epochs': 1,
    'device': 'cpu',
    'dropout': 0.1,
    'layer_norm_epsilon': 1e-5,
    'train_set_size': 50 * 10**4,
    'test_set_size': 1000,
    'num_workers': 0,
    }

    wandb.init(project="W1D3 Shakespeare Transformer Tilman and Joseph",
            entity="arena-ldn",
            #name ="MPS - Rescale embedding Variance - batch size 16, seq len 120",
            config=config)

    # get data
    train_set_size = wandb.config.train_set_size
    test_set_size = wandb.config.test_set_size

    shakespeare_text = open('w1d3/shakespeare.txt', 'r').read()

    train_dataset = WordDataset(shakespeare_text,
                          block_size=wandb.config.seq_len,
                          overwrite_length=train_set_size)
    
    test_dataset = WordDataset(shakespeare_text,
                        block_size=wandb.config.seq_len,
                        overwrite_length=test_set_size)

    word_tokenizer = WordsTokenizer(train_dataset)
    trainloader = DataLoader(train_dataset,
                            batch_size=wandb.config.batch_size,
                            shuffle=True,
                            pin_memory=True,
                            num_workers=wandb.config.num_workers)

    testloader = DataLoader(test_dataset,
                            batch_size=wandb.config.batch_size,
                            shuffle=True,
                            pin_memory=True,
                            num_workers=wandb.config.num_workers)


    batch_size = wandb.config.batch_size
    seq_len = wandb.config.seq_len
    transformer_config = TransformerConfig(
        num_layers=wandb.config.num_layers,
        num_heads=wandb.config.num_heads,
        vocab_size=wandb.config.vocab_size,
        hidden_size=wandb.config.hidden_size,
        max_seq_len=wandb.config.seq_len,
        dropout=wandb.config.dropout,
        layer_norm_epsilon=wandb.config.layer_norm_epsilon)

    model = DecoderOnlyTransformer(config=transformer_config)

    # from torchinfo import torchinfo
    # torchinfo.summary(model,
    #                   input_data=t.zeros((1, wandb.config.seq_len)))

    # run model
    num_epochs = wandb.config.num_epochs
    device = t.device(wandb.config.device)
    criterion = nn.CrossEntropyLoss()
    optimizer = t.optim.Adam(model.parameters(), lr=wandb.config.lr)
    model = train(model,
                  word_tokenizer,
                  optimizer,
                  trainloader,
                  testloader,
                  criterion,
                  num_epochs=num_epochs,
                  device=device)
