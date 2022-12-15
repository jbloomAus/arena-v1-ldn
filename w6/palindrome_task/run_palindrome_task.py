import os
import argparse
import logging
import string

import torch
from src.dataset import PalindromeDataset
from src.fun_run_naming import get_random_name
from src.model import Classifier
from src.tokenizer import ToyTokenizer
from src.train import train
from src.utils import save_classifier
from torch.utils.data import DataLoader
from transformer_lens import EasyTransformer, EasyTransformerConfig

logging.getLogger("asyncio").setLevel(logging.INFO)

# format logger 
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--total_examples", type=int, default=5*10**5)
    parser.add_argument("--size_alphabet", type=int, default=10)
    parser.add_argument("--sequence_length", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    
    parser.add_argument("--test_interval", type=int, default=1000)
    parser.add_argument("--save_interval", type=int, default=1000)
    parser.add_argument("--save_path", type=str, default="models")

    parser.add_argument("--n_layers", type=int, default=3)
    parser.add_argument("--n_heads", type=int, default=2)
    parser.add_argument("--d_head", type=int, default=16)
    parser.add_argument("--d_model", type=int, default=32)
    parser.add_argument("--d_mlp", type=int, default=64)
    parser.add_argument("--d_vocab_out", type=int, default=64)

    parser.add_argument("--use_wandb", type=int, default=1)

    args = parser.parse_args()

    logger = logging.getLogger(__name__)

    run_name = get_random_name()
    save_path = os.path.join(args.save_path, run_name)

    logger.info("Beginning Palindrome Task - {}".format(run_name))

    if args.use_wandb:
        import wandb
        wandb.init(project="palindrome_task", name=run_name, config=args, entity="arena-ldn")

    loss_fn = torch.nn.CrossEntropyLoss()
    logger.info("Loss function: {}".format(loss_fn))

    alphabet = string.ascii_lowercase[:args.size_alphabet]
    logger.info("Alphabet: {}, length {}".format(alphabet, len(alphabet)))

    sequence_length = args.sequence_length
    logger.info(f"Sequence length: {sequence_length}")

    assert sequence_length % 2 == 0, "Sequence length must be even"
    assert sequence_length > 3, "Sequence length must be greater than 3"
    k = sequence_length // 2 

    logging.info("Batch size: {}".format(args.batch_size))

    train_dataset = PalindromeDataset(args.total_examples, k = k, perturb_n_times=8, alphabet=alphabet)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    test_dataset = PalindromeDataset(1000, k = k, perturb_n_times=8, alphabet=alphabet)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
    
    tokenizer = ToyTokenizer(
        alphabet, 
        pad=False,  # we are providing sequences of fixed length
        sep=True, # adds a SEP token at the end of the sequence
        cls=False, # don't need a CLS token
    )

    cfg = EasyTransformerConfig(
        n_layers=args.n_layers,
        d_model=args.d_model,
        d_head=args.d_head,
        n_heads=args.n_heads,
        d_mlp=args.d_mlp,
        d_vocab= len(alphabet) + 1, # +2 for the CLS and SEP tokens
        n_ctx=sequence_length + 1, # +1 for the CLS token
        act_fn="relu",
        normalization_type=None,
        attention_dir="causal",
        d_vocab_out=args.d_vocab_out,
    )

    # logging.info("Model config: {}".format(cfg))

    model = EasyTransformer(cfg)
    classifier = Classifier(cfg)
    classifier.to(args.device)

    optimizer = torch.optim.Adam(classifier.parameters(), lr=1e-4)

    logger.info("Training model")
    classifier = train(
        classifier = classifier,
        train_loader= train_loader,
        test_loader= test_loader,
        tokenizer = tokenizer,
        optimizer = optimizer,
        loss_fn = loss_fn,
        device = args.device,
        test_interval= args.test_interval,
        save_interval = args.save_interval,
        save_path = save_path,
        use_wandb = args.use_wandb,
    )

    # save final model 
    logger.info("Saving final model to {}".format(save_path))
    save_classifier(save_path, classifier, tokenizer, cfg)
    save_tokenizer_and_dataset(save_path, tokenizer, train_dataset)

