import torch 
import string

import argparse
from torch.utils.data import DataLoader
from src.dataset import PalindromeDataset
from src.model import Classifier
from src.tokenizer import ToyTokenizer
from src.train import train
from transformer_lens import EasyTransformer, EasyTransformerConfig
from palindrome_task import PalindromeDataset, Classifier, ToyTokenizer

import logging 

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
    parser.add_argument("--total_examples", type=int, default=1*10**6)
    parser.add_argument("--loss_fn", type=str, default="CrossEntropyLoss")
    parser.add_argument("--size_alphabet", type=int, default=10)
    parser.add_argument("--sequence_length", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--d_head", type=int, default=16)
    parser.add_argument("--n_layers", type=int, default=3)
    parser.add_argument("--n_heads", type=int, default=2)
    parser.add_argument("--d_mlp", type=int, default=64)
    parser.add_argument("--d_vocab_out", type=int, default=64)
    parser.add_argument("--n_ctx", type=int, default=12)
    parser.add_argument("--act_fn", type=str, default="relu")
    parser.add_argument("--normalization_type", type=str, default=None)
    parser.add_argument("--attention_dir", type=str, default="bidirectional")
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--n_epochs", type=int, default=10)
    
    
    parser.add_argument("--test_interval", type=int, default=1000)
    parser.add_argument("--save_interval", type=int, default=1000)
    parser.add_argument("--save_path", type=str, default="model.pt")

    args = parser.parse_args()

    logger = logging.getLogger(__name__)

    logger.info("Beginning Palindrome Task")

    loss_fn = torch.nn.CrossEntropyLoss()
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
    test_dataset = PalindromeDataset(args.total_examples, k = k, perturb_n_times=8, alphabet=alphabet)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    d_head = 16

    cfg = EasyTransformerConfig(
        n_layers=3,
        d_model=d_head*2,
        d_head=d_head,
        n_heads=2,
        d_mlp=d_head*4,
        d_vocab= len(alphabet) + 3,
        n_ctx=sequence_length + 2,
        act_fn="relu",
        normalization_type=None,
        attention_dir="causal",
        d_vocab_out=64,
    )


    model = EasyTransformer(cfg)
    classifier = Classifier(cfg)
    classifier.to(args.device)
    tokenizer = ToyTokenizer(alphabet)

    optimizer = torch.optim.Adam(classifier.parameters(), lr=1e-4)

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
    )

    

