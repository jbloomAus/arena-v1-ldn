# %% 
import os
os.environ["TOKENIZERS_PARALLELISM"] = "true"
import re
import tarfile
import time
from dataclasses import dataclass
from typing import Callable, List, Optional

import pandas as pd
import plotly.express as px
import requests
import torch as t
import torch.nn as nn
import transformers
from build_bert import BERTLanguageMODEL, copy_weights_from_bert
from src.neural_networks import Dropout, Linear
from src.transformers import TransformerConfig
from torch.nn import CrossEntropyLoss, L1Loss
from torch.optim import AdamW
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
import wandb

IMDB_URL = "https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz"
DATA_FOLDER = "./data/bert-imdb/"
IMDB_PATH = os.path.join(DATA_FOLDER, "acllmdb_v1.tar.gz")
SAVED_TOKENS_PATH = os.path.join(DATA_FOLDER, "tokens.pt")

def maybe_download(url: str, path: str) -> None:
    '''
    Download the file from url and save it to path. 
    If path already exists, do nothing.
    '''
    if not os.path.exists(path):
        result = requests.get(url)
    
        with open(path, 'wb') as f:
            f.write(result.content)

os.makedirs(DATA_FOLDER, exist_ok=True)
maybe_download(IMDB_URL, IMDB_PATH)

# %%
@dataclass(frozen=True)
class Review:
    split: str          # "train" or "test"
    is_positive: bool   # sentiment classification
    stars: int          # num stars classification
    text: str           # text content of review

def load_reviews(path: str) -> list[Review]:
    '''
    Load the reviews from the tar.gz file at path.
    Returns a list of Review objects.
    '''
    pattern = re.compile(r"aclImdb\/(train|test)\/(pos|neg)\/\d+_\d+\.txt")
    reviews = []
    with tarfile.open(path) as tar:
        for member in tqdm(tar.getmembers(), desc="Loading reviews"):
            if member.isfile() and member.name.endswith(".txt"):
                if pattern.match(member.name):
                    
                    split = member.name.split("/")[1]
                    is_positive = member.name.split("/")[2] == "pos"
                    stars = int(re.search(r"_(\d+)\.txt", member.name).group(1))
                    text = tar.extractfile(member).read().decode("utf-8")
                    reviews.append(Review(split, is_positive, stars, text))
                    #print(member.name, is_positive, stars)
    return reviews

# %%
# review_df = pd.DataFrame(reviews).sort_values(by="stars", ascending=False)
# review_df.text.apply(len).hist()
# review_df["text_length"] = review_df.text.apply(len)
# review_df["log_text_length"] = review_df.text.apply(len).apply(np.log)
# # %%

# # get a historgram of text lenght by starts
# px.ecdf(review_df, x="text_length", color="stars").show()
# px.violin(review_df, x="text_length", color="stars").show()

# # %% 
# px.ecdf(review_df, x="text_length", color="is_positive").show()
# px.violin(review_df, x="text_length", color="is_positive").show()

# review_df.groupby("is_positive").text.apply(len).plot(kind="bar")
# review_df.groupby("stars").text.count().plot(kind="bar")
# %%
# %%
# !pip install lingua-language-detector
# from IPython import display
# from lingua import Language, LanguageDetectorBuilder

# detector = LanguageDetectorBuilder.from_languages(*Language).build()
# # Note, detector takes much longer to run when it is detecting all languages

# # Sample 500 datapoints, because it takes a while to run
# languages_detected = review_df.sample(500)["text"].apply(detector.detect_language_of).value_counts()
# display(languages_detected)

# %% 
def to_dataset(tokenizer, reviews: list[Review]) -> TensorDataset:
    '''Tokenize the reviews (which should all belong to the same split) and bundle into a TensorDataset.

    The tensors in the TensorDataset should be (in this exact order):

    input_ids: shape (batch, sequence length), dtype int64
    attention_mask: shape (batch, sequence_length), dtype int
    sentiment_labels: shape (batch, ), dtype int
    star_labels: shape (batch, ), dtype int
    '''
    dataset = []
    input_ids = []
    attention_mask = []
    sentiment_labels = []
    star_labels = []
    for review in reviews:
        result = tokenizer(review.text, padding="max_length", max_length = 512, truncation=True)
        input_ids.append(result.input_ids)
        attention_mask.append(result.attention_mask)
        sentiment_labels.append(t.tensor(int(review.is_positive)))
        star_labels.append(t.tensor(review.stars))

    return TensorDataset(t.tensor(input_ids), 
        t.tensor(attention_mask), 
        t.tensor(sentiment_labels), 
        t.tensor(star_labels))

class BertClassifier(nn.Module):

    def __init__(self, trained_bert_common, num_classes):
        super().__init__()
        self.config = trained_bert_common.config
        self.bert = trained_bert_common#BERTCommon(config)
        self.dropout = Dropout(self.config.dropout)
        self.sentiment_classifier = Linear(self.config.hidden_size, num_classes)
        self.star_regressor = Linear(self.config.hidden_size, 1)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids, attention_mask)
        pooled_output = outputs[:, 0, :]
        sentiment_logits = self.sentiment_classifier(pooled_output)
        rating_logits = self.star_regressor(pooled_output).squeeze(1)*10+5
        return sentiment_logits, rating_logits

# %%


if __name__ == "__main__":

    print(f"pytorch version {t.__version__}")
    print(f"default data type: {t.get_default_dtype()}")

    config = {
        'batch_size': 8,
        'hidden_size': 768,
        'lr': 1e-5,
        'seq_len': 512,
        'num_layers': 12,
        'num_heads': 12,
        'vocab_size': 28996,
        'num_epochs': 1,
        'device': 'mps',
        'dropout': 0.1,
        'layer_norm_epsilon': 1e-12,
        'train_set_size': 1000,
        'test_set_size': 1000,
        'num_workers': 2,
    }

    wandb.init(project="Joseph - Fine Tuning BERT on IMDB",
            entity="arena-ldn",
            config=config)

    print("Loading reviews...")
    reviews = load_reviews(IMDB_PATH)
    # assert sum((r.split == "train" for r in reviews)) == 25000
    # assert sum((r.split == "test" for r in reviews)) == 25000

    #print("Tokenizing reviews...")
    # tokenizer = transformers.AutoTokenizer.from_pretrained("bert-base-cased")
    # train_data = to_dataset(tokenizer, [r for r in reviews if r.split == "train"][:wandb.config.train_set_size])
    # test_data = to_dataset(tokenizer, [r for r in reviews if r.split == "test"][:wandb.config.test_set_size])
    # t.save((train_data, test_data), SAVED_TOKENS_PATH)

    train_data, test_data = t.load(SAVED_TOKENS_PATH)
    assert len(train_data) == wandb.config.train_set_size, f"Expected {wandb.config.train_set_size} training examples, got {len(train_data)}"
    assert len(test_data) == wandb.config.test_set_size, f"Expected {wandb.config.test_set_size} test examples, got {len(test_data)}"

    # %%

    transformer_config = TransformerConfig(
        hidden_size=wandb.config.hidden_size,
        num_heads=wandb.config.num_heads,
        num_layers= wandb.config.num_layers,
        layer_norm_epsilon=wandb.config.layer_norm_epsilon,
        max_seq_len=wandb.config.seq_len,
        dropout=wandb.config.dropout,
        vocab_size=wandb.config.vocab_size,
    )

    tokenizer = transformers.AutoTokenizer.from_pretrained("bert-base-cased")
    bert = transformers.BertForMaskedLM.from_pretrained("bert-base-cased")
    my_bert = BERTLanguageMODEL(transformer_config)
    
    print("Copying weights from bert-base-cased to my_bert")
    my_bert = copy_weights_from_bert(my_bert, bert)

    # %% 
    my_bert_classifier = BertClassifier(my_bert.bert, 2)
    trainloader = DataLoader(train_data, batch_size=wandb.config.batch_size, shuffle=True, num_workers=wandb.config.num_workers)
    testloader = DataLoader(test_data, batch_size=wandb.config.batch_size, shuffle=False, num_workers=wandb.config.num_workers)
    print(f"Number of Batches: train: {len(trainloader)}, test {len(testloader)}")

    def train(bert_classifier, trainloader, testloader, epochs = 1, lr = 1e-5, device = "cpu"):
        bert_classifier.to(device)
        categorical_criterion = CrossEntropyLoss()
        regression_criterion = L1Loss()
        optimizer = AdamW(bert_classifier.parameters(), lr=lr)
        loss = 0 

        wandb.watch(bert_classifier, criterion=categorical_criterion, log="parameters", log_freq=10, log_graph=True)

        examples_seen = 0
        since = time.time()
        for epoch in range(epochs):
            pbar = tqdm(enumerate(trainloader, 0))
            for i, data in pbar:
                # get the inputs
                inputs, masks, sentiment_labels, star_labels = data
                inputs, masks, sentiment_labels, star_labels = inputs.to(device), masks.to(device), \
                    sentiment_labels.to(device), star_labels.to(device)
                
                sentiment_logits, star_logits = bert_classifier(inputs, masks)
                sentiment_loss = categorical_criterion(sentiment_logits, sentiment_labels)
                star_loss = regression_criterion(star_logits, star_labels)
                loss += sentiment_loss + star_loss

                examples_seen += len(inputs)

                if (((i+1) % 2)==0):
                    wandb.log({"sentiment_loss": sentiment_loss, 
                            "star_loss": star_loss,
                            "elapsed": time.time() - since}, step=examples_seen)
                    
                    loss.backward()
                    nn.utils.clip_grad_norm_(bert_classifier.parameters(), max_norm=1.0, norm_type=2)
                    optimizer.step()

                    # 0 gradient
                    optimizer.zero_grad()
                    pbar.set_description(f"Epoch {epoch}/{epochs}, Iteration {i}, loss: {loss.item()}")
                    loss = 0

        filename = f"bert_imdb_model.pt"
        print(f"Saving best model to {filename}")
        t.save(bert_classifier.state_dict(), filename)

        return bert_classifier

    trained_bert_classifier = train(
        my_bert_classifier, 
        trainloader, 
        testloader, 
        epochs = wandb.config.num_epochs,
        lr = wandb.config.lr,
        device = wandb.config.device
    )
    
    # %%
    def test(bert_classifier, testloader, device = "cpu"):
    

        bert_classifier.to(device)
        categorical_criterion = CrossEntropyLoss()
        regression_criterion = L1Loss()

        # test the model
        star_loss = 0
        sentiment_loss = 0
        correct_sentiment = 0
        total = 0
        with t.no_grad():
            for data in testloader:
                inputs, masks, sentiment_labels, star_labels = data
                inputs, masks, sentiment_labels, star_labels = inputs.to(device), masks.to(device), \
                    sentiment_labels.to(device), star_labels.to(device)
                sentiment_logits, star_logits = bert_classifier(inputs, masks)
                predicted_star, predicted_sentiment = t.max(sentiment_logits.data, 1)
                total += sentiment_labels.size(0)
                correct_sentiment += (predicted_sentiment == sentiment_labels).sum().item()

                sentiment_loss += categorical_criterion(sentiment_logits, sentiment_labels)
                star_loss += regression_criterion(star_logits, star_labels)

        print(f'Accuracy of the network on the {total} test reviews (sentiment): {100 * correct_sentiment / total}')
        print(f'Average Sentiment loss / batch: {sentiment_loss/len(testloader)}')
        print(f'Avergae Star loss / batch: {star_loss/len(testloader)}')

        wandb.log({"test_sentiment_loss": 100 * correct_sentiment / total, 
        "test_star_loss": sentiment_loss/len(testloader),
        "test_sentiment_accuracy": star_loss/len(testloader)})


    test(trained_bert_classifier, testloader, device = wandb.config.device)