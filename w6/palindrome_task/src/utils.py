import os
import pickle
import torch
import json
import string

from transformer_lens import EasyTransformerConfig

from .model import Classifier
from .dataset import PalindromeDataset

def save_classifier(model_path, classifier, tokenizer, cfg):

    os.makedirs(model_path, exist_ok=True)

    model_file = os.path.join(model_path, "palindrome_classifier.pt")
    torch.save(classifier.state_dict(), model_file)

    tokenizer_file = os.path.join(model_path, "palindrome_tokenizer.pkl")
    with open(tokenizer_file, "wb") as f:
        pickle.dump(tokenizer, f)

    # save config file 
    config_file = os.path.join(model_path, "palindrome_config.json")
    with open(config_file, "w") as f:
        dictionary = cfg.__dict__
        json.dump(dictionary, f)

def load_classifier(model_path):

    model_file = os.path.join(model_path, "palindrome_classifier.pt")
    classifier = torch.load(model_file)

    tokenizer_file = os.path.join(model_path, "palindrome_tokenizer.pkl")
    with open(tokenizer_file, "rb") as f:
        tokenizer = pickle.load(f)

    config_file = os.path.join(model_path, "palindrome_config.json")
    with open(config_file, "r") as f:
        dictionary = json.load(f)
        cfg = EasyTransformerConfig(**dictionary)

    return classifier, tokenizer, cfg

def save_checkpoint(model_path, classifier, tokenizer,cfg, num_examples):
    
    checkpoint_path = os.path.join(model_path, f"checkpoint.{num_examples}")
    
    if not os.path.exists(checkpoint_path):
        os.makedirs(checkpoint_path)

    save_classifier(checkpoint_path, classifier, tokenizer, cfg)


def load_from_model(path, analysis_data_set_size = 1000):

    classifier_state, tokenizer, cfg = load_classifier(path)

    # # make classifer
    classifier = Classifier(cfg)
    classifier.load_state_dict(classifier_state)

    # make compatible dataset
    k = cfg.n_ctx // 2
    alphabet = string.ascii_lowercase[:tokenizer.vocab_size]

    test_dataset = PalindromeDataset(analysis_data_set_size, k = k, perturb_n_times=8, alphabet=alphabet)
    # test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    return classifier, test_dataset, tokenizer