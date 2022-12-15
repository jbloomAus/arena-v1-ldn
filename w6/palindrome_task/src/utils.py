import os
import pickle
import torch
import json

from transformer_lens import EasyTransformerConfig

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
    
    save_classifier(checkpoint_path, classifier, tokenizer, cfg)
