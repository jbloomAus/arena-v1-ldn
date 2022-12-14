
import torch
import torch.nn as nn
from transformer_lens import EasyTransformer, EasyTransformerConfig

class Classifier(torch.nn.Module):
    def __init__(self, cfg: EasyTransformerConfig):
        '''
        model = Classifier(cfg)
        '''
        super().__init__()
        self.transformer = EasyTransformer(cfg)
        self.transformer.unembed = nn.Identity()
        self.linear = torch.nn.Linear(cfg.d_model, 2)
        
    def forward(self, x):
        x = self.transformer(x)
        x = x[:, 0, :]
        x = self.linear(x)
        return x
