
import torch
import torch.nn as nn
from transformer_lens import EasyTransformer, EasyTransformerConfig

class Classifier(torch.nn.Module):
    def __init__(self, cfg: EasyTransformerConfig):
        '''
        model = Classifier(cfg)
        '''
        super().__init__()

        assert cfg.attention_dir == "causal", "Attention direction must be causal"
        assert cfg.normalization_type is None, "Normalization type must be None"

        self.transformer = EasyTransformer(cfg)
        self.transformer.unembed = nn.Identity() # Don't unembed
        self.linear = torch.nn.Linear(cfg.d_model, 2) # 2 classes
        
    def forward(self, x):
        x = self.transformer(x)
        x = x[:, -1, :] # Take the last token
        x = self.linear(x)
        return x