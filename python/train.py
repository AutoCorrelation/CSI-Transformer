from torch import nn
from torch.optim import Adam


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def initialize_weights(model):
    if hasattr(model, "weight") and model.weight.dim() > 1:
        nn.init.kaiming_uniform_(model.weight.data)

model = Transformer(input_dim=256, d_model=512, nhead=8, num_layers=6, dropout=0.2)        