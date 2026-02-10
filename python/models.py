import math
import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x):
        x = x + self.pe[:, : x.size(1)]
        return self.dropout(x)


class CSITransformer(nn.Module):
    def __init__(self, input_dim, d_model=256, nhead=8, num_layers=4, dropout=0.1):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, d_model)
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        self.pos_decoder = PositionalEncoding(d_model, dropout)
        self.transformer = nn.Transformer(
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=num_layers,
            num_decoder_layers=num_layers,
            dropout=dropout,
            batch_first=True,
        )
        self.output_proj = nn.Linear(d_model, input_dim)

    @staticmethod
    def _causal_mask(size, device):
        return torch.triu(torch.ones(size, size, device=device), diagonal=1).bool()

    def forward(self, src, tgt=None):
        if tgt is None:
            tgt = src

        src = self.input_proj(src)
        tgt = self.input_proj(tgt)

        src = self.pos_encoder(src)
        tgt = self.pos_decoder(tgt)

        tgt_mask = self._causal_mask(tgt.size(1), tgt.device)
        out = self.transformer(src, tgt, tgt_mask=tgt_mask)
        out = out[:, -1, :]
        return self.output_proj(out)
