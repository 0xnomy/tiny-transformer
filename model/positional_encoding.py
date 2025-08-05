import torch
import torch.nn as nn
import math


def get_positional_encoding(seq_len: int, d_model: int, device: torch.device) -> torch.Tensor:
    pe = torch.zeros(seq_len, d_model, device=device)
    
    position = torch.arange(0, seq_len, device=device).unsqueeze(1).float()
    div_term = torch.exp(torch.arange(0, d_model, 2, device=device).float() * 
                        -(math.log(10000.0) / d_model))
    
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    
    return pe


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_seq_len: int = 5000, dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.max_seq_len = max_seq_len
        
        # Create positional encoding
        pe = get_positional_encoding(max_seq_len, d_model, torch.device('cpu'))
        self.register_buffer('pe', pe)
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        seq_len = x.size(1)
        x = x + self.pe[:seq_len]
        return self.dropout(x) 