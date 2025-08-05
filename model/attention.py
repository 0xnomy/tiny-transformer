import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple


class ScaledDotProductAttention(nn.Module):
    def __init__(self, d_k: int, dropout: float = 0.1):
        super().__init__()
        self.d_k = d_k
        self.scale = math.sqrt(d_k)
        self.dropout = nn.Dropout(dropout)
    
    def forward(
        self, 
        query: torch.Tensor, 
        key: torch.Tensor, 
        value: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # Compute attention scores: QK^T
        # Shape: (batch_size, seq_len, seq_len)
        scores = torch.matmul(query, key.transpose(-2, -1))
        
        # Scale the scores by √d_k
        scores = scores / self.scale
        
        # Apply mask if provided (set masked positions to -inf)
        if mask is not None:
            # Ensure mask has the right shape for broadcasting
            if mask.dim() == 4:  # (batch_size, 1, seq_len, seq_len)
                mask = mask.squeeze(1)  # Remove the extra dimension
            scores = scores.masked_fill(mask == 0, -1e9)
        
        # Apply softmax to get attention weights
        # Shape: (batch_size, seq_len, seq_len)
        attention_weights = F.softmax(scores, dim=-1)
        
        # Apply dropout to attention weights
        attention_weights = self.dropout(attention_weights)
        
        # Apply attention weights to values: softmax(QK^T/√d_k)V
        # Shape: (batch_size, seq_len, d_v)
        output = torch.matmul(attention_weights, value)
        
        return output, attention_weights


class MultiHeadAttention(nn.Module):
    
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        self.d_v = d_model // n_heads
        
        # Linear projections for Q, K, V
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        
        # Output projection
        self.w_o = nn.Linear(d_model, d_model)
        
        # Attention mechanism
        self.attention = ScaledDotProductAttention(self.d_k, dropout)
        
        # Dropout for output
        self.dropout = nn.Dropout(dropout)
        
    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute multi-head attention.
        """
        batch_size, seq_len, d_model = query.size()
        assert d_model == self.d_model, f"Expected d_model={self.d_model}, got {d_model}"

        # Get key and value sequence lengths (they might be different for cross-attention)
        _, key_seq_len, _ = key.size()
        _, value_seq_len, _ = value.size()

        # Linear projections
        Q = self.w_q(query)
        K = self.w_k(key)
        V = self.w_v(value)

        # Reshape for multi-head: (batch_size, seq_len, n_heads, d_k) -> (batch_size, n_heads, seq_len, d_k)
        Q = Q.view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2).contiguous()
        K = K.view(batch_size, key_seq_len, self.n_heads, self.d_k).transpose(1, 2).contiguous()
        V = V.view(batch_size, value_seq_len, self.n_heads, self.d_k).transpose(1, 2).contiguous()

        # If mask is provided, expand for multi-head
        if mask is not None:
            # Ensure mask has the right shape for broadcasting
            if mask.dim() == 2:
                mask = mask.unsqueeze(0).unsqueeze(0)  # (1, 1, seq_len, seq_len)
            elif mask.dim() == 3:
                mask = mask.unsqueeze(1)  # (batch_size, 1, seq_len, seq_len)

        # Apply attention to each head
        outputs = []
        attention_weights = []
        for head in range(self.n_heads):
            q_head = Q[:, head, :, :]  # (batch_size, seq_len, d_k)
            k_head = K[:, head, :, :]
            v_head = V[:, head, :, :]
            
            # Pass the mask as is - ScaledDotProductAttention will handle it
            head_output, head_weights = self.attention(q_head, k_head, v_head, mask)
            outputs.append(head_output)
            attention_weights.append(head_weights)

        output = torch.cat(outputs, dim=-1)
        
        output = self.w_o(output)
        output = self.dropout(output)
        
        # Ensure output has correct shape
        assert output.shape == (batch_size, seq_len, self.d_model), f"Output shape {output.shape} != expected {(batch_size, seq_len, self.d_model)}"
        
        attention_weights = torch.stack(attention_weights, dim=1)  # (batch_size, n_heads, seq_len, seq_len)
        return output, attention_weights


def create_padding_mask(seq: torch.Tensor, pad_idx: int = 0) -> torch.Tensor:
    mask = (seq != pad_idx).unsqueeze(1).unsqueeze(2)
    return mask


def create_causal_mask(seq_len: int, device: torch.device) -> torch.Tensor:
    mask = torch.triu(torch.ones(seq_len, seq_len, device=device), diagonal=1)
    # Invert: 1 for positions to attend to, 0 for masked positions
    mask = (mask == 0).unsqueeze(0).unsqueeze(0)
    return mask 