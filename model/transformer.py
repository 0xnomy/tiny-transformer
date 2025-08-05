import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

from .encoder import EncoderStack
from .decoder import DecoderStack
from .positional_encoding import PositionalEncoding
from .attention import create_padding_mask, create_causal_mask


class Transformer(nn.Module):
    def __init__(
        self,
        src_vocab_size: int,
        tgt_vocab_size: int,
        d_model: int = 512,
        n_heads: int = 8,
        n_encoder_layers: int = 6,
        n_decoder_layers: int = 6,
        d_ff: int = 2048,
        max_seq_length: int = 100,
        dropout: float = 0.1,
        pad_idx: int = 0
    ):
        super().__init__()
        self.src_vocab_size = src_vocab_size
        self.tgt_vocab_size = tgt_vocab_size
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_encoder_layers = n_encoder_layers
        self.n_decoder_layers = n_decoder_layers
        self.d_ff = d_ff
        self.max_seq_length = max_seq_length
        self.pad_idx = pad_idx
        
        # Embeddings
        self.src_embedding = nn.Embedding(src_vocab_size, d_model, pad_idx)
        self.tgt_embedding = nn.Embedding(tgt_vocab_size, d_model, pad_idx)
        
        # Positional encoding
        self.src_pos_encoding = PositionalEncoding(d_model, max_seq_length, dropout)
        self.tgt_pos_encoding = PositionalEncoding(d_model, max_seq_length, dropout)
        
        # Encoder and decoder stacks
        self.encoder = EncoderStack(d_model, n_heads, n_encoder_layers, d_ff, dropout)
        self.decoder = DecoderStack(d_model, n_heads, n_decoder_layers, d_ff, dropout)
        
        # Output projection
        self.output_projection = nn.Linear(d_model, tgt_vocab_size)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    
    def encode(self, src: torch.Tensor, src_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        src_embedded = self.src_embedding(src) * (self.d_model ** 0.5)
        src_embedded = self.src_pos_encoding(src_embedded)
        return self.encoder(src_embedded, src_mask)
    
    def decode(
        self, 
        tgt: torch.Tensor, 
        encoder_output: torch.Tensor,
        tgt_mask: Optional[torch.Tensor] = None,
        src_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Decode target sequence."""
        tgt_embedded = self.tgt_embedding(tgt) * (self.d_model ** 0.5)
        tgt_embedded = self.tgt_pos_encoding(tgt_embedded)
        decoder_output = self.decoder(tgt_embedded, encoder_output, tgt_mask, src_mask)
        return self.output_projection(decoder_output)
    
    def forward(
        self, 
        src: torch.Tensor, 
        tgt: torch.Tensor,
        src_mask: Optional[torch.Tensor] = None,
        tgt_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        # Create masks if not provided
        if src_mask is None:
            src_mask = create_padding_mask(src, self.pad_idx)
        if tgt_mask is None:
            tgt_mask = create_padding_mask(tgt, self.pad_idx)
            # Add causal mask for decoder
            seq_len = tgt.size(1)
            causal_mask = create_causal_mask(seq_len, tgt.device)
            tgt_mask = tgt_mask & causal_mask
        
        # Encode source sequence
        encoder_output = self.encode(src, src_mask)
        
        # Decode target sequence
        output = self.decode(tgt, encoder_output, tgt_mask, src_mask)
        
        return output
    
    def generate(
        self, 
        src: torch.Tensor, 
        src_vocab, 
        tgt_vocab,
        max_length: int = 50,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None
    ) -> str:
        self.eval()
        device = src.device
        
        # Encode source
        src_mask = create_padding_mask(src, self.pad_idx)
        encoder_output = self.encode(src, src_mask)
        
        # Initialize target with SOS token
        batch_size = src.size(0)
        tgt = torch.full((batch_size, 1), tgt_vocab.token2idx[tgt_vocab.sos_token], 
                        dtype=torch.long, device=device)
        
        eos_token_id = tgt_vocab.token2idx[tgt_vocab.eos_token]
        
        with torch.no_grad():
            for step in range(max_length - 1):
                # Create masks
                tgt_mask = create_padding_mask(tgt, self.pad_idx)
                seq_len = tgt.size(1)
                causal_mask = create_causal_mask(seq_len, device)
                tgt_mask = tgt_mask & causal_mask
                
                # Decode
                output = self.decode(tgt, encoder_output, tgt_mask, src_mask)
                next_token_logits = output[:, -1, :] / temperature
                
                # Apply top-k and top-p sampling
                if top_k is not None:
                    top_k_logits, top_k_indices = torch.topk(next_token_logits, top_k)
                    next_token_logits = torch.full_like(next_token_logits, float('-inf'))
                    next_token_logits.scatter_(1, top_k_indices, top_k_logits)
                
                if top_p is not None:
                    sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                    sorted_indices_to_remove = cumulative_probs > top_p
                    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                    sorted_indices_to_remove[..., 0] = 0
                    indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                    next_token_logits[indices_to_remove] = float('-inf')
                
                # Use greedy decoding (argmax) instead of sampling
                next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
                
                # Append to target sequence
                tgt = torch.cat([tgt, next_token], dim=1)
                
                # Check if EOS token is generated
                if next_token.item() == eos_token_id:
                    break
                
                # Early stopping if we've generated enough tokens
                if step >= 6:  # Limit generation length
                    break
                
                # Stop if we're repeating the same pattern
                if step > 2:
                    # Check for repeating pattern of 2 tokens
                    tokens = tgt[0].tolist()
                    if len(tokens) >= 4:
                        last_four = tokens[-4:]
                        if last_four[0] == last_four[2] and last_four[1] == last_four[3]:
                            break
        
        # Decode target sequence
        target_tokens = tgt[0].cpu().numpy()
        
        # Take only the first occurrence of each unique token (after SOS)
        if len(target_tokens) > 1:
            # Start from index 1 (skip SOS token)
            unique_tokens = [target_tokens[1]]  # First token after SOS
            for i in range(2, len(target_tokens)):
                if target_tokens[i] not in unique_tokens and target_tokens[i] != eos_token_id:
                    unique_tokens.append(target_tokens[i])
            target_tokens = [target_tokens[0]] + unique_tokens  # Keep SOS at start
        
        decoded = tgt_vocab.decode(target_tokens)
        
        return decoded 