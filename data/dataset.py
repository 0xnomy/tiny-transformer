import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import re
from typing import List, Tuple, Dict, Optional
from collections import Counter

# Load data from CSV file
def load_data_from_csv(csv_path: str = "data/english_to_piglatin.csv") -> List[Tuple[str, str]]:
    """Load English-Pig Latin pairs from CSV file."""
    df = pd.read_csv(csv_path)
    return list(zip(df['english'], df['piglatin']))

# Use the CSV data
EXAMPLE_SENTENCES = [pair[0] for pair in load_data_from_csv()]

class Vocabulary:
    def __init__(self, min_freq: int = 1):
        self.min_freq = min_freq
        self.token2idx = {}
        self.idx2token = {}
        self.token_freq = Counter()
        
        # Special tokens
        self.pad_token = "<pad>"
        self.sos_token = "<sos>"
        self.eos_token = "<eos>"
        self.unk_token = "<unk>"
        
        # Add special tokens
        self.add_token(self.pad_token)
        self.add_token(self.sos_token)
        self.add_token(self.eos_token)
        self.add_token(self.unk_token)
    
    def add_token(self, token: str, freq: int = 1):
        if token not in self.token2idx:
            idx = len(self.token2idx)
            self.token2idx[token] = idx
            self.idx2token[idx] = token
        self.token_freq[token] += freq
    
    def build_vocab(self, texts: List[str], max_vocab_size: Optional[int] = None):
        for text in texts:
            tokens = text.split()
            for token in tokens:
                self.token_freq[token] += 1
        
        filtered_tokens = [
            token for token, freq in self.token_freq.items()
            if freq >= self.min_freq
        ]
        
        filtered_tokens.sort(key=lambda x: self.token_freq[x], reverse=True)
        
        if max_vocab_size is not None:
            available_space = max_vocab_size - len(self.token2idx)
            filtered_tokens = filtered_tokens[:available_space]
        
        for token in filtered_tokens:
            if token not in self.token2idx:
                self.add_token(token)
    
    def encode(self, text: str) -> List[int]:
        tokens = text.split()
        indices = []
        
        for token in tokens:
            if token in self.token2idx:
                indices.append(self.token2idx[token])
            else:
                indices.append(self.token2idx[self.unk_token])
        
        return indices
    
    def decode(self, indices: List[int]) -> str:
        tokens = []
        for idx in indices:
            if idx in self.idx2token:
                token = self.idx2token[idx]
                if token not in [self.pad_token, self.sos_token, self.eos_token]:
                    tokens.append(token)
        
        return ' '.join(tokens)
    
    def __len__(self) -> int:
        return len(self.token2idx)

class PigLatinDataset(Dataset):
    def __init__(
        self,
        english_texts: List[str],
        pig_latin_texts: List[str],
        src_vocab: Vocabulary,
        tgt_vocab: Vocabulary,
        max_length: int = 100
    ):
        self.english_texts = english_texts
        self.pig_latin_texts = pig_latin_texts
        self.src_vocab = src_vocab
        self.tgt_vocab = tgt_vocab
        self.max_length = max_length
        
        assert len(english_texts) == len(pig_latin_texts)
    
    def __len__(self) -> int:
        return len(self.english_texts)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        english_text = self.english_texts[idx]
        pig_latin_text = self.pig_latin_texts[idx]
        
        src_tokens = self.src_vocab.encode(english_text)
        tgt_tokens = self.tgt_vocab.encode(pig_latin_text)
        
        src_tokens = [self.src_vocab.token2idx[self.src_vocab.sos_token]] + src_tokens + [self.src_vocab.token2idx[self.src_vocab.eos_token]]
        tgt_tokens = [self.tgt_vocab.token2idx[self.tgt_vocab.sos_token]] + tgt_tokens + [self.tgt_vocab.token2idx[self.tgt_vocab.eos_token]]
        
        src_tokens = src_tokens[:self.max_length]
        tgt_tokens = tgt_tokens[:self.max_length]
        
        src_tensor = torch.tensor(src_tokens, dtype=torch.long)
        tgt_tensor = torch.tensor(tgt_tokens, dtype=torch.long)
        
        return src_tensor, tgt_tensor

def collate_fn(batch: List[Tuple[torch.Tensor, torch.Tensor]], pad_idx: int = 0) -> Tuple[torch.Tensor, torch.Tensor]:
    src_tensors, tgt_tensors = zip(*batch)
    
    src_max_len = max(tensor.size(0) for tensor in src_tensors)
    tgt_max_len = max(tensor.size(0) for tensor in tgt_tensors)
    
    padded_src = []
    for tensor in src_tensors:
        padding_size = src_max_len - tensor.size(0)
        if padding_size > 0:
            padded = torch.cat([tensor, torch.full((padding_size,), pad_idx, dtype=torch.long)])
        else:
            padded = tensor
        padded_src.append(padded)
    
    padded_tgt = []
    for tensor in tgt_tensors:
        padding_size = tgt_max_len - tensor.size(0)
        if padding_size > 0:
            padded = torch.cat([tensor, torch.full((padding_size,), pad_idx, dtype=torch.long)])
        else:
            padded = tensor
        padded_tgt.append(padded)
    
    src_batch = torch.stack(padded_src)
    tgt_batch = torch.stack(padded_tgt)
    
    return src_batch, tgt_batch

def create_pig_latin_dataset(
    english_sentences: List[str],
    max_vocab_size: int = 10000,
    max_length: int = 100,
    train_split: float = 0.8,
    val_split: float = 0.1,
    test_split: float = 0.1,
    batch_size: int = 32,
    shuffle: bool = True
) -> Tuple[DataLoader, DataLoader, DataLoader, Vocabulary, Vocabulary]:
    
    assert abs(train_split + val_split + test_split - 1.0) < 1e-6
    
    # Load the CSV data
    data_pairs = load_data_from_csv()
    english_texts = [pair[0] for pair in data_pairs]
    pig_latin_texts = [pair[1] for pair in data_pairs]
    
    # Create vocabularies
    src_vocab = Vocabulary()
    tgt_vocab = Vocabulary()
    
    src_vocab.build_vocab(english_texts, max_vocab_size)
    tgt_vocab.build_vocab(pig_latin_texts, max_vocab_size)
    
    # Split data
    total_size = len(english_texts)
    train_size = int(total_size * train_split)
    val_size = int(total_size * val_split)
    
    train_dataset = PigLatinDataset(
        english_texts[:train_size],
        pig_latin_texts[:train_size],
        src_vocab,
        tgt_vocab,
        max_length
    )
    
    val_dataset = PigLatinDataset(
        english_texts[train_size:train_size + val_size],
        pig_latin_texts[train_size:train_size + val_size],
        src_vocab,
        tgt_vocab,
        max_length
    )
    
    test_dataset = PigLatinDataset(
        english_texts[train_size + val_size:],
        pig_latin_texts[train_size + val_size:],
        src_vocab,
        tgt_vocab,
        max_length
    )
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    
    return train_loader, val_loader, test_loader, src_vocab, tgt_vocab 