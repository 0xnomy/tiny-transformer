# Mini Transformer - English to Pig Latin Translator

A complete PyTorch implementation of a Transformer model for English to Pig Latin translation. This is a working, trainable model with all the essential components.

## What This Is

- **Complete Transformer implementation** from scratch (no `nn.Transformer`)
- **English to Pig Latin translator** using sequence-to-sequence learning
- **Ready to train** with 1000+ sentence pairs
- **Production-ready code** with proper error handling and logging

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Train the model
python main.py --train

# Run inference (uses best_model.pt automatically)
python main.py --inference "hello world"
```

## Project Structure

### Core Files

**`main.py`** - Entry point
- Loads config, creates model, handles CLI arguments
- `--train`: Start training
- `--inference "text"`: Translate text (uses best model automatically)

**`config.yaml`** - Hyperparameters
- Model size, training settings, data config
- Adjust for your hardware/needs

### Model Components (`model/`)

**`attention.py`** - Attention mechanisms
- `ScaledDotProductAttention`: Core attention formula
- `MultiHeadAttention`: Multi-head attention with splitting/concatenation
- Mask functions for padding and causal masking

**`positional_encoding.py`** - Position information
- `PositionalEncoding`: Sinusoidal positional encoding
- `LearnablePositionalEncoding`: Learnable position embeddings

**`encoder.py`** - Transformer encoder
- `EncoderLayer`: Self-attention + FFN + residual connections
- `EncoderStack`: Stack of encoder layers

**`decoder.py`** - Transformer decoder  
- `DecoderLayer`: Self-attention + cross-attention + FFN
- `DecoderStack`: Stack of decoder layers

**`transformer.py`** - Complete model
- `Transformer`: Combines encoder/decoder
- Handles embeddings, final projection, generation

### Data Pipeline (`data/`)

**`dataset.py`** - Data handling
- `Vocabulary`: Token-to-index mapping
- `PigLatinDataset`: PyTorch dataset
- `collate_fn`: Pads sequences in batches
- Loads from `english_to_piglatin.csv`

**`english_to_piglatin.csv`** - Training data
- 1000+ English-Pig Latin pairs
- Format: `english,piglatin`

### Training (`utils/`)

**`training.py`** - Training utilities
- `LabelSmoothingLoss`: Better training stability
- `WarmupCosineScheduler`: Learning rate scheduling
- `EarlyStopping`: Prevents overfitting
- `Trainer`: Complete training loop with logging

## How It Works

### 1. Data Processing
- Loads English-Pig Latin pairs from CSV
- Tokenizes text into word-level tokens
- Creates vocabularies for source/target
- Pads sequences to same length in batches

### 2. Model Architecture
- **Encoder**: Processes English input with self-attention
- **Decoder**: Generates Pig Latin output with:
  - Self-attention (causal masking)
  - Cross-attention to encoder output
  - Feed-forward networks
- **Teacher Forcing**: Uses ground truth during training

### 3. Training Process
- **Loss**: Label smoothing cross-entropy
- **Optimizer**: Adam with warmup + cosine decay
- **Regularization**: Dropout, gradient clipping
- **Monitoring**: Progress bars, sample translations, plots
- **Checkpointing**: Only keeps the best model (`best_model.pt`)

### 4. Inference
- Encodes input English sentence
- Decodes Pig Latin token by token
- Uses greedy decoding (argmax)
- **Automatically loads best model** from training

## Configuration

Key settings in `config.yaml`:

```yaml
# Model size
d_model: 128          # Hidden dimension
n_heads: 4            # Attention heads
n_encoder_layers: 2   # Encoder depth
n_decoder_layers: 2   # Decoder depth

# Training
batch_size: 4         # Adjust for your GPU
num_epochs: 20        # Training epochs
learning_rate: 0.0001 # Learning rate
```

## Training Output

The model will show:
- Progress bars for each epoch
- Loss values and learning rates
- Sample translations every 5 epochs
- Training plots saved to `checkpoints/`
- **Only best model kept** - other checkpoints deleted automatically

## Files Explained

| File | Purpose |
|------|---------|
| `main.py` | Entry point, CLI handling |
| `config.yaml` | All hyperparameters |
| `model/attention.py` | Core attention mechanisms |
| `model/positional_encoding.py` | Position information |
| `model/encoder.py` | Transformer encoder |
| `model/decoder.py` | Transformer decoder |
| `model/transformer.py` | Complete model assembly |
| `data/dataset.py` | Data loading and processing |
| `data/english_to_piglatin.csv` | Training data |
| `utils/training.py` | Training loop and utilities |
| `requirements.txt` | Python dependencies |

## Requirements

- Python 3.7+
- PyTorch
- pandas, tqdm, matplotlib, numpy, pyyaml

## What You Get

- **Working Transformer**: Complete implementation
- **Training Data**: 1000+ English-Pig Latin pairs
- **Training Loop**: With all best practices
- **Inference**: Ready to translate new sentences
- **Clean Code**: No BS, just working code
- **Smart Checkpointing**: Only keeps the best model

## Example Usage

```python
# Train
python main.py --train

# Translate (uses best model automatically)
python main.py --inference "hello world"
# Output: ellohay orldway

# Use specific checkpoint (optional)
python main.py --inference "hello world" --checkpoint path/to/model.pt
```

This is a complete, working Transformer implementation. No fluff, just code that trains and translates.