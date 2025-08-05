import argparse
import yaml
import torch
import torch.optim as optim
import os
import sys
from typing import Dict, Any

# Add the project root to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from model.transformer import Transformer
from data.dataset import create_pig_latin_dataset, EXAMPLE_SENTENCES
from utils.training import Trainer, WarmupCosineScheduler


def load_config(config_path: str = "config.yaml") -> Dict[str, Any]:
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def create_model(config: Dict[str, Any], src_vocab_size: int, tgt_vocab_size: int) -> Transformer:
    model_config = config['model']
    
    model = Transformer(
        src_vocab_size=src_vocab_size,
        tgt_vocab_size=tgt_vocab_size,
        d_model=model_config['d_model'],
        n_heads=model_config['n_heads'],
        n_encoder_layers=model_config['n_encoder_layers'],
        n_decoder_layers=model_config['n_decoder_layers'],
        d_ff=model_config['d_ff'],
        max_seq_length=model_config['max_seq_length'],
        dropout=model_config['dropout'],
        pad_idx=0
    )
    
    return model


def create_optimizer_and_scheduler(
    model: Transformer, 
    config: Dict[str, Any], 
    total_steps: int
) -> tuple:
    training_config = config['training']
    
    # Create optimizer
    optimizer = optim.Adam(
        model.parameters(),
        lr=training_config['learning_rate'],
        betas=(0.9, 0.98),
        eps=1e-9
    )
    
    # Create scheduler
    scheduler = WarmupCosineScheduler(
        optimizer=optimizer,
        max_lr=training_config['learning_rate'],
        warmup_steps=training_config['warmup_steps'],
        total_steps=total_steps
    )
    
    return optimizer, scheduler


def train_model(config: Dict[str, Any]):
    print("Loading dataset...")
    
    # Create dataset and data loaders
    train_loader, val_loader, test_loader, src_vocab, tgt_vocab = create_pig_latin_dataset(
        english_sentences=EXAMPLE_SENTENCES,
        max_vocab_size=config['data']['vocab_size'],
        max_length=config['model']['max_seq_length'],
        batch_size=config['training']['batch_size'],
        train_split=config['data']['train_split'],
        val_split=config['data']['val_split'],
        test_split=config['data']['test_split']
    )
    
    print(f"Dataset loaded:")
    print(f"  Source vocabulary size: {len(src_vocab)}")
    print(f"  Target vocabulary size: {len(tgt_vocab)}")
    print(f"  Training batches: {len(train_loader)}")
    print(f"  Validation batches: {len(val_loader)}")
    
    # Create model
    print("Creating model...")
    model = create_model(config, len(src_vocab), len(tgt_vocab))
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create optimizer and scheduler
    total_steps = len(train_loader) * config['training']['num_epochs']
    optimizer, scheduler = create_optimizer_and_scheduler(model, config, total_steps)
    
    # Create trainer
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        label_smoothing=config['training']['label_smoothing'],
        grad_clip=config['training']['grad_clip'],
        save_dir=config['training']['checkpoint_dir'],
        log_interval=config['training'].get('log_interval', 100)
    )
    
    # Train the model
    print("Starting training...")
    trainer.train(
        num_epochs=config['training']['num_epochs'],
        src_vocab=src_vocab,
        tgt_vocab=tgt_vocab,
        early_stopping_patience=config['training'].get('patience', 5)
    )
    
    print("Training completed!")


def run_inference(config: Dict[str, Any], input_text: str, checkpoint_path: str = None):
    print("Loading dataset for vocabularies...")
    
    # Create dataset to get vocabularies
    train_loader, val_loader, test_loader, src_vocab, tgt_vocab = create_pig_latin_dataset(
        english_sentences=EXAMPLE_SENTENCES,
        max_vocab_size=config['data']['vocab_size'],
        max_length=config['model']['max_seq_length'],
        batch_size=1,
        train_split=config['data']['train_split'],
        val_split=config['data']['val_split'],
        test_split=config['data']['test_split']
    )
    
    # Create model
    model = create_model(config, len(src_vocab), len(tgt_vocab))
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    
    # Use best_model.pt by default if no checkpoint specified
    if checkpoint_path is None:
        checkpoint_path = os.path.join(config['training']['checkpoint_dir'], 'best_model.pt')
    
    # Load checkpoint if provided
    if checkpoint_path and os.path.exists(checkpoint_path):
        print(f"Loading checkpoint from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        print("No checkpoint found, using untrained model")
    
    # Tokenize input
    src_tokens = src_vocab.encode(input_text)
    src_tokens = [src_vocab.token2idx[src_vocab.sos_token]] + src_tokens + [src_vocab.token2idx[src_vocab.eos_token]]
    src_tensor = torch.tensor([src_tokens], dtype=torch.long, device=device)
    
    # Generate translation
    print(f"Input: {input_text}")
    with torch.no_grad():
        translation = model.generate(
            src=src_tensor,
            src_vocab=src_vocab,
            tgt_vocab=tgt_vocab,
            max_length=config['model']['max_seq_length']
        )
    
    print(f"Translation: {translation}")


def main():
    parser = argparse.ArgumentParser(description='Mini Transformer Training and Inference')
    parser.add_argument('--train', action='store_true', help='Train the model')
    parser.add_argument('--inference', type=str, help='Run inference on input text')
    parser.add_argument('--checkpoint', type=str, help='Path to model checkpoint (defaults to best_model.pt)')
    parser.add_argument('--config', type=str, default='config.yaml', help='Path to config file')
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    if args.train:
        train_model(config)
    elif args.inference:
        run_inference(config, args.inference, args.checkpoint)
    else:
        print("Please specify --train or --inference")
        parser.print_help()


if __name__ == "__main__":
    main() 