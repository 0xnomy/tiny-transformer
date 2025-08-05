import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from typing import Dict, List, Optional, Tuple
import time
import os
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np

from data.dataset import Vocabulary


class LabelSmoothingLoss(nn.Module):
    def __init__(self, vocab_size: int, smoothing: float = 0.1, ignore_index: int = 0):
        super().__init__()
        self.vocab_size = vocab_size
        self.smoothing = smoothing
        self.ignore_index = ignore_index
        
        self.confidence = 1.0 - smoothing
        self.smoothing_value = smoothing / (vocab_size - 1)
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # Create smoothed target distribution
        target_dist = torch.zeros_like(pred)
        target_dist.fill_(self.smoothing_value)
        
        target_dist.scatter_(1, target.unsqueeze(1), self.confidence)
        target_dist[target == self.ignore_index] = 0.0
        
        # Compute cross-entropy loss
        loss = -torch.sum(target_dist * torch.log_softmax(pred, dim=-1), dim=-1)
        
        # Mask out padding positions
        mask = (target != self.ignore_index).float()
        loss = (loss * mask).sum() / mask.sum()
        
        return loss


class WarmupCosineScheduler:
    def __init__(
        self,
        optimizer: optim.Optimizer,
        max_lr: float,
        warmup_steps: int,
        total_steps: int,
        min_lr: float = 0.0
    ):
        self.optimizer = optimizer
        self.max_lr = max_lr
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.min_lr = min_lr
        self.current_step = 0
    
    def step(self):
        self.current_step += 1
        
        if self.current_step <= self.warmup_steps:
            lr = self.max_lr * (self.current_step / self.warmup_steps)
        else:
            progress = (self.current_step - self.warmup_steps) / (self.total_steps - self.warmup_steps)
            lr = self.min_lr + (self.max_lr - self.min_lr) * 0.5 * (1 + np.cos(np.pi * progress))
        
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
    
    def get_last_lr(self) -> List[float]:
        return [group['lr'] for group in self.optimizer.param_groups]


class EarlyStopping:
    def __init__(self, patience: int = 5, min_delta: float = 0.0, restore_best_weights: bool = True):
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        self.best_loss = float('inf')
        self.counter = 0
        self.best_weights = None
    
    def __call__(self, val_loss: float, model: nn.Module) -> bool:
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            if self.restore_best_weights:
                self.best_weights = model.state_dict().copy()
        else:
            self.counter += 1
        
        return self.counter >= self.patience
    
    def restore_weights(self, model: nn.Module):
        if self.best_weights is not None:
            model.load_state_dict(self.best_weights)


class Trainer:
    def __init__(
        self,
        model: nn.Module, # Changed from Transformer to nn.Module as Transformer is removed
        train_loader: DataLoader,
        val_loader: DataLoader,
        optimizer: optim.Optimizer,
        scheduler: WarmupCosineScheduler,
        device: torch.device,
        label_smoothing: float = 0.1,
        grad_clip: float = 1.0,
        save_dir: str = "checkpoints",
        log_interval: int = 100
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.grad_clip = grad_clip
        self.save_dir = save_dir
        self.log_interval = log_interval
        
        # Loss function with label smoothing
        self.criterion = LabelSmoothingLoss(
            vocab_size=model.tgt_vocab_size, # Assuming model has tgt_vocab_size attribute
            smoothing=label_smoothing,
            ignore_index=0
        )
        
        # Training history
        self.train_losses = []
        self.val_losses = []
        self.learning_rates = []
        
        os.makedirs(save_dir, exist_ok=True)
        self.model.to(device)
    
    def train_epoch(self, epoch: int) -> float:
        self.model.train()
        total_loss = 0.0
        num_batches = len(self.train_loader)
        
        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch}")
        
        for batch_idx, (src, tgt) in enumerate(pbar):
            src = src.to(self.device)
            tgt = tgt.to(self.device)
            
            # Prepare target for teacher forcing
            tgt_input = tgt[:, :-1]
            tgt_output = tgt[:, 1:]
            
            # Forward pass
            self.optimizer.zero_grad()
            output = self.model(src, tgt_input)
            
            # Compute loss
            loss = self.criterion(output.contiguous().view(-1, output.size(-1)), 
                                tgt_output.contiguous().view(-1))
            
            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
            
            self.optimizer.step()
            self.scheduler.step()
            
            total_loss += loss.item()
            
            if batch_idx % self.log_interval == 0:
                pbar.set_postfix({
                    'loss': f'{loss.item():.4f}',
                    'lr': f'{self.scheduler.get_last_lr()[0]:.6f}'
                })
        
        return total_loss / num_batches
    
    def validate(self) -> float:
        self.model.eval()
        total_loss = 0.0
        num_batches = len(self.val_loader)
        
        with torch.no_grad():
            for src, tgt in tqdm(self.val_loader, desc="Validation"):
                src = src.to(self.device)
                tgt = tgt.to(self.device)
                
                tgt_input = tgt[:, :-1]
                tgt_output = tgt[:, 1:]
                
                output = self.model(src, tgt_input)
                
                loss = self.criterion(output.contiguous().view(-1, output.size(-1)), 
                                    tgt_output.contiguous().view(-1))
                
                total_loss += loss.item()
        
        return total_loss / num_batches
    
    def generate_sample_outputs(self, src_vocab: Vocabulary, tgt_vocab: Vocabulary, num_samples: int = 5):
        self.model.eval()
        
        src, tgt = next(iter(self.val_loader))
        batch_size = src.size(0)
        num_samples = min(num_samples, batch_size)
        src = src[:num_samples].to(self.device)
        tgt = tgt[:num_samples].to(self.device)
        
        print("\n" + "="*50)
        print("SAMPLE TRANSLATIONS")
        print("="*50)
        
        with torch.no_grad():
            tgt_input = tgt[:, :-1]
            output = self.model(src, tgt_input)
            predictions = output.argmax(dim=-1)
            
            for i in range(num_samples):
                src_tokens = src[i].cpu().numpy()
                src_text = src_vocab.decode(src_tokens)
                
                tgt_tokens = tgt[i].cpu().numpy()
                tgt_text = tgt_vocab.decode(tgt_tokens)
                
                pred_tokens = predictions[i].cpu().numpy()
                pred_text = tgt_vocab.decode(pred_tokens)
                
                print(f"Source:     {src_text}")
                print(f"Target:     {tgt_text}")
                print(f"Prediction: {pred_text}")
                print("-" * 50)
    
    def save_checkpoint(self, epoch: int, val_loss: float, is_best: bool = False):
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.current_step,
            'val_loss': val_loss,
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'learning_rates': self.learning_rates
        }
        
        # Save best model
        if is_best:
            best_path = os.path.join(self.save_dir, 'best_model.pt')
            torch.save(checkpoint, best_path)
            print(f"New best model saved! Validation loss: {val_loss:.4f}")
        
        # Delete old checkpoints except the best one
        for filename in os.listdir(self.save_dir):
            if filename.startswith('checkpoint_epoch_') and filename.endswith('.pt'):
                filepath = os.path.join(self.save_dir, filename)
                try:
                    os.remove(filepath)
                except OSError:
                    pass  # File might not exist
    
    def plot_training_history(self):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        ax1.plot(self.train_losses, label='Training Loss')
        ax1.plot(self.val_losses, label='Validation Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.set_title('Training and Validation Loss')
        ax1.legend()
        ax1.grid(True)
        
        ax2.plot(self.learning_rates, label='Learning Rate')
        ax2.set_xlabel('Step')
        ax2.set_ylabel('Learning Rate')
        ax2.set_title('Learning Rate Schedule')
        ax2.legend()
        ax2.grid(True)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.save_dir, 'training_history.png'))
        plt.close()
    
    def train(
        self,
        num_epochs: int,
        src_vocab: Vocabulary,
        tgt_vocab: Vocabulary,
        early_stopping_patience: int = 5
    ):
        early_stopping = EarlyStopping(patience=early_stopping_patience)
        best_val_loss = float('inf')
        
        print(f"Starting training for {num_epochs} epochs...")
        print(f"Device: {self.device}")
        print(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        
        for epoch in range(1, num_epochs + 1):
            start_time = time.time()
            
            train_loss = self.train_epoch(epoch)
            val_loss = self.validate()
            current_lr = self.scheduler.get_last_lr()[0]
            
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            self.learning_rates.append(current_lr)
            
            epoch_time = time.time() - start_time
            print(f"Epoch {epoch}/{num_epochs} - "
                  f"Train Loss: {train_loss:.4f}, "
                  f"Val Loss: {val_loss:.4f}, "
                  f"LR: {current_lr:.6f}, "
                  f"Time: {epoch_time:.2f}s")
            
            is_best = val_loss < best_val_loss
            if is_best:
                best_val_loss = val_loss
            
            self.save_checkpoint(epoch, val_loss, is_best)
            
            if epoch % 5 == 0:
                self.generate_sample_outputs(src_vocab, tgt_vocab)
            
            if early_stopping(val_loss, self.model):
                print(f"Early stopping triggered after {epoch} epochs")
                early_stopping.restore_weights(self.model)
                break
        
        self.plot_training_history()
        print(f"Training completed! Best validation loss: {best_val_loss:.4f}") 