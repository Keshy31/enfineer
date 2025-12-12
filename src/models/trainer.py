"""
Walk-Forward Autoencoder Trainer
================================
Trains the HedgeFundBrain with proper out-of-sample validation.

Key principles:
1. Train on expanding window of historical data
2. Validate on unseen future data
3. Never leak future information into training
4. Save best model based on validation loss

The walk-forward approach ensures the latent space learned
by the autoencoder generalizes to unseen market conditions.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import pandas as pd
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional, List, Tuple, Dict
import time
import json

from .autoencoder import HedgeFundBrain, AutoencoderConfig, AutoencoderLoss


@dataclass
class TrainingConfig:
    """Configuration for autoencoder training."""
    # Training hyperparameters
    learning_rate: float = 1e-3
    batch_size: int = 32
    epochs: int = 100
    early_stopping_patience: int = 15
    
    # Loss weights
    lambda_macro: float = 2.0
    
    # Optimizer
    weight_decay: float = 1e-5
    
    # Learning rate scheduling
    lr_scheduler: str = 'plateau'  # 'plateau', 'cosine', or 'none'
    lr_patience: int = 5
    lr_factor: float = 0.5
    
    # Checkpointing
    save_dir: Path = field(default_factory=lambda: Path("./checkpoints"))
    save_best_only: bool = True
    
    # Device
    device: str = 'auto'  # 'auto', 'cuda', 'cpu'
    
    def get_device(self) -> torch.device:
        if self.device == 'auto':
            return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        return torch.device(self.device)


@dataclass
class TrainingHistory:
    """Training history for one fold."""
    fold_num: int
    train_losses: List[float] = field(default_factory=list)
    val_losses: List[float] = field(default_factory=list)
    train_temporal_losses: List[float] = field(default_factory=list)
    train_macro_losses: List[float] = field(default_factory=list)
    val_temporal_losses: List[float] = field(default_factory=list)
    val_macro_losses: List[float] = field(default_factory=list)
    best_epoch: int = 0
    best_val_loss: float = float('inf')
    training_time: float = 0.0


class AutoencoderTrainer:
    """
    Trainer for the HedgeFundBrain autoencoder.
    
    Implements:
    - Standard training loop with validation
    - Early stopping
    - Learning rate scheduling
    - Model checkpointing
    - Training history logging
    
    Parameters
    ----------
    model : HedgeFundBrain
        The autoencoder model.
    config : TrainingConfig
        Training configuration.
        
    Example
    -------
    >>> model = HedgeFundBrain(AutoencoderConfig())
    >>> trainer = AutoencoderTrainer(model, TrainingConfig())
    >>> history = trainer.fit(train_loader, val_loader)
    """
    
    def __init__(
        self, 
        model: HedgeFundBrain, 
        config: Optional[TrainingConfig] = None
    ):
        if config is None:
            config = TrainingConfig()
        
        self.model = model
        self.config = config
        self.device = config.get_device()
        
        # Move model to device
        self.model.to(self.device)
        
        # Loss function
        self.loss_fn = AutoencoderLoss(lambda_macro=config.lambda_macro)
        
        # Optimizer
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )
        
        # Learning rate scheduler
        self.scheduler = self._create_scheduler()
        
        # Create save directory
        config.save_dir.mkdir(parents=True, exist_ok=True)
    
    def _create_scheduler(self):
        """Create learning rate scheduler based on config."""
        if self.config.lr_scheduler == 'plateau':
            return torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode='min',
                factor=self.config.lr_factor,
                patience=self.config.lr_patience,
                verbose=True,
            )
        elif self.config.lr_scheduler == 'cosine':
            return torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=self.config.epochs,
            )
        else:
            return None
    
    def _train_epoch(self, train_loader: DataLoader) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()
        
        total_loss = 0.0
        total_temporal = 0.0
        total_macro = 0.0
        n_batches = 0
        
        for x_temporal, x_macro in train_loader:
            x_temporal = x_temporal.to(self.device)
            x_macro = x_macro.to(self.device)
            
            # Forward pass
            latent, temp_recon, macro_recon = self.model(x_temporal, x_macro)
            
            # Compute loss
            loss, components = self.loss_fn(
                x_temporal, x_macro, temp_recon, macro_recon
            )
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            self.optimizer.step()
            
            total_loss += components['loss_total']
            total_temporal += components['loss_temporal']
            total_macro += components['loss_macro']
            n_batches += 1
        
        return {
            'loss': total_loss / n_batches,
            'loss_temporal': total_temporal / n_batches,
            'loss_macro': total_macro / n_batches,
        }
    
    @torch.no_grad()
    def _validate(self, val_loader: DataLoader) -> Dict[str, float]:
        """Validate on held-out data."""
        self.model.eval()
        
        total_loss = 0.0
        total_temporal = 0.0
        total_macro = 0.0
        n_batches = 0
        
        for x_temporal, x_macro in val_loader:
            x_temporal = x_temporal.to(self.device)
            x_macro = x_macro.to(self.device)
            
            # Forward pass
            latent, temp_recon, macro_recon = self.model(x_temporal, x_macro)
            
            # Compute loss
            loss, components = self.loss_fn(
                x_temporal, x_macro, temp_recon, macro_recon
            )
            
            total_loss += components['loss_total']
            total_temporal += components['loss_temporal']
            total_macro += components['loss_macro']
            n_batches += 1
        
        return {
            'loss': total_loss / n_batches,
            'loss_temporal': total_temporal / n_batches,
            'loss_macro': total_macro / n_batches,
        }
    
    def fit(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        fold_num: int = 1,
        verbose: bool = True,
    ) -> TrainingHistory:
        """
        Train the model.
        
        Parameters
        ----------
        train_loader : DataLoader
            Training data.
        val_loader : DataLoader
            Validation data.
        fold_num : int
            Fold number (for logging).
        verbose : bool
            Print progress.
            
        Returns
        -------
        TrainingHistory
            Training history.
        """
        history = TrainingHistory(fold_num=fold_num)
        best_model_state = None
        patience_counter = 0
        
        start_time = time.time()
        
        if verbose:
            print(f"Training on {self.device} for {self.config.epochs} epochs...")
            print()
        
        for epoch in range(1, self.config.epochs + 1):
            # Train
            train_metrics = self._train_epoch(train_loader)
            
            # Validate
            val_metrics = self._validate(val_loader)
            
            # Record history
            history.train_losses.append(train_metrics['loss'])
            history.val_losses.append(val_metrics['loss'])
            history.train_temporal_losses.append(train_metrics['loss_temporal'])
            history.train_macro_losses.append(train_metrics['loss_macro'])
            history.val_temporal_losses.append(val_metrics['loss_temporal'])
            history.val_macro_losses.append(val_metrics['loss_macro'])
            
            # Learning rate scheduling
            if self.scheduler is not None:
                if self.config.lr_scheduler == 'plateau':
                    self.scheduler.step(val_metrics['loss'])
                else:
                    self.scheduler.step()
            
            # Check for improvement
            if val_metrics['loss'] < history.best_val_loss:
                history.best_val_loss = val_metrics['loss']
                history.best_epoch = epoch
                patience_counter = 0
                
                # Save best model state
                best_model_state = {
                    k: v.cpu().clone() for k, v in self.model.state_dict().items()
                }
                
                if verbose:
                    print(f"  Epoch {epoch:3d}: train={train_metrics['loss']:.4f}, "
                          f"val={val_metrics['loss']:.4f} ★ (best)")
            else:
                patience_counter += 1
                if verbose and epoch % 10 == 0:
                    print(f"  Epoch {epoch:3d}: train={train_metrics['loss']:.4f}, "
                          f"val={val_metrics['loss']:.4f}")
            
            # Early stopping
            if patience_counter >= self.config.early_stopping_patience:
                if verbose:
                    print(f"\n  Early stopping at epoch {epoch}")
                break
        
        # Restore best model
        if best_model_state is not None:
            self.model.load_state_dict(best_model_state)
        
        history.training_time = time.time() - start_time
        
        if verbose:
            print(f"\n  Best epoch: {history.best_epoch}, "
                  f"Best val loss: {history.best_val_loss:.4f}, "
                  f"Time: {history.training_time:.1f}s")
        
        return history
    
    def save_checkpoint(self, path: Path, history: Optional[TrainingHistory] = None):
        """Save model checkpoint."""
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'config': self.model.config.__dict__,
            'training_config': self.config.__dict__,
        }
        if history is not None:
            checkpoint['history'] = {
                'train_losses': history.train_losses,
                'val_losses': history.val_losses,
                'best_epoch': history.best_epoch,
                'best_val_loss': history.best_val_loss,
            }
        
        torch.save(checkpoint, path)
    
    def load_checkpoint(self, path: Path):
        """Load model checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    @torch.no_grad()
    def encode_dataset(
        self,
        data_loader: DataLoader,
    ) -> np.ndarray:
        """
        Encode entire dataset to latent space.
        
        Parameters
        ----------
        data_loader : DataLoader
            Data to encode.
            
        Returns
        -------
        np.ndarray
            Latent coordinates. Shape: [n_samples, latent_dim]
        """
        self.model.eval()
        
        latents = []
        for x_temporal, x_macro in data_loader:
            x_temporal = x_temporal.to(self.device)
            x_macro = x_macro.to(self.device)
            
            latent = self.model.encode(x_temporal, x_macro)
            latents.append(latent.cpu().numpy())
        
        return np.vstack(latents)


def create_data_loaders(
    X_temporal: np.ndarray,
    X_macro: np.ndarray,
    batch_size: int = 32,
    shuffle: bool = True,
) -> DataLoader:
    """
    Create PyTorch DataLoader from numpy arrays.
    
    Parameters
    ----------
    X_temporal : np.ndarray
        Shape: [n_samples, seq_len, features]
    X_macro : np.ndarray
        Shape: [n_samples, features]
    batch_size : int
        Batch size.
    shuffle : bool
        Whether to shuffle.
        
    Returns
    -------
    DataLoader
    """
    tensor_temporal = torch.FloatTensor(X_temporal)
    tensor_macro = torch.FloatTensor(X_macro)
    
    dataset = TensorDataset(tensor_temporal, tensor_macro)
    
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,  # Windows compatibility
        pin_memory=True,
    )


if __name__ == "__main__":
    # Quick training test with random data
    print("=" * 60)
    print("AutoencoderTrainer Test")
    print("=" * 60)
    print()
    
    # Create model
    config = AutoencoderConfig(temporal_features=9, macro_features=21)
    model = HedgeFundBrain(config)
    
    # Create random training data
    n_train = 500
    n_val = 100
    
    X_temp_train = np.random.randn(n_train, 30, 9).astype(np.float32)
    X_macro_train = np.random.randn(n_train, 21).astype(np.float32)
    X_temp_val = np.random.randn(n_val, 30, 9).astype(np.float32)
    X_macro_val = np.random.randn(n_val, 21).astype(np.float32)
    
    # Create data loaders
    train_loader = create_data_loaders(X_temp_train, X_macro_train, batch_size=32)
    val_loader = create_data_loaders(X_temp_val, X_macro_val, batch_size=32, shuffle=False)
    
    print(f"Train samples: {n_train}")
    print(f"Val samples: {n_val}")
    print()
    
    # Train
    train_config = TrainingConfig(
        epochs=30,
        early_stopping_patience=10,
        learning_rate=1e-3,
    )
    trainer = AutoencoderTrainer(model, train_config)
    
    history = trainer.fit(train_loader, val_loader, fold_num=1)
    
    print()
    print("=" * 60)
    print("Training Complete!")
    print("=" * 60)
    print()
    
    # Test encoding
    latents = trainer.encode_dataset(val_loader)
    print(f"Encoded latent shape: {latents.shape}")
    print(f"Latent mean: {latents.mean(axis=0)}")
    print(f"Latent std: {latents.std(axis=0)}")

