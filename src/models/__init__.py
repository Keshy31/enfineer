"""
Models Package
==============
Neural network architectures for the Simons-Dalio Regime Engine.

Main Components:
- HedgeFundBrain: Multi-modal LSTM-Autoencoder for regime detection
- WalkForwardTrainer: Training with proper OOS validation
"""

from .autoencoder import (
    HedgeFundBrain,
    AutoencoderConfig,
    AutoencoderLoss,
    TemporalEncoder,
    MacroEncoder,
    TemporalDecoder,
    MacroDecoder,
)
from .trainer import (
    AutoencoderTrainer,
    TrainingConfig,
    TrainingHistory,
    create_data_loaders,
)

__all__ = [
    # Model
    "HedgeFundBrain",
    "AutoencoderConfig",
    "AutoencoderLoss",
    "TemporalEncoder",
    "MacroEncoder",
    "TemporalDecoder",
    "MacroDecoder",
    # Training
    "AutoencoderTrainer",
    "TrainingConfig",
    "TrainingHistory",
    "create_data_loaders",
]

