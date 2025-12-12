"""
LSTM-Autoencoder Architecture
=============================
The "HedgeFundBrain" - a multi-modal autoencoder that compresses
market state into a 3D latent space for regime detection.

Architecture from PROJ.md Section 4:
- Dual-branch encoder (Temporal LSTM + Macro Dense)
- 3D latent space (x, y, z coordinates)
- Dual-branch decoder (reconstruct both inputs)
- Auxiliary loss to force macro awareness

The key insight: By forcing reconstruction of BOTH price patterns
AND macro context, the latent space must encode meaningful
market states, not just noise.
"""

import torch
import torch.nn as nn
from typing import Tuple, Optional
from dataclasses import dataclass


@dataclass
class AutoencoderConfig:
    """
    Configuration for the HedgeFundBrain autoencoder.
    
    Version 1.2 Update: Default latent_dim changed from 3 to 8.
    PCA analysis shows 17 dimensions explain 95% variance; 3D loses
    too much signal. Use t-SNE/UMAP for visualization instead.
    """
    # Input dimensions
    temporal_features: int = 9       # Simons features per timestep
    macro_features: int = 21         # Dalio features (static)
    sequence_length: int = 30        # Temporal window
    
    # Encoder dimensions
    lstm_hidden: int = 64            # LSTM hidden size
    lstm_layers: int = 2             # LSTM depth
    macro_hidden: int = 32           # Macro encoder hidden
    
    # Latent space
    # v1.2: Changed from 3 to 8 (see PROJ.md Section 4.3)
    # 3D was too constrained (Sharpe spread 0.79 vs GMM baseline 3.50)
    latent_dim: int = 8              # 8D recommended for regime detection
    
    # Decoder dimensions
    decoder_hidden: int = 64         # Decoder hidden size
    
    # Regularization
    dropout: float = 0.2             # Dropout rate
    
    def __repr__(self) -> str:
        return (
            f"AutoencoderConfig(\n"
            f"  temporal: [{self.sequence_length}, {self.temporal_features}]\n"
            f"  macro: [{self.macro_features}]\n"
            f"  latent: {self.latent_dim}D\n"
            f"  lstm: hidden={self.lstm_hidden}, layers={self.lstm_layers}\n"
            f")"
        )


# Preset configurations for different experiments
CONFIG_PRESETS = {
    "3d_visualization": AutoencoderConfig(latent_dim=3),  # Original, for viz only
    "8d_recommended": AutoencoderConfig(latent_dim=8),    # Recommended default
    "16d_high_capacity": AutoencoderConfig(latent_dim=16, lstm_hidden=128),
}


class TemporalEncoder(nn.Module):
    """
    LSTM-based encoder for temporal (price) sequences.
    
    Processes 30-day windows of price features to capture:
    - Momentum patterns
    - Volatility clustering
    - Path dependency
    
    Input: [batch, seq_len, features] = [B, 30, 9]
    Output: [batch, hidden] = [B, 64]
    """
    
    def __init__(self, config: AutoencoderConfig):
        super().__init__()
        self.config = config
        
        self.lstm = nn.LSTM(
            input_size=config.temporal_features,
            hidden_size=config.lstm_hidden,
            num_layers=config.lstm_layers,
            batch_first=True,
            dropout=config.dropout if config.lstm_layers > 1 else 0,
            bidirectional=False,
        )
        
        # Layer norm for stability
        self.layer_norm = nn.LayerNorm(config.lstm_hidden)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : torch.Tensor
            Shape: [batch, seq_len, temporal_features]
            
        Returns
        -------
        torch.Tensor
            Shape: [batch, lstm_hidden]
        """
        # LSTM forward
        # output: [batch, seq_len, hidden]
        # (h_n, c_n): [layers, batch, hidden]
        output, (h_n, c_n) = self.lstm(x)
        
        # Take the last hidden state from the last layer
        # h_n[-1] is [batch, hidden]
        encoded = h_n[-1]
        
        # Normalize
        encoded = self.layer_norm(encoded)
        
        return encoded


class MacroEncoder(nn.Module):
    """
    Dense encoder for macro (economic) features.
    
    Processes static macro snapshot to capture:
    - Interest rate environment
    - Dollar strength
    - Risk-on/risk-off context
    
    Input: [batch, macro_features] = [B, 21]
    Output: [batch, macro_hidden] = [B, 32]
    """
    
    def __init__(self, config: AutoencoderConfig):
        super().__init__()
        self.config = config
        
        self.encoder = nn.Sequential(
            nn.Linear(config.macro_features, config.macro_hidden * 2),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.macro_hidden * 2, config.macro_hidden),
            nn.ReLU(),
            nn.LayerNorm(config.macro_hidden),
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : torch.Tensor
            Shape: [batch, macro_features]
            
        Returns
        -------
        torch.Tensor
            Shape: [batch, macro_hidden]
        """
        return self.encoder(x)


class FusionLayer(nn.Module):
    """
    Fuses temporal and macro encodings into latent space.
    
    This is where the "magic" happens - the network must learn
    to combine price patterns with economic context into a
    compressed 3D representation.
    
    Input: temporal [B, 64] + macro [B, 32] = [B, 96]
    Output: latent [B, 3]
    """
    
    def __init__(self, config: AutoencoderConfig):
        super().__init__()
        self.config = config
        
        fusion_input = config.lstm_hidden + config.macro_hidden
        
        self.fusion = nn.Sequential(
            nn.Linear(fusion_input, fusion_input // 2),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(fusion_input // 2, config.latent_dim),
            # No activation on latent - allow full range
        )
    
    def forward(
        self, 
        temporal_encoding: torch.Tensor, 
        macro_encoding: torch.Tensor
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        temporal_encoding : torch.Tensor
            Shape: [batch, lstm_hidden]
        macro_encoding : torch.Tensor
            Shape: [batch, macro_hidden]
            
        Returns
        -------
        torch.Tensor
            Latent coordinates. Shape: [batch, latent_dim]
        """
        # Concatenate encodings
        combined = torch.cat([temporal_encoding, macro_encoding], dim=1)
        
        # Project to latent space
        latent = self.fusion(combined)
        
        return latent


class TemporalDecoder(nn.Module):
    """
    LSTM-based decoder to reconstruct temporal sequences.
    
    Reconstructs the 30-day price sequence from the latent code.
    Forces the latent space to preserve temporal dynamics.
    
    Input: [batch, latent_dim] = [B, 3]
    Output: [batch, seq_len, temporal_features] = [B, 30, 9]
    """
    
    def __init__(self, config: AutoencoderConfig):
        super().__init__()
        self.config = config
        
        # Expand latent to LSTM input
        self.latent_to_hidden = nn.Sequential(
            nn.Linear(config.latent_dim, config.decoder_hidden),
            nn.ReLU(),
        )
        
        # LSTM decoder
        self.lstm = nn.LSTM(
            input_size=config.decoder_hidden,
            hidden_size=config.decoder_hidden,
            num_layers=1,
            batch_first=True,
        )
        
        # Output projection
        self.output_proj = nn.Linear(config.decoder_hidden, config.temporal_features)
    
    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        latent : torch.Tensor
            Shape: [batch, latent_dim]
            
        Returns
        -------
        torch.Tensor
            Reconstructed sequence. Shape: [batch, seq_len, temporal_features]
        """
        batch_size = latent.size(0)
        seq_len = self.config.sequence_length
        
        # Expand latent to initial hidden
        hidden = self.latent_to_hidden(latent)
        
        # Repeat across sequence length
        # [batch, hidden] -> [batch, seq_len, hidden]
        decoder_input = hidden.unsqueeze(1).repeat(1, seq_len, 1)
        
        # LSTM decode
        output, _ = self.lstm(decoder_input)
        
        # Project to feature space
        reconstructed = self.output_proj(output)
        
        return reconstructed


class MacroDecoder(nn.Module):
    """
    Dense decoder to reconstruct macro features.
    
    Forces the latent space to preserve economic context.
    The lambda weighting in the loss makes this critical.
    
    Input: [batch, latent_dim] = [B, 3]
    Output: [batch, macro_features] = [B, 21]
    """
    
    def __init__(self, config: AutoencoderConfig):
        super().__init__()
        self.config = config
        
        self.decoder = nn.Sequential(
            nn.Linear(config.latent_dim, config.decoder_hidden),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.decoder_hidden, config.macro_features),
            # No activation - features are normalized
        )
    
    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        latent : torch.Tensor
            Shape: [batch, latent_dim]
            
        Returns
        -------
        torch.Tensor
            Reconstructed macro. Shape: [batch, macro_features]
        """
        return self.decoder(latent)


class HedgeFundBrain(nn.Module):
    """
    Multi-Modal LSTM-Autoencoder for Market Regime Detection.
    
    The "brain" of the Simons-Dalio Regime Engine. Compresses
    complex market state (price patterns + macro context) into
    a 3D latent space where regimes can be visualized and clustered.
    
    Architecture:
    ```
    X_temporal [B,30,9] ──► TemporalEncoder ──┐
                                              ├──► FusionLayer ──► Latent [B,3]
    X_macro [B,21] ────────► MacroEncoder ────┘            │
                                                           ├──► TemporalDecoder ──► X_temporal_recon
                                                           └──► MacroDecoder ──────► X_macro_recon
    ```
    
    Loss Function:
        L_total = L_temporal_recon + lambda_macro * L_macro_recon
    
    Parameters
    ----------
    config : AutoencoderConfig
        Model configuration.
        
    Example
    -------
    >>> config = AutoencoderConfig(temporal_features=9, macro_features=21)
    >>> model = HedgeFundBrain(config)
    >>> 
    >>> # Forward pass
    >>> x_temp = torch.randn(32, 30, 9)  # [batch, seq, features]
    >>> x_macro = torch.randn(32, 21)    # [batch, features]
    >>> latent, temp_recon, macro_recon = model(x_temp, x_macro)
    >>> 
    >>> # Encode only (for inference)
    >>> latent = model.encode(x_temp, x_macro)
    """
    
    def __init__(self, config: Optional[AutoencoderConfig] = None):
        super().__init__()
        
        if config is None:
            config = AutoencoderConfig()
        
        self.config = config
        
        # Encoders
        self.temporal_encoder = TemporalEncoder(config)
        self.macro_encoder = MacroEncoder(config)
        
        # Fusion
        self.fusion = FusionLayer(config)
        
        # Decoders
        self.temporal_decoder = TemporalDecoder(config)
        self.macro_decoder = MacroDecoder(config)
    
    def encode(
        self, 
        x_temporal: torch.Tensor, 
        x_macro: torch.Tensor
    ) -> torch.Tensor:
        """
        Encode inputs to latent space (inference mode).
        
        Parameters
        ----------
        x_temporal : torch.Tensor
            Temporal features. Shape: [batch, seq_len, temporal_features]
        x_macro : torch.Tensor
            Macro features. Shape: [batch, macro_features]
            
        Returns
        -------
        torch.Tensor
            Latent coordinates. Shape: [batch, latent_dim]
        """
        # Encode both modalities
        temporal_enc = self.temporal_encoder(x_temporal)
        macro_enc = self.macro_encoder(x_macro)
        
        # Fuse to latent
        latent = self.fusion(temporal_enc, macro_enc)
        
        return latent
    
    def decode(
        self, 
        latent: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Decode latent to reconstructions.
        
        Parameters
        ----------
        latent : torch.Tensor
            Latent coordinates. Shape: [batch, latent_dim]
            
        Returns
        -------
        x_temporal_recon : torch.Tensor
            Shape: [batch, seq_len, temporal_features]
        x_macro_recon : torch.Tensor
            Shape: [batch, macro_features]
        """
        x_temporal_recon = self.temporal_decoder(latent)
        x_macro_recon = self.macro_decoder(latent)
        
        return x_temporal_recon, x_macro_recon
    
    def forward(
        self, 
        x_temporal: torch.Tensor, 
        x_macro: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Full forward pass: encode then decode.
        
        Parameters
        ----------
        x_temporal : torch.Tensor
            Temporal features. Shape: [batch, seq_len, temporal_features]
        x_macro : torch.Tensor
            Macro features. Shape: [batch, macro_features]
            
        Returns
        -------
        latent : torch.Tensor
            Latent coordinates. Shape: [batch, latent_dim]
        x_temporal_recon : torch.Tensor
            Reconstructed temporal. Shape: [batch, seq_len, temporal_features]
        x_macro_recon : torch.Tensor
            Reconstructed macro. Shape: [batch, macro_features]
        """
        # Encode
        latent = self.encode(x_temporal, x_macro)
        
        # Decode
        x_temporal_recon, x_macro_recon = self.decode(latent)
        
        return latent, x_temporal_recon, x_macro_recon
    
    def count_parameters(self) -> int:
        """Count total trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def __repr__(self) -> str:
        return (
            f"HedgeFundBrain(\n"
            f"  {self.config}\n"
            f"  parameters: {self.count_parameters():,}\n"
            f")"
        )


class AutoencoderLoss(nn.Module):
    """
    Combined loss for the autoencoder.
    
    L_total = L_temporal + lambda_macro * L_macro
    
    The lambda_macro weight forces the network to pay attention
    to macro features even though they're slower-moving and
    "easier" to reconstruct.
    
    Parameters
    ----------
    lambda_macro : float, default=2.0
        Weight for macro reconstruction loss.
    reduction : str, default='mean'
        Loss reduction method.
    """
    
    def __init__(self, lambda_macro: float = 2.0, reduction: str = 'mean'):
        super().__init__()
        self.lambda_macro = lambda_macro
        self.mse = nn.MSELoss(reduction=reduction)
    
    def forward(
        self,
        x_temporal: torch.Tensor,
        x_macro: torch.Tensor,
        x_temporal_recon: torch.Tensor,
        x_macro_recon: torch.Tensor,
    ) -> Tuple[torch.Tensor, dict]:
        """
        Compute combined reconstruction loss.
        
        Returns
        -------
        loss : torch.Tensor
            Total loss.
        components : dict
            Individual loss components for logging.
        """
        loss_temporal = self.mse(x_temporal_recon, x_temporal)
        loss_macro = self.mse(x_macro_recon, x_macro)
        
        loss_total = loss_temporal + self.lambda_macro * loss_macro
        
        components = {
            'loss_total': loss_total.item(),
            'loss_temporal': loss_temporal.item(),
            'loss_macro': loss_macro.item(),
        }
        
        return loss_total, components


if __name__ == "__main__":
    # Quick architecture test
    print("=" * 60)
    print("HedgeFundBrain Architecture Test")
    print("=" * 60)
    print()
    
    # Create model
    config = AutoencoderConfig(
        temporal_features=9,
        macro_features=21,
        sequence_length=30,
    )
    model = HedgeFundBrain(config)
    print(model)
    print()
    
    # Test forward pass
    batch_size = 32
    x_temporal = torch.randn(batch_size, 30, 9)
    x_macro = torch.randn(batch_size, 21)
    
    print("Input shapes:")
    print(f"  x_temporal: {x_temporal.shape}")
    print(f"  x_macro:    {x_macro.shape}")
    print()
    
    # Forward pass
    model.eval()
    with torch.no_grad():
        latent, temp_recon, macro_recon = model(x_temporal, x_macro)
    
    print("Output shapes:")
    print(f"  latent:       {latent.shape}")
    print(f"  temp_recon:   {temp_recon.shape}")
    print(f"  macro_recon:  {macro_recon.shape}")
    print()
    
    # Test loss
    loss_fn = AutoencoderLoss(lambda_macro=2.0)
    loss, components = loss_fn(x_temporal, x_macro, temp_recon, macro_recon)
    
    print("Loss components:")
    for name, value in components.items():
        print(f"  {name}: {value:.4f}")
    print()
    
    # Verify shapes
    assert latent.shape == (batch_size, 3), f"Latent shape wrong: {latent.shape}"
    assert temp_recon.shape == x_temporal.shape, f"Temporal recon shape wrong"
    assert macro_recon.shape == x_macro.shape, f"Macro recon shape wrong"
    
    print("✓ All shape checks passed!")

