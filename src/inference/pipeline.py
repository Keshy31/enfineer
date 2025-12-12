"""
Inference Pipeline
==================
End-to-end inference from raw data to trading signal.

This is the main class that orchestrates:
1. Loading the trained model
2. Fetching and validating data
3. Computing features
4. Running inference
5. Generating signals
6. Logging predictions
"""

import json
import numpy as np
import pandas as pd
import torch
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import pickle

from .data_validator import DataValidator, DataQualityReport
from .signal_mapper import SignalMapper, SignalResult, TradingSignal


@dataclass
class InferenceResult:
    """
    Complete result from inference pipeline.
    
    Contains everything needed for trading decision and logging.
    """
    # Timestamps
    inference_time: datetime
    data_as_of: datetime
    
    # Data quality
    data_quality: DataQualityReport
    
    # Model outputs
    latent_vector: np.ndarray
    regime: int
    regime_probabilities: np.ndarray
    
    # Trading signal
    signal: SignalResult
    
    # Feature snapshot (for debugging/analysis)
    features_snapshot: Dict[str, float] = field(default_factory=dict)
    
    # Model metadata
    model_id: str = ""
    model_sharpe_spread: float = 0.0
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON logging."""
        return {
            "inference_time": self.inference_time.isoformat(),
            "data_as_of": self.data_as_of.isoformat(),
            "data_quality": {
                "score": float(self.data_quality.overall_score),
                "can_proceed": bool(self.data_quality.can_proceed),
                "warnings": self.data_quality.warnings,
                "errors": self.data_quality.errors,
                "sources": {
                    k: {
                        "status": v.status.value,
                        "age_hours": float(v.age_hours) if v.age_hours else None,
                        "message": v.message,
                    }
                    for k, v in self.data_quality.sources.items()
                }
            },
            "latent_vector": self.latent_vector.tolist(),
            "regime": int(self.regime),
            "regime_probabilities": self.regime_probabilities.tolist(),
            "signal": self.signal.to_dict(),
            "features_snapshot": {k: float(v) if v is not None else None 
                                  for k, v in self.features_snapshot.items()},
            "model_id": self.model_id,
            "model_sharpe_spread": float(self.model_sharpe_spread),
        }
    
    def to_json(self) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict(), indent=2)


class InferencePipeline:
    """
    Production inference pipeline for regime detection.
    
    Handles the complete flow from data fetching to signal generation,
    with proper error handling and logging.
    
    Parameters
    ----------
    model_path : Path or str
        Path to saved model checkpoint directory.
    log_dir : Path or str, optional
        Directory for prediction logs. Defaults to ./logs/predictions/
        
    Example
    -------
    >>> pipeline = InferencePipeline("./checkpoints/best_model")
    >>> result = pipeline.run()
    >>> print(f"Signal: {result.signal.signal.value}")
    >>> print(f"Confidence: {result.signal.confidence.value}")
    """
    
    def __init__(
        self,
        model_path: Optional[Path] = None,
        log_dir: Optional[Path] = None,
    ):
        self.model_path = Path(model_path) if model_path else None
        self.log_dir = Path(log_dir) if log_dir else Path("./logs/predictions")
        
        # Components (lazy loaded)
        self.model = None
        self.model_config = None
        self.norm_stats = None
        self.gmm = None
        self.metadata = None
        
        # Validators and mappers
        self.data_validator = DataValidator()
        self.signal_mapper = SignalMapper()
        
        # Data manager (lazy loaded)
        self._data_manager = None
        
        # Load model if path provided
        if self.model_path:
            self.load_model(self.model_path)
    
    @property
    def data_manager(self):
        """Lazy load DataManager to avoid import issues."""
        if self._data_manager is None:
            from data import DataManager
            self._data_manager = DataManager("./data")
        return self._data_manager
    
    def load_model(self, model_path: Path) -> None:
        """
        Load a saved model package.
        
        Parameters
        ----------
        model_path : Path
            Directory containing model.pt, norm_stats.pkl, gmm.pkl, metadata.json
        """
        model_path = Path(model_path)
        
        if not model_path.exists():
            raise FileNotFoundError(f"Model path not found: {model_path}")
        
        # Load metadata
        metadata_path = model_path / "metadata.json"
        if metadata_path.exists():
            with open(metadata_path, 'r') as f:
                self.metadata = json.load(f)
        else:
            self.metadata = {}
        
        # Load model
        from models.autoencoder import HedgeFundBrain, AutoencoderConfig
        
        checkpoint_path = model_path / "model.pt"
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        
        self.model_config = AutoencoderConfig(**checkpoint['model_config'])
        self.model = HedgeFundBrain(self.model_config)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
        
        # Load normalization stats
        norm_path = model_path / "norm_stats.pkl"
        with open(norm_path, 'rb') as f:
            self.norm_stats = pickle.load(f)
        
        # Load GMM
        gmm_path = model_path / "gmm.pkl"
        with open(gmm_path, 'rb') as f:
            self.gmm = pickle.load(f)
        
        # Update signal mapper with model's regime stats
        if 'performance' in self.metadata and 'regime_stats' in self.metadata['performance']:
            self.signal_mapper.update_profiles_from_metadata(
                self.metadata['performance']['regime_stats']
            )
        
        print(f"Loaded model from {model_path}")
        print(f"  Latent dim: {self.model_config.latent_dim}")
        print(f"  Sharpe spread: {self.metadata.get('performance', {}).get('sharpe_spread', 'N/A')}")
    
    def fetch_data(self, lookback_days: int = 60) -> Dict[str, pd.DataFrame]:
        """
        Fetch latest data for all required symbols.
        
        Parameters
        ----------
        lookback_days : int
            Number of days of history to fetch.
            
        Returns
        -------
        Dict[str, pd.DataFrame]
            Dictionary mapping symbol to OHLCV DataFrame.
        """
        end_date = datetime.now()
        start_date = end_date - timedelta(days=lookback_days)
        
        symbols = ["BTC-USD", "^TNX", "DX-Y.NYB", "GLD"]
        data = {}
        
        for symbol in symbols:
            try:
                df = self.data_manager.get_ohlcv(
                    symbol,
                    timeframe="1d",
                    start=start_date.strftime("%Y-%m-%d"),
                    end=end_date.strftime("%Y-%m-%d"),
                )
                data[symbol] = df
            except Exception as e:
                print(f"  Warning: Failed to fetch {symbol}: {e}")
                data[symbol] = None
        
        return data
    
    def compute_features(self, data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """
        Compute features from raw data.
        
        Uses the same feature pipeline as training.
        """
        from data.alignment import align_to_crypto
        from data.dalio_features import compute_combined_features
        
        # Prepare macro data dict
        macro_dfs = {}
        if data.get("^TNX") is not None and len(data.get("^TNX", [])) > 0:
            macro_dfs["TNX"] = data["^TNX"]
        if data.get("DX-Y.NYB") is not None and len(data.get("DX-Y.NYB", [])) > 0:
            macro_dfs["DXY"] = data["DX-Y.NYB"]
        if data.get("GLD") is not None and len(data.get("GLD", [])) > 0:
            macro_dfs["GLD"] = data["GLD"]
        
        # Align all data to BTC dates
        aligned = align_to_crypto(
            crypto_df=data["BTC-USD"],
            macro_dfs=macro_dfs,
        )
        
        # Compute combined features
        features = compute_combined_features(aligned, window=30)
        
        return features
    
    def prepare_input(self, features: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepare model input from features.
        
        Returns temporal sequences and macro features, normalized
        using saved statistics.
        """
        from data.training import get_temporal_feature_cols, get_macro_feature_cols
        from data.walk_forward import get_stationary_features
        
        # Get feature columns
        temporal_cols = get_temporal_feature_cols()
        macro_cols = [c for c in get_macro_feature_cols() if c in features.columns]
        stationary = get_stationary_features()
        macro_cols = [c for c in macro_cols if c in stationary]
        
        # Get last 30 days for temporal input
        seq_len = self.model_config.sequence_length
        if len(features) < seq_len:
            raise ValueError(f"Need at least {seq_len} days, have {len(features)}")
        
        # Extract temporal sequence
        temporal_data = features[temporal_cols].iloc[-seq_len:].values
        temporal_data = temporal_data.reshape(1, seq_len, len(temporal_cols))
        
        # Extract macro features (last day)
        macro_data = features[macro_cols].iloc[-1:].values
        
        # Normalize using saved stats
        temp_mean = self.norm_stats['temp_mean']
        temp_std = self.norm_stats['temp_std']
        macro_mean = self.norm_stats['macro_mean']
        macro_std = self.norm_stats['macro_std']
        
        temporal_norm = (temporal_data - temp_mean) / temp_std
        macro_norm = (macro_data - macro_mean) / macro_std
        
        return temporal_norm.astype(np.float32), macro_norm.astype(np.float32)
    
    def predict(
        self,
        temporal: np.ndarray,
        macro: np.ndarray,
    ) -> Tuple[np.ndarray, int, np.ndarray]:
        """
        Run model inference.
        
        Returns
        -------
        latent : np.ndarray
            8D latent vector.
        regime : int
            Predicted regime label.
        probabilities : np.ndarray
            GMM probability for each regime.
        """
        # Convert to tensors
        temporal_t = torch.from_numpy(temporal)
        macro_t = torch.from_numpy(macro)
        
        # Encode to latent space
        with torch.no_grad():
            latent = self.model.encode(temporal_t, macro_t)
            latent = latent.numpy()
        
        # Classify with GMM
        regime = self.gmm.predict(latent)[0]
        probabilities = self.gmm.predict_proba(latent)[0]
        
        return latent[0], regime, probabilities
    
    def run(
        self,
        dry_run: bool = False,
    ) -> InferenceResult:
        """
        Run complete inference pipeline.
        
        Parameters
        ----------
        dry_run : bool
            If True, don't log prediction.
            
        Returns
        -------
        InferenceResult
            Complete inference result with signal.
        """
        inference_time = datetime.now()
        
        if self.model is None:
            raise RuntimeError("No model loaded. Call load_model() first.")
        
        # Step 1: Fetch data
        print("Fetching latest data...")
        data = self.fetch_data(lookback_days=60)
        
        # Step 2: Validate data
        print("Validating data quality...")
        quality_report = self.data_validator.validate(data)
        
        if not quality_report.can_proceed:
            raise RuntimeError(
                f"Data quality too low to proceed: {quality_report.errors}"
            )
        
        # Step 3: Compute features
        print("Computing features...")
        features = self.compute_features(data)
        
        # Get data timestamp
        data_as_of = features.index[-1].to_pydatetime()
        if data_as_of.tzinfo is not None:
            data_as_of = data_as_of.replace(tzinfo=None)
        
        # Step 4: Prepare input
        temporal, macro = self.prepare_input(features)
        
        # Step 5: Run inference
        print("Running inference...")
        latent, regime, probabilities = self.predict(temporal, macro)
        
        # Step 6: Generate signal
        gmm_proba = probabilities[regime]
        signal = self.signal_mapper.get_signal(regime, gmm_proba)
        
        # Step 7: Build result
        # Get feature snapshot (last row of key features)
        feature_snapshot = {}
        key_features = ['log_return', 'volatility', 'momentum', 'risk_on_score']
        for col in key_features:
            if col in features.columns:
                val = features[col].iloc[-1]
                feature_snapshot[col] = float(val) if not pd.isna(val) else None
        
        result = InferenceResult(
            inference_time=inference_time,
            data_as_of=data_as_of,
            data_quality=quality_report,
            latent_vector=latent,
            regime=regime,
            regime_probabilities=probabilities,
            signal=signal,
            features_snapshot=feature_snapshot,
            model_id=self.metadata.get('run_id', 'unknown'),
            model_sharpe_spread=self.metadata.get('performance', {}).get('sharpe_spread', 0.0),
        )
        
        # Step 8: Log prediction
        if not dry_run:
            self.log_prediction(result)
        
        return result
    
    def log_prediction(self, result: InferenceResult) -> Path:
        """
        Log prediction to daily JSON file.
        
        Returns path to log file.
        """
        # Ensure log directory exists
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Create daily log file
        date_str = result.inference_time.strftime("%Y-%m-%d")
        log_path = self.log_dir / f"{date_str}.json"
        
        # Load existing predictions for today
        daily_log = {"date": date_str, "predictions": []}
        if log_path.exists():
            try:
                with open(log_path, 'r') as f:
                    daily_log = json.load(f)
            except json.JSONDecodeError:
                # File is corrupted, start fresh
                print(f"  Warning: Existing log file corrupted, creating new one")
                daily_log = {"date": date_str, "predictions": []}
        
        # Get prediction data
        pred_data = result.to_dict()
        
        # Append new prediction
        daily_log["predictions"].append(pred_data)
        
        # Save
        with open(log_path, 'w') as f:
            json.dump(daily_log, f, indent=2)
        
        print(f"Logged prediction to {log_path}")
        return log_path
    
    def print_result(self, result: InferenceResult) -> None:
        """Print formatted result to console."""
        print()
        print("=" * 70)
        print("SIMONS-DALIO REGIME ENGINE - Daily Inference")
        print("=" * 70)
        print()
        
        print(f"Date: {result.inference_time.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Data as of: {result.data_as_of.strftime('%Y-%m-%d')}")
        print()
        
        # Data status
        print("DATA STATUS:")
        for symbol, status in result.data_quality.sources.items():
            print(f"  {status}")
        print(f"\n  Data Quality Score: {result.data_quality.overall_score:.2f}", end="")
        if result.data_quality.overall_score >= 0.9:
            print(" (Excellent)")
        elif result.data_quality.overall_score >= 0.7:
            print(" (Good)")
        else:
            print(" (Degraded)")
        
        if result.data_quality.warnings:
            print("\n  Warnings:")
            for w in result.data_quality.warnings:
                print(f"    ⚠ {w}")
        print()
        
        # Regime prediction
        print("REGIME PREDICTION:")
        print(f"  Current Regime: {result.regime} ({result.signal.regime_name})")
        print(f"  Confidence: {result.signal.gmm_probability*100:.1f}%")
        print()
        print(f"  Regime Performance (historical):")
        print(f"    Sharpe: {result.signal.regime_sharpe:.2f} "
              f"[{result.signal.regime_sharpe_ci[0]:.2f}, {result.signal.regime_sharpe_ci[1]:.2f}]")
        print()
        
        # Signal
        signal_str = result.signal.signal.value
        if signal_str == "BUY":
            print(f"SIGNAL: ████ {signal_str} ████")
        elif signal_str == "SELL":
            print(f"SIGNAL: ░░░░ {signal_str} ░░░░")
        else:
            print(f"SIGNAL: ──── {signal_str} ────")
        print()
        
        print(f"  Position Recommendation: {result.signal.position_recommendation}")
        print(f"  Confidence Level: {result.signal.confidence.value}")
        print()
        
        # Latent position
        latent_str = ", ".join([f"{x:.2f}" for x in result.latent_vector])
        print(f"  Latent Position: [{latent_str}]")
        print()
        
        # Model info
        print(f"Model: {result.model_id}")
        print(f"Model Sharpe Spread: {result.model_sharpe_spread:.2f}")
        print("=" * 70)


def get_latest_model_path() -> Optional[Path]:
    """Find the most recent saved model."""
    checkpoint_dir = Path("./checkpoints")
    if not checkpoint_dir.exists():
        return None
    
    # Find directories with metadata.json
    model_dirs = []
    for d in checkpoint_dir.iterdir():
        if d.is_dir() and (d / "metadata.json").exists():
            with open(d / "metadata.json", 'r') as f:
                meta = json.load(f)
            model_dirs.append((d, meta.get('created_at', '')))
    
    if not model_dirs:
        return None
    
    # Sort by creation time (most recent first)
    model_dirs.sort(key=lambda x: x[1], reverse=True)
    return model_dirs[0][0]


if __name__ == "__main__":
    print("InferencePipeline Test")
    print("=" * 60)
    
    # Find latest model
    model_path = get_latest_model_path()
    
    if model_path is None:
        print("No saved models found. Run train_autoencoder.py first.")
    else:
        print(f"Found model: {model_path}")
        
        # Create pipeline
        pipeline = InferencePipeline(model_path)
        
        # Run inference
        result = pipeline.run(dry_run=True)
        
        # Print result
        pipeline.print_result(result)

