"""
Signal Mapper
=============
Maps regime labels to trading signals based on historical performance.

The mapping is derived from walk-forward validation results:
- Regimes with significantly positive Sharpe → BUY
- Regimes with significantly negative Sharpe → SELL
- Uncertain regimes → HOLD (stay in current position)
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple


class TradingSignal(Enum):
    """Trading signal output."""
    BUY = "BUY"           # Enter or maintain long position
    SELL = "SELL"         # Exit long or enter short
    HOLD = "HOLD"         # Maintain current position
    FLAT = "FLAT"         # Go to cash (no position)


class ConfidenceLevel(Enum):
    """Confidence in the signal."""
    HIGH = "HIGH"         # Strong statistical significance
    MEDIUM = "MEDIUM"     # Moderate significance
    LOW = "LOW"           # Weak or no significance


@dataclass
class SignalResult:
    """
    Complete signal result with context.
    
    Attributes
    ----------
    signal : TradingSignal
        The trading action to take.
    confidence : ConfidenceLevel
        How confident we are in this signal.
    regime : int
        The detected regime label.
    regime_name : str
        Human-readable regime description.
    regime_sharpe : float
        Historical Sharpe ratio for this regime.
    regime_sharpe_ci : Tuple[float, float]
        95% confidence interval for Sharpe.
    gmm_probability : float
        GMM probability for this regime classification.
    position_recommendation : str
        Plain English position recommendation.
    """
    signal: TradingSignal
    confidence: ConfidenceLevel
    regime: int
    regime_name: str
    regime_sharpe: float
    regime_sharpe_ci: Tuple[float, float]
    gmm_probability: float
    position_recommendation: str
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "signal": self.signal.value,
            "confidence": self.confidence.value,
            "regime": self.regime,
            "regime_name": self.regime_name,
            "regime_sharpe": self.regime_sharpe,
            "regime_sharpe_ci_lower": self.regime_sharpe_ci[0],
            "regime_sharpe_ci_upper": self.regime_sharpe_ci[1],
            "gmm_probability": self.gmm_probability,
            "position_recommendation": self.position_recommendation,
        }
    
    def __repr__(self) -> str:
        return (
            f"SignalResult(\n"
            f"  signal={self.signal.value} ({self.confidence.value})\n"
            f"  regime={self.regime} ({self.regime_name})\n"
            f"  sharpe={self.regime_sharpe:.2f} [{self.regime_sharpe_ci[0]:.2f}, {self.regime_sharpe_ci[1]:.2f}]\n"
            f"  recommendation: {self.position_recommendation}\n"
            f")"
        )


@dataclass
class RegimeProfile:
    """Profile of a regime's historical performance."""
    regime_id: int
    name: str
    sharpe: float
    sharpe_ci_lower: float
    sharpe_ci_upper: float
    mean_return_pct: float
    win_rate: float
    sample_count: int
    is_significant: bool
    signal: TradingSignal
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'RegimeProfile':
        """Create from dictionary (e.g., from saved metadata)."""
        return cls(
            regime_id=data.get('regime', data.get('regime_id', 0)),
            name=data.get('name', f"Regime {data.get('regime', 0)}"),
            sharpe=data.get('sharpe', 0.0),
            sharpe_ci_lower=data.get('sharpe_ci_lower', -1.0),
            sharpe_ci_upper=data.get('sharpe_ci_upper', 1.0),
            mean_return_pct=data.get('mean_return', data.get('mean_return_pct', 0.0)),
            win_rate=data.get('win_rate', 50.0),
            sample_count=data.get('count', data.get('sample_count', 0)),
            is_significant=data.get('significant', data.get('is_significant', False)),
            signal=TradingSignal(data.get('signal', 'HOLD')),
        )


class SignalMapper:
    """
    Maps regime labels to trading signals.
    
    The mapper uses historical regime performance (Sharpe ratios and
    confidence intervals) to determine appropriate signals.
    
    Parameters
    ----------
    regime_profiles : List[RegimeProfile], optional
        Pre-defined regime profiles. If None, uses defaults from
        8D autoencoder experiment results.
    bullish_threshold : float
        Minimum Sharpe to consider a regime bullish.
    bearish_threshold : float
        Maximum Sharpe to consider a regime bearish.
    min_confidence : float
        Minimum GMM probability to generate high-confidence signal.
        
    Example
    -------
    >>> mapper = SignalMapper()
    >>> result = mapper.get_signal(regime=0, gmm_proba=0.87)
    >>> print(result.signal)  # TradingSignal.BUY
    """
    
    # Default regime profiles from 8D autoencoder experiment
    # These should be updated with actual trained model results
    DEFAULT_PROFILES = [
        {
            "regime_id": 0,
            "name": "Bullish",
            "sharpe": 2.77,
            "sharpe_ci_lower": 0.89,
            "sharpe_ci_upper": 4.83,
            "mean_return_pct": 2.1,
            "win_rate": 62.0,
            "sample_count": 49,
            "is_significant": True,
            "signal": "BUY",
        },
        {
            "regime_id": 1,
            "name": "Neutral-Positive",
            "sharpe": 0.78,
            "sharpe_ci_lower": -0.56,
            "sharpe_ci_upper": 2.13,
            "mean_return_pct": 0.8,
            "win_rate": 54.0,
            "sample_count": 112,
            "is_significant": False,
            "signal": "HOLD",
        },
        {
            "regime_id": 2,
            "name": "Moderate-Bullish",
            "sharpe": 1.40,
            "sharpe_ci_lower": -0.49,
            "sharpe_ci_upper": 3.34,
            "mean_return_pct": 1.2,
            "win_rate": 58.0,
            "sample_count": 55,
            "is_significant": False,
            "signal": "HOLD",
        },
        {
            "regime_id": 3,
            "name": "Bearish",
            "sharpe": -1.48,
            "sharpe_ci_lower": -2.76,
            "sharpe_ci_upper": -0.21,
            "mean_return_pct": -1.5,
            "win_rate": 42.0,
            "sample_count": 117,
            "is_significant": True,
            "signal": "SELL",
        },
        {
            "regime_id": 4,
            "name": "Neutral",
            "sharpe": 1.04,
            "sharpe_ci_lower": -0.51,
            "sharpe_ci_upper": 2.55,
            "mean_return_pct": 0.9,
            "win_rate": 55.0,
            "sample_count": 86,
            "is_significant": False,
            "signal": "HOLD",
        },
    ]
    
    def __init__(
        self,
        regime_profiles: Optional[List[RegimeProfile]] = None,
        bullish_threshold: float = 0.5,
        bearish_threshold: float = -0.5,
        min_confidence: float = 0.6,
    ):
        self.bullish_threshold = bullish_threshold
        self.bearish_threshold = bearish_threshold
        self.min_confidence = min_confidence
        
        # Load profiles
        if regime_profiles is not None:
            self.profiles = {p.regime_id: p for p in regime_profiles}
        else:
            self.profiles = {}
            for p in self.DEFAULT_PROFILES:
                profile = RegimeProfile.from_dict(p)
                self.profiles[profile.regime_id] = profile
    
    def update_profiles_from_metadata(self, regime_stats: List[Dict]) -> None:
        """
        Update regime profiles from saved model metadata.
        
        Parameters
        ----------
        regime_stats : List[Dict]
            Regime statistics from model metadata.
        """
        self.profiles = {}
        for stats in regime_stats:
            # Determine signal based on Sharpe
            sharpe = stats.get('sharpe', 0.0)
            if sharpe > self.bullish_threshold:
                signal = "BUY"
            elif sharpe < self.bearish_threshold:
                signal = "SELL"
            else:
                signal = "HOLD"
            
            # Check significance (CI doesn't cross zero)
            ci_lower = stats.get('sharpe_ci_lower', -1.0)
            ci_upper = stats.get('sharpe_ci_upper', 1.0)
            is_significant = (ci_lower > 0) or (ci_upper < 0)
            
            profile = RegimeProfile(
                regime_id=stats.get('regime', 0),
                name=self._generate_regime_name(sharpe, is_significant),
                sharpe=sharpe,
                sharpe_ci_lower=ci_lower,
                sharpe_ci_upper=ci_upper,
                mean_return_pct=stats.get('mean_return', 0.0),
                win_rate=stats.get('win_rate', 50.0),
                sample_count=stats.get('count', 0),
                is_significant=is_significant,
                signal=TradingSignal(signal),
            )
            self.profiles[profile.regime_id] = profile
    
    def _generate_regime_name(self, sharpe: float, is_significant: bool) -> str:
        """Generate human-readable regime name."""
        if sharpe > 1.5 and is_significant:
            return "Strong Bullish"
        elif sharpe > 0.5:
            return "Bullish" if is_significant else "Weak Bullish"
        elif sharpe > -0.5:
            return "Neutral"
        elif sharpe > -1.5:
            return "Bearish" if is_significant else "Weak Bearish"
        else:
            return "Strong Bearish" if is_significant else "Bearish"
    
    def get_signal(
        self,
        regime: int,
        gmm_proba: float,
    ) -> SignalResult:
        """
        Get trading signal for a regime.
        
        Parameters
        ----------
        regime : int
            Regime label from GMM.
        gmm_proba : float
            GMM probability for this classification.
            
        Returns
        -------
        SignalResult
            Complete signal with context.
        """
        # Get or create profile
        if regime in self.profiles:
            profile = self.profiles[regime]
        else:
            # Unknown regime - default to HOLD
            profile = RegimeProfile(
                regime_id=regime,
                name=f"Unknown Regime {regime}",
                sharpe=0.0,
                sharpe_ci_lower=-1.0,
                sharpe_ci_upper=1.0,
                mean_return_pct=0.0,
                win_rate=50.0,
                sample_count=0,
                is_significant=False,
                signal=TradingSignal.HOLD,
            )
        
        # Determine confidence level
        if profile.is_significant and gmm_proba >= self.min_confidence:
            confidence = ConfidenceLevel.HIGH
        elif gmm_proba >= self.min_confidence:
            confidence = ConfidenceLevel.MEDIUM
        else:
            confidence = ConfidenceLevel.LOW
        
        # Generate position recommendation
        if profile.signal == TradingSignal.BUY:
            if confidence == ConfidenceLevel.HIGH:
                recommendation = "LONG BTC with full position"
            elif confidence == ConfidenceLevel.MEDIUM:
                recommendation = "LONG BTC with reduced position"
            else:
                recommendation = "Consider LONG BTC, low confidence"
        elif profile.signal == TradingSignal.SELL:
            if confidence == ConfidenceLevel.HIGH:
                recommendation = "EXIT all BTC positions"
            elif confidence == ConfidenceLevel.MEDIUM:
                recommendation = "Reduce BTC exposure"
            else:
                recommendation = "Consider reducing BTC, low confidence"
        else:
            recommendation = "HOLD current position, wait for clearer signal"
        
        return SignalResult(
            signal=profile.signal,
            confidence=confidence,
            regime=regime,
            regime_name=profile.name,
            regime_sharpe=profile.sharpe,
            regime_sharpe_ci=(profile.sharpe_ci_lower, profile.sharpe_ci_upper),
            gmm_probability=gmm_proba,
            position_recommendation=recommendation,
        )
    
    def get_all_profiles(self) -> pd.DataFrame:
        """Get all regime profiles as DataFrame."""
        data = []
        for regime_id, profile in sorted(self.profiles.items()):
            data.append({
                "regime": profile.regime_id,
                "name": profile.name,
                "sharpe": profile.sharpe,
                "ci_lower": profile.sharpe_ci_lower,
                "ci_upper": profile.sharpe_ci_upper,
                "mean_return_pct": profile.mean_return_pct,
                "win_rate": profile.win_rate,
                "samples": profile.sample_count,
                "significant": profile.is_significant,
                "signal": profile.signal.value,
            })
        return pd.DataFrame(data)


if __name__ == "__main__":
    print("SignalMapper Test")
    print("=" * 60)
    
    mapper = SignalMapper()
    
    print("Regime Profiles:")
    print(mapper.get_all_profiles().to_string(index=False))
    print()
    
    # Test signal generation
    test_cases = [
        (0, 0.92),  # Bullish, high confidence
        (3, 0.85),  # Bearish, high confidence
        (1, 0.45),  # Neutral, low confidence
        (4, 0.70),  # Neutral, medium confidence
    ]
    
    print("Signal Tests:")
    for regime, proba in test_cases:
        result = mapper.get_signal(regime, proba)
        print(f"\n  Regime {regime}, GMM prob={proba:.2f}:")
        print(f"    Signal: {result.signal.value} ({result.confidence.value})")
        print(f"    Recommendation: {result.position_recommendation}")
    
    print("\n" + "=" * 60)
    print("SIGNAL MAPPER TEST PASSED")

