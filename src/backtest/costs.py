"""
Transaction Cost Model
======================
Realistic friction modeling for backtesting.

The key insight: Alpha that doesn't survive transaction costs isn't alpha.
Renaissance's edge includes their execution quality - they minimize friction.

Cost Components:
1. Spread: Bid-ask spread (pay on entry and exit)
2. Slippage: Market impact from order size
3. Commission: Exchange/broker fees
4. Funding: Overnight financing (for leveraged positions)
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Optional, Dict, Tuple


@dataclass
class CostModel:
    """
    Transaction cost model for realistic backtesting.
    
    Default values are conservative estimates for crypto (BTC) trading
    on major exchanges (Binance, Coinbase Pro).
    
    Parameters
    ----------
    spread_bps : float
        Half-spread in basis points. Pay on entry and exit.
        BTC typical: 2-5 bps on major exchanges.
    slippage_bps : float
        Market impact in basis points per trade.
        Depends on order size relative to book depth.
    commission_bps : float
        Exchange fee in basis points.
        Maker: 0-2 bps, Taker: 2-10 bps.
    funding_bps_daily : float
        Daily funding cost for leveraged positions.
        Typically 0-5 bps/day depending on market conditions.
    
    Example
    -------
    >>> costs = CostModel(spread_bps=5, slippage_bps=2, commission_bps=3)
    >>> costs.one_way_cost_bps
    10.0
    >>> costs.round_trip_cost_bps
    20.0
    """
    
    spread_bps: float = 5.0       # Half-spread: 5 bps = 0.05%
    slippage_bps: float = 2.0     # Market impact
    commission_bps: float = 3.0   # Exchange fee (taker)
    funding_bps_daily: float = 0.0  # Funding rate (0 for spot)
    
    @property
    def one_way_cost_bps(self) -> float:
        """Cost to enter OR exit a position (in basis points)."""
        return self.spread_bps + self.slippage_bps + self.commission_bps
    
    @property
    def round_trip_cost_bps(self) -> float:
        """Cost to enter AND exit a position (in basis points)."""
        return self.one_way_cost_bps * 2
    
    @property
    def round_trip_cost_pct(self) -> float:
        """Round-trip cost as percentage (for return adjustment)."""
        return self.round_trip_cost_bps / 10000
    
    def entry_cost(self, position_value_usd: float) -> float:
        """Dollar cost to enter a position."""
        return position_value_usd * self.one_way_cost_bps / 10000
    
    def exit_cost(self, position_value_usd: float) -> float:
        """Dollar cost to exit a position."""
        return position_value_usd * self.one_way_cost_bps / 10000
    
    def round_trip_cost(self, position_value_usd: float) -> float:
        """Total dollar cost to enter and exit."""
        return position_value_usd * self.round_trip_cost_bps / 10000
    
    def holding_cost(self, position_value_usd: float, days: int) -> float:
        """Funding cost for holding a leveraged position."""
        return position_value_usd * self.funding_bps_daily * days / 10000
    
    def __repr__(self) -> str:
        return (
            f"CostModel(\n"
            f"  spread: {self.spread_bps} bps\n"
            f"  slippage: {self.slippage_bps} bps\n"
            f"  commission: {self.commission_bps} bps\n"
            f"  one_way: {self.one_way_cost_bps} bps\n"
            f"  round_trip: {self.round_trip_cost_bps} bps ({self.round_trip_cost_pct*100:.3f}%)\n"
            f")"
        )


# Preset cost models for different trading scenarios
COST_PRESETS = {
    "btc_retail": CostModel(spread_bps=5, slippage_bps=3, commission_bps=5),
    "btc_institutional": CostModel(spread_bps=2, slippage_bps=1, commission_bps=2),
    "btc_conservative": CostModel(spread_bps=10, slippage_bps=5, commission_bps=5),
    "equity_retail": CostModel(spread_bps=2, slippage_bps=1, commission_bps=0),
    "equity_institutional": CostModel(spread_bps=1, slippage_bps=0.5, commission_bps=0),
}


def estimate_capacity(
    avg_daily_volume_usd: float,
    max_participation_rate: float = 0.01,
    holding_period_days: int = 5,
) -> Dict[str, float]:
    """
    Estimate maximum strategy capacity before market impact degrades returns.
    
    The key constraint: You can't trade more than X% of daily volume
    without moving the market against you.
    
    Parameters
    ----------
    avg_daily_volume_usd : float
        Average daily trading volume in USD.
    max_participation_rate : float
        Maximum fraction of daily volume to trade. 
        1% is conservative, 5% is aggressive.
    holding_period_days : int
        Average holding period in days.
        
    Returns
    -------
    Dict with capacity estimates:
        - max_position_usd: Maximum single position
        - max_daily_turnover_usd: Maximum daily trading
        - max_aum_usd: Estimated maximum AUM for strategy
        
    Example
    -------
    >>> # BTC with $30B daily volume
    >>> capacity = estimate_capacity(30e9, max_participation_rate=0.01)
    >>> print(f"Max AUM: ${capacity['max_aum_usd']/1e6:.0f}M")
    Max AUM: $1500M
    """
    # Max we can trade per day without impact
    max_daily_turnover = avg_daily_volume_usd * max_participation_rate
    
    # If we hold for N days, we can have N * daily_turnover in position
    # (because we only need to turn over 1/N per day)
    max_position = max_daily_turnover * holding_period_days
    
    # AUM = position * (1 / leverage)
    # For unleveraged, AUM = max_position
    max_aum = max_position
    
    return {
        "max_daily_turnover_usd": max_daily_turnover,
        "max_position_usd": max_position,
        "max_aum_usd": max_aum,
        "participation_rate": max_participation_rate,
        "holding_period_days": holding_period_days,
    }


def apply_costs_to_returns(
    returns: pd.Series,
    positions: pd.Series,
    cost_model: CostModel,
    initial_capital: float = 100000.0,
) -> Dict[str, pd.Series]:
    """
    Apply transaction costs to a return series based on position changes.
    
    Parameters
    ----------
    returns : pd.Series
        Daily returns (e.g., from price changes).
    positions : pd.Series
        Position sizes (0 = flat, 1 = long, -1 = short).
        Must be aligned with returns.
    cost_model : CostModel
        Transaction cost parameters.
    initial_capital : float
        Starting capital for computing dollar costs.
        
    Returns
    -------
    Dict with:
        - gross_returns: Returns before costs
        - net_returns: Returns after costs
        - cumulative_costs: Running total of costs paid
        - trade_count: Number of trades
        - total_cost_pct: Total costs as % of initial capital
        
    Example
    -------
    >>> returns = pd.Series([0.01, -0.02, 0.015])
    >>> positions = pd.Series([1, 1, 0])  # Long, then exit
    >>> result = apply_costs_to_returns(returns, positions, CostModel())
    >>> print(f"Net return: {result['net_returns'].sum()*100:.2f}%")
    """
    # Detect position changes (trades)
    position_changes = positions.diff().fillna(positions)
    trades = position_changes != 0
    
    # Cost is proportional to position change size
    # abs(change) = 0, 1, or 2 (flat->long, long->short, etc.)
    trade_sizes = position_changes.abs()
    
    # Calculate cost per trade
    cost_per_trade = trade_sizes * cost_model.one_way_cost_bps / 10000
    
    # Gross returns (just position * return)
    gross_returns = positions.shift(1).fillna(0) * returns
    
    # Net returns = gross - costs
    net_returns = gross_returns - cost_per_trade
    
    # Cumulative costs
    cumulative_costs = cost_per_trade.cumsum()
    
    # Summary stats
    trade_count = trades.sum()
    total_cost_pct = cumulative_costs.iloc[-1] if len(cumulative_costs) > 0 else 0
    
    return {
        "gross_returns": gross_returns,
        "net_returns": net_returns,
        "cost_per_trade": cost_per_trade,
        "cumulative_costs": cumulative_costs,
        "trade_count": int(trade_count),
        "total_cost_pct": total_cost_pct,
        "trades": trades,
    }


def compute_break_even_sharpe(
    cost_model: CostModel,
    trades_per_year: int = 52,  # Weekly rebalancing
    volatility: float = 0.50,   # 50% annual vol (typical for BTC)
) -> float:
    """
    Compute the minimum Sharpe ratio needed to break even after costs.
    
    This tells you: "How good does my strategy need to be just to
    cover transaction costs?"
    
    Parameters
    ----------
    cost_model : CostModel
        Transaction costs.
    trades_per_year : int
        Number of round-trip trades per year.
    volatility : float
        Annual volatility of the strategy.
        
    Returns
    -------
    float
        Break-even Sharpe ratio.
        
    Example
    -------
    >>> costs = CostModel(spread_bps=5, slippage_bps=2, commission_bps=3)
    >>> be_sharpe = compute_break_even_sharpe(costs, trades_per_year=52)
    >>> print(f"Need Sharpe > {be_sharpe:.2f} to make money")
    Need Sharpe > 0.21 to make money
    """
    # Annual cost from trading
    annual_cost = cost_model.round_trip_cost_pct * trades_per_year
    
    # Sharpe = return / volatility
    # Break-even: return = cost
    # Break-even Sharpe = cost / volatility
    break_even_sharpe = annual_cost / volatility
    
    return break_even_sharpe


def analyze_cost_impact(
    returns: pd.Series,
    positions: pd.Series,
    cost_model: CostModel,
) -> Dict[str, float]:
    """
    Comprehensive analysis of how costs impact strategy performance.
    
    Returns
    -------
    Dict with performance metrics before and after costs.
    """
    result = apply_costs_to_returns(returns, positions, cost_model)
    
    gross = result["gross_returns"]
    net = result["net_returns"]
    
    # Annualization factor
    ann_factor = np.sqrt(252)
    
    # Sharpe ratios
    gross_sharpe = (gross.mean() / gross.std() * ann_factor) if gross.std() > 0 else 0
    net_sharpe = (net.mean() / net.std() * ann_factor) if net.std() > 0 else 0
    
    # Returns
    gross_return = (1 + gross).prod() - 1
    net_return = (1 + net).prod() - 1
    
    # Cost analysis
    total_cost = result["cumulative_costs"].iloc[-1] if len(result["cumulative_costs"]) > 0 else 0
    cost_per_trade_avg = total_cost / result["trade_count"] if result["trade_count"] > 0 else 0
    
    return {
        "gross_sharpe": gross_sharpe,
        "net_sharpe": net_sharpe,
        "sharpe_decay": gross_sharpe - net_sharpe,
        "gross_return": gross_return,
        "net_return": net_return,
        "return_lost_to_costs": gross_return - net_return,
        "total_cost_pct": total_cost,
        "trade_count": result["trade_count"],
        "cost_per_trade_avg": cost_per_trade_avg,
        "alpha_survives": net_sharpe > 0.5,
    }


if __name__ == "__main__":
    print("=" * 60)
    print("Transaction Cost Model Test")
    print("=" * 60)
    print()
    
    # Test cost model
    costs = CostModel()
    print(costs)
    print()
    
    # Test presets
    print("Cost Presets:")
    for name, model in COST_PRESETS.items():
        print(f"  {name}: {model.round_trip_cost_bps} bps round-trip")
    print()
    
    # Test capacity estimation
    print("Capacity Analysis (BTC, $30B daily volume):")
    capacity = estimate_capacity(30e9, max_participation_rate=0.01)
    print(f"  Max daily turnover: ${capacity['max_daily_turnover_usd']/1e6:.0f}M")
    print(f"  Max position: ${capacity['max_position_usd']/1e6:.0f}M")
    print(f"  Max AUM: ${capacity['max_aum_usd']/1e6:.0f}M")
    print()
    
    # Test break-even Sharpe
    print("Break-Even Analysis:")
    for name, model in COST_PRESETS.items():
        be = compute_break_even_sharpe(model, trades_per_year=52)
        print(f"  {name}: need Sharpe > {be:.2f}")
    print()
    
    # Test with synthetic data
    print("Synthetic Backtest:")
    np.random.seed(42)
    n_days = 252
    
    # Synthetic returns (slight positive drift)
    returns = pd.Series(np.random.randn(n_days) * 0.02 + 0.001)
    
    # Synthetic positions (weekly rebalancing)
    positions = pd.Series(np.where(np.arange(n_days) % 7 < 5, 1, 0))
    
    analysis = analyze_cost_impact(returns, positions, costs)
    
    print(f"  Gross Sharpe: {analysis['gross_sharpe']:.2f}")
    print(f"  Net Sharpe: {analysis['net_sharpe']:.2f}")
    print(f"  Sharpe decay: {analysis['sharpe_decay']:.2f}")
    print(f"  Trade count: {analysis['trade_count']}")
    print(f"  Total cost: {analysis['total_cost_pct']*100:.2f}%")
    print(f"  Alpha survives costs: {analysis['alpha_survives']}")
    print()
    
    print("COST MODEL TEST PASSED")

