"""
Statistical Rigor Test Suite
============================
Comprehensive validation of regime detection using the new
statistical analysis and transaction cost modules.

This is the "final exam" before risking capital.

Run from project root:
    python scripts/test_statistical_rigor.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from data.manager import DataManager
from data.walk_forward import (
    WalkForwardGMM,
    get_stationary_features,
)
from analysis.statistical_tests import (
    bootstrap_sharpe_ci,
    bootstrap_sharpe_difference_test,
    compute_regime_transition_matrix,
    test_regime_significance,
    validate_alpha,
)
from backtest.costs import (
    CostModel,
    COST_PRESETS,
    analyze_cost_impact,
    compute_break_even_sharpe,
    estimate_capacity,
)


def run_statistical_rigor_test():
    """Run comprehensive statistical validation on walk-forward results."""
    
    print("=" * 70)
    print("SIMONS-DALIO REGIME ENGINE")
    print("Statistical Rigor Validation Suite (v1.2)")
    print("=" * 70)
    print()
    
    # =========================================
    # Step 1: Load Data and Run Walk-Forward
    # =========================================
    print("[1/7] Loading data and running walk-forward GMM...")
    
    dm = DataManager("./data")
    features = dm.get_features(
        "BTC-USD", "1d",
        feature_set="combined",
        start="2020-01-01",
    )
    
    stationary_cols = get_stationary_features()
    available_cols = [c for c in stationary_cols if c in features.columns]
    
    wf = WalkForwardGMM(
        n_regimes=8,
        min_train_days=504,
        test_days=126,
        gap_days=30,
        use_pca=True,
        pca_variance=0.95,
    )
    
    wf_result = wf.fit_predict(features, available_cols)
    
    print(f"\n  Total OOS samples: {len(wf_result.oos_labels)}")
    print(f"  Date range: {wf_result.oos_dates.min().date()} to {wf_result.oos_dates.max().date()}")
    
    # =========================================
    # Step 2: Extract Regime Returns
    # =========================================
    print("\n[2/7] Extracting regime-specific returns...")
    
    oos_df = features.loc[wf_result.oos_dates].copy()
    oos_df['regime'] = wf_result.oos_labels
    oos_df['fwd_return_5d'] = oos_df['Close'].shift(-5) / oos_df['Close'] - 1
    oos_df['daily_return'] = oos_df['Close'].pct_change()
    
    # Group returns by regime
    returns_by_regime = {}
    for regime in range(wf_result.n_regimes):
        mask = oos_df['regime'] == regime
        returns = oos_df.loc[mask, 'fwd_return_5d'].dropna().values
        if len(returns) > 10:
            returns_by_regime[regime] = returns
    
    print(f"  Regimes with sufficient data: {list(returns_by_regime.keys())}")
    
    # =========================================
    # Step 3: Bootstrap Sharpe CIs
    # =========================================
    print("\n[3/7] Computing bootstrap confidence intervals...")
    print()
    
    bootstrap_results = {}
    for regime, returns in returns_by_regime.items():
        # Convert 5-day returns to annualized Sharpe
        # Sharpe = mean/std * sqrt(periods_per_year)
        # For 5-day returns: periods_per_year = 252/5 = 50.4
        try:
            result = bootstrap_sharpe_ci(
                returns,
                n_bootstrap=10000,
                annualization=252/5,  # 5-day periods per year
            )
            bootstrap_results[regime] = result
            
            sig = "***" if result.significant else ""
            print(f"  Regime {regime}: Sharpe = {result.point_estimate:6.2f}  "
                  f"95% CI: [{result.ci_lower:6.2f}, {result.ci_upper:6.2f}] {sig}")
        except Exception as e:
            print(f"  Regime {regime}: Error - {e}")
    
    # =========================================
    # Step 4: Regime Significance Tests
    # =========================================
    print("\n[4/7] Testing regime significance (with Bonferroni correction)...")
    print()
    
    sig_df = test_regime_significance(returns_by_regime, alpha=0.05, method='bonferroni')
    
    # Format output
    cols_to_show = ['regime', 'n_obs', 'mean_return', 'sharpe', 'adjusted_p', 'significant']
    print(sig_df[cols_to_show].to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    
    n_significant = sig_df['significant'].sum()
    print(f"\n  Regimes with significant performance: {n_significant}/{len(sig_df)}")
    
    # =========================================
    # Step 5: Regime Transition Analysis
    # =========================================
    print("\n[5/7] Analyzing regime transitions...")
    
    trans_matrix, trans_stats = compute_regime_transition_matrix(
        wf_result.oos_labels,
        n_regimes=wf_result.n_regimes
    )
    
    print(f"\n  Avg regime persistence: {trans_stats['avg_persistence']:.1%}")
    print(f"  Overall stability: {trans_stats['stability']:.1%}")
    print(f"  Regime changes: {trans_stats['n_changes']} over {trans_stats['n_observations']} days")
    
    print("\n  Persistence by regime:")
    for regime, persist in trans_stats['persistence_by_regime'].items():
        if regime in returns_by_regime:
            print(f"    Regime {regime}: {persist:.1%} daily persistence")
    
    # =========================================
    # Step 6: Transaction Cost Analysis
    # =========================================
    print("\n[6/7] Analyzing transaction cost impact...")
    print()
    
    # Create strategy returns
    best_regime = sig_df.loc[sig_df['sharpe'].idxmax(), 'regime'] if len(sig_df) > 0 else 0
    bullish_regimes = sig_df[sig_df['sharpe'] > 0.3]['regime'].tolist() if len(sig_df) > 0 else []
    
    if not bullish_regimes:
        bullish_regimes = [int(best_regime)]
    
    oos_df['position'] = oos_df['regime'].isin(bullish_regimes).astype(int)
    
    # Test with different cost models
    print("  Cost Impact Analysis:")
    print("  " + "-" * 60)
    
    for preset_name, cost_model in COST_PRESETS.items():
        if 'btc' in preset_name:
            analysis = analyze_cost_impact(
                oos_df['daily_return'].dropna(),
                oos_df['position'].dropna(),
                cost_model
            )
            
            status = "PASS" if analysis['alpha_survives'] else "FAIL"
            print(f"  {preset_name:20s}: Gross Sharpe={analysis['gross_sharpe']:5.2f}, "
                  f"Net Sharpe={analysis['net_sharpe']:5.2f}, "
                  f"Decay={analysis['sharpe_decay']:5.2f} [{status}]")
    
    # Break-even analysis
    print("\n  Break-Even Sharpe (to cover costs):")
    for preset_name, cost_model in COST_PRESETS.items():
        if 'btc' in preset_name:
            be_sharpe = compute_break_even_sharpe(cost_model, trades_per_year=52, volatility=0.60)
            print(f"    {preset_name}: Need Sharpe > {be_sharpe:.2f}")
    
    # =========================================
    # Step 7: Capacity Estimation
    # =========================================
    print("\n[7/7] Estimating strategy capacity...")
    
    # Estimate BTC daily volume (approximate)
    btc_daily_volume = 30e9  # $30B typical
    
    capacity = estimate_capacity(
        btc_daily_volume,
        max_participation_rate=0.01,
        holding_period_days=5
    )
    
    print(f"\n  Assuming BTC daily volume: ${btc_daily_volume/1e9:.0f}B")
    print(f"  Max daily turnover (1% participation): ${capacity['max_daily_turnover_usd']/1e6:.0f}M")
    print(f"  Max position size: ${capacity['max_position_usd']/1e6:.0f}M")
    print(f"  Estimated max AUM: ${capacity['max_aum_usd']/1e6:.0f}M")
    
    # =========================================
    # Summary Report
    # =========================================
    print("\n" + "=" * 70)
    print("STATISTICAL RIGOR SUMMARY")
    print("=" * 70)
    print()
    
    # Best and worst regimes
    if len(sig_df) > 0:
        best = sig_df.loc[sig_df['sharpe'].idxmax()]
        worst = sig_df.loc[sig_df['sharpe'].idxmin()]
        
        print(f"Best Regime: {int(best['regime'])}")
        print(f"  Sharpe: {best['sharpe']:.2f}")
        if int(best['regime']) in bootstrap_results:
            br = bootstrap_results[int(best['regime'])]
            print(f"  95% CI: [{br.ci_lower:.2f}, {br.ci_upper:.2f}]")
            print(f"  Significant: {br.significant}")
        
        print(f"\nWorst Regime: {int(worst['regime'])}")
        print(f"  Sharpe: {worst['sharpe']:.2f}")
        if int(worst['regime']) in bootstrap_results:
            br = bootstrap_results[int(worst['regime'])]
            print(f"  95% CI: [{br.ci_lower:.2f}, {br.ci_upper:.2f}]")
    
    # Overall verdict
    print("\n" + "-" * 70)
    print("VALIDATION CHECKLIST")
    print("-" * 70)
    
    checks = []
    
    # Check 1: Sharpe spread
    sharpe_spread = sig_df['sharpe'].max() - sig_df['sharpe'].min() if len(sig_df) > 0 else 0
    check1 = sharpe_spread > 2.0
    checks.append(check1)
    print(f"[{'X' if check1 else ' '}] Sharpe spread > 2.0: {sharpe_spread:.2f}")
    
    # Check 2: Best regime significant
    check2 = len(sig_df[sig_df['significant']]) > 0
    checks.append(check2)
    print(f"[{'X' if check2 else ' '}] At least one regime statistically significant")
    
    # Check 3: Regime persistence
    check3 = trans_stats['avg_persistence'] > 0.5
    checks.append(check3)
    print(f"[{'X' if check3 else ' '}] Avg regime persistence > 50%: {trans_stats['avg_persistence']:.1%}")
    
    # Check 4: Survives retail costs
    retail_analysis = analyze_cost_impact(
        oos_df['daily_return'].dropna(),
        oos_df['position'].dropna(),
        COST_PRESETS['btc_retail']
    )
    check4 = retail_analysis['net_sharpe'] > 0.5
    checks.append(check4)
    print(f"[{'X' if check4 else ' '}] Net Sharpe > 0.5 (retail costs): {retail_analysis['net_sharpe']:.2f}")
    
    # Final verdict
    passed = sum(checks)
    total = len(checks)
    
    print("\n" + "=" * 70)
    if passed == total:
        print(f"VERDICT: ALL TESTS PASSED ({passed}/{total})")
        print("Strategy meets statistical rigor requirements for paper trading.")
    elif passed >= total // 2:
        print(f"VERDICT: PARTIAL PASS ({passed}/{total})")
        print("Some criteria unmet. Review failed checks before deployment.")
    else:
        print(f"VERDICT: NEEDS WORK ({passed}/{total})")
        print("Significant issues found. Do not deploy until resolved.")
    print("=" * 70)
    
    # =========================================
    # Generate Visualization
    # =========================================
    print("\nGenerating visualization...")
    
    output_dir = Path("./output")
    output_dir.mkdir(exist_ok=True)
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 1. Sharpe with CIs
    ax1 = axes[0, 0]
    regimes = list(bootstrap_results.keys())
    sharpes = [bootstrap_results[r].point_estimate for r in regimes]
    ci_lowers = [bootstrap_results[r].ci_lower for r in regimes]
    ci_uppers = [bootstrap_results[r].ci_upper for r in regimes]
    
    x = np.arange(len(regimes))
    bars = ax1.bar(x, sharpes, color=['green' if s > 0 else 'red' for s in sharpes], alpha=0.7)
    ax1.errorbar(x, sharpes, yerr=[np.array(sharpes)-np.array(ci_lowers), 
                                   np.array(ci_uppers)-np.array(sharpes)],
                 fmt='none', color='black', capsize=5)
    ax1.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax1.set_xticks(x)
    ax1.set_xticklabels([f'R{r}' for r in regimes])
    ax1.set_xlabel('Regime')
    ax1.set_ylabel('Sharpe Ratio')
    ax1.set_title('Regime Sharpe Ratios with 95% Bootstrap CI')
    
    # 2. Transition matrix heatmap
    ax2 = axes[0, 1]
    # Only show regimes with data
    active_regimes = sorted(returns_by_regime.keys())
    if len(active_regimes) > 0:
        sub_trans = trans_matrix[np.ix_(active_regimes, active_regimes)]
        im = ax2.imshow(sub_trans, cmap='Blues', aspect='auto')
        ax2.set_xticks(range(len(active_regimes)))
        ax2.set_yticks(range(len(active_regimes)))
        ax2.set_xticklabels([f'R{r}' for r in active_regimes])
        ax2.set_yticklabels([f'R{r}' for r in active_regimes])
        ax2.set_xlabel('To Regime')
        ax2.set_ylabel('From Regime')
        ax2.set_title('Regime Transition Probabilities')
        plt.colorbar(im, ax=ax2)
        
        # Add text annotations
        for i in range(len(active_regimes)):
            for j in range(len(active_regimes)):
                ax2.text(j, i, f'{sub_trans[i,j]:.2f}', ha='center', va='center', 
                        color='white' if sub_trans[i,j] > 0.5 else 'black')
    
    # 3. Cost impact
    ax3 = axes[1, 0]
    cost_names = []
    gross_sharpes = []
    net_sharpes = []
    
    for preset_name, cost_model in COST_PRESETS.items():
        if 'btc' in preset_name:
            analysis = analyze_cost_impact(
                oos_df['daily_return'].dropna(),
                oos_df['position'].dropna(),
                cost_model
            )
            cost_names.append(preset_name.replace('btc_', ''))
            gross_sharpes.append(analysis['gross_sharpe'])
            net_sharpes.append(analysis['net_sharpe'])
    
    x = np.arange(len(cost_names))
    width = 0.35
    ax3.bar(x - width/2, gross_sharpes, width, label='Gross Sharpe', color='steelblue')
    ax3.bar(x + width/2, net_sharpes, width, label='Net Sharpe', color='coral')
    ax3.axhline(y=0.5, color='green', linestyle='--', label='Threshold (0.5)')
    ax3.set_xticks(x)
    ax3.set_xticklabels(cost_names)
    ax3.set_xlabel('Cost Scenario')
    ax3.set_ylabel('Sharpe Ratio')
    ax3.set_title('Strategy Performance vs Transaction Costs')
    ax3.legend()
    
    # 4. Summary table
    ax4 = axes[1, 1]
    ax4.axis('off')
    
    summary_data = [
        ['Metric', 'Value', 'Status'],
        ['Sharpe Spread', f'{sharpe_spread:.2f}', 'PASS' if sharpe_spread > 2 else 'FAIL'],
        ['Significant Regimes', f'{n_significant}/{len(sig_df)}', 'PASS' if n_significant > 0 else 'FAIL'],
        ['Avg Persistence', f'{trans_stats["avg_persistence"]:.1%}', 'PASS' if trans_stats["avg_persistence"] > 0.5 else 'FAIL'],
        ['Net Sharpe (retail)', f'{retail_analysis["net_sharpe"]:.2f}', 'PASS' if retail_analysis["net_sharpe"] > 0.5 else 'FAIL'],
        ['Overall', f'{passed}/{total}', 'PASS' if passed == total else 'NEEDS WORK'],
    ]
    
    table = ax4.table(
        cellText=summary_data[1:],
        colLabels=summary_data[0],
        loc='center',
        cellLoc='center',
    )
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1.2, 1.8)
    ax4.set_title('Statistical Rigor Summary', y=0.9)
    
    plt.tight_layout()
    plt.savefig(output_dir / "statistical_rigor_results.png", dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Saved visualization to output/statistical_rigor_results.png")
    
    return {
        'wf_result': wf_result,
        'sig_df': sig_df,
        'bootstrap_results': bootstrap_results,
        'trans_stats': trans_stats,
        'checks': checks,
        'passed': passed,
        'total': total,
    }


if __name__ == "__main__":
    results = run_statistical_rigor_test()

