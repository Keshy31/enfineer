#!/usr/bin/env python
"""
Daily Regime Inference Script
=============================
Run inference to get today's trading signal from the trained model.

Usage:
    python scripts/infer_regime.py                    # Basic usage
    python scripts/infer_regime.py --model ./checkpoints/my_model
    python scripts/infer_regime.py --json             # JSON output
    python scripts/infer_regime.py --dry-run          # No logging

Run from project root:
    python scripts/infer_regime.py
"""

import sys
import json
import argparse
from pathlib import Path
from datetime import datetime

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from inference.pipeline import InferencePipeline, get_latest_model_path, InferenceResult


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Run daily regime inference for BTC trading signal."
    )
    
    parser.add_argument(
        "--model", "-m",
        type=str,
        default=None,
        help="Path to model checkpoint directory. If not specified, uses latest."
    )
    
    parser.add_argument(
        "--json", "-j",
        action="store_true",
        help="Output result as JSON (for programmatic use)."
    )
    
    parser.add_argument(
        "--dry-run", "-n",
        action="store_true",
        help="Don't log prediction to file."
    )
    
    parser.add_argument(
        "--log-dir",
        type=str,
        default="./logs/predictions",
        help="Directory for prediction logs."
    )
    
    parser.add_argument(
        "--list-models",
        action="store_true",
        help="List all saved models and exit."
    )
    
    return parser.parse_args()


def list_models():
    """List all saved models with key metrics."""
    checkpoint_dir = Path("./checkpoints")
    
    if not checkpoint_dir.exists():
        print("No checkpoints directory found.")
        return
    
    print("=" * 70)
    print("SAVED MODELS")
    print("=" * 70)
    print()
    
    models = []
    for d in sorted(checkpoint_dir.iterdir()):
        if not d.is_dir():
            continue
        metadata_path = d / "metadata.json"
        if not metadata_path.exists():
            continue
        
        with open(metadata_path, 'r') as f:
            meta = json.load(f)
        
        models.append({
            "path": d.name,
            "created": meta.get("created_at", "unknown"),
            "latent_dim": meta.get("model", {}).get("latent_dim", "?"),
            "sharpe_spread": meta.get("performance", {}).get("sharpe_spread", 0.0),
            "optimal_k": meta.get("performance", {}).get("optimal_k", "?"),
        })
    
    if not models:
        print("No models found.")
        return
    
    # Sort by sharpe spread
    models.sort(key=lambda x: x["sharpe_spread"], reverse=True)
    
    print(f"{'Path':<50} {'Latent':<8} {'Sharpe':<10} {'K':<5}")
    print("-" * 70)
    
    for m in models:
        print(f"{m['path']:<50} {m['latent_dim']:<8} {m['sharpe_spread']:<10.2f} {m['optimal_k']:<5}")
    
    print()
    print(f"Total models: {len(models)}")
    print(f"Best model: {models[0]['path']}")


def run_inference(args) -> InferenceResult:
    """Run the inference pipeline."""
    
    # Find model path
    if args.model:
        model_path = Path(args.model)
        if not model_path.exists():
            print(f"Error: Model path not found: {model_path}")
            sys.exit(1)
    else:
        model_path = get_latest_model_path()
        if model_path is None:
            print("Error: No saved models found.")
            print("Run 'python scripts/train_autoencoder.py' first to train a model.")
            sys.exit(1)
    
    if not args.json:
        print()
        print("=" * 70)
        print("SIMONS-DALIO REGIME ENGINE")
        print("Daily Inference")
        print("=" * 70)
        print()
        print(f"Model: {model_path.name}")
        print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print()
    
    # Create pipeline
    try:
        pipeline = InferencePipeline(
            model_path=model_path,
            log_dir=Path(args.log_dir),
        )
    except Exception as e:
        if args.json:
            print(json.dumps({"error": str(e)}))
        else:
            print(f"Error loading model: {e}")
        sys.exit(1)
    
    # Run inference
    try:
        result = pipeline.run(dry_run=args.dry_run)
    except Exception as e:
        if args.json:
            print(json.dumps({"error": str(e)}))
        else:
            print(f"Error running inference: {e}")
        sys.exit(1)
    
    # Output result
    if args.json:
        print(result.to_json())
    else:
        pipeline.print_result(result)
        
        if args.dry_run:
            print("(Dry run - prediction not logged)")
        else:
            log_path = Path(args.log_dir) / f"{datetime.now().strftime('%Y-%m-%d')}.json"
            print(f"Logged to: {log_path}")
    
    return result


def main():
    """Main entry point."""
    args = parse_args()
    
    if args.list_models:
        list_models()
        return
    
    result = run_inference(args)
    
    # Exit with code based on signal for scripting
    if result.signal.signal.value == "BUY":
        sys.exit(0)  # Success/Buy
    elif result.signal.signal.value == "SELL":
        sys.exit(1)  # Sell
    else:
        sys.exit(2)  # Hold


if __name__ == "__main__":
    main()

