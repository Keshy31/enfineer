"""
Latent Space Decoder
====================
Decodes the abstract 8D latent space into human-readable themes.

It calculates the correlation between each latent dimension and the 
original input features to answer: "What does Dimension X represent?"
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import seaborn as sns
import matplotlib.pyplot as plt

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from inference.pipeline import InferencePipeline, get_latest_model_path

def interpret_latent_space():
    print("=" * 70)
    print("LATENT SPACE DECODER")
    print("=" * 70)

    # 1. Load Model
    model_path = get_latest_model_path()
    if not model_path:
        print("No model found!")
        return
        
    print(f"\n[1/4] Loading model from: {model_path.name}")
    pipeline = InferencePipeline(model_path)
    
    # 2. Fetch & Prep All Data
    print("[2/4] Processing historical data...")
    # Get last 2 years of data for robust correlation
    data = pipeline.fetch_data(lookback_days=730)
    features = pipeline.compute_features(data)
    
    # We need to process data in batches or loop to get latent vectors
    # (Reusing pipeline logic for batch processing)
    temporal_list = []
    macro_list = []
    valid_indices = []
    
    seq_len = pipeline.model_config.sequence_length
    
    # Prepare inputs for all valid dates
    print(f"  Generating latent vectors for {len(features)} days...")
    
    # Get stats for normalization
    temp_mean = pipeline.norm_stats['temp_mean']
    temp_std = pipeline.norm_stats['temp_std']
    macro_mean = pipeline.norm_stats['macro_mean']
    macro_std = pipeline.norm_stats['macro_std']
    
    from data.training import get_temporal_feature_cols, get_macro_feature_cols
    from data.walk_forward import get_stationary_features
    
    temp_cols = get_temporal_feature_cols()
    macro_cols = [c for c in get_macro_feature_cols() if c in features.columns]
    stationary = get_stationary_features()
    macro_cols = [c for c in macro_cols if c in stationary]

    latent_vectors = []
    
    # Iterate through history
    for i in range(seq_len, len(features)):
        # Extract window
        window = features.iloc[i-seq_len:i]
        current_macro = features.iloc[i:i+1]
        
        # Normalize
        t_data = (window[temp_cols].values - temp_mean) / temp_std
        m_data = (current_macro[macro_cols].values - macro_mean) / macro_std
        
        # Convert to tensor
        t_tensor = torch.from_numpy(t_data.astype(np.float32)).unsqueeze(0)
        m_tensor = torch.from_numpy(m_data.astype(np.float32))
        
        # Encode
        with torch.no_grad():
            latent = pipeline.model.encode(t_tensor, m_tensor)
            latent_vectors.append(latent.numpy()[0])
            valid_indices.append(features.index[i])

    # Create DataFrames
    latent_df = pd.DataFrame(latent_vectors, index=valid_indices, columns=[f"Latent_{i}" for i in range(8)])
    feature_df = features.loc[valid_indices]
    
    # 3. Correlation Analysis
    print("\n[3/4] computing correlations...")
    
    # Select key features to correlate against (exclude noise)
    key_features = [
        'log_return', 'volatility_20d', 'momentum_20d', 
        'TNX_Close', 'DXY_Close', 'GLD_Close',
        'rsi_14', 'bb_width'
    ]
    available_features = [f for f in key_features if f in feature_df.columns]
    
    correlation_matrix = pd.DataFrame(index=latent_df.columns, columns=available_features)
    
    for lat_col in latent_df.columns:
        for feat_col in available_features:
            correlation_matrix.loc[lat_col, feat_col] = latent_df[lat_col].corr(feature_df[feat_col])
            
    # 4. Results
    print("\n" + "=" * 70)
    print("DECODER RESULTS: What does each dimension mean?")
    print("=" * 70)
    
    for lat_col in latent_df.columns:
        # Find strongest correlation
        corrs = correlation_matrix.loc[lat_col].astype(float)
        top_feature = corrs.abs().idxmax()
        top_val = corrs[top_feature]
        
        meaning = "Unknown"
        if abs(top_val) > 0.6:
            meaning = f"Strongly tracks {top_feature}"
        elif abs(top_val) > 0.3:
            meaning = f"Weakly tracks {top_feature}"
        else:
            meaning = "Abstract / Mixed"
            
        direction = "Positive" if top_val > 0 else "Inverse"
        print(f"{lat_col}: {meaning:<30} ({direction} corr: {top_val:.2f})")

    # Optional: Save Heatmap
    plt.figure(figsize=(12, 8))
    sns.heatmap(correlation_matrix.astype(float), annot=True, cmap="coolwarm", center=0)
    plt.title("Latent Space Decoder Ring")
    plt.tight_layout()
    plt.savefig("output/latent_decoder.png")
    print(f"\nSaved visualization to output/latent_decoder.png")

if __name__ == "__main__":
    interpret_latent_space()