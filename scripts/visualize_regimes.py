"""
Regime Visualization (Star Chart)
=================================
Visualizes the 8D latent space in 2D and 3D using UMAP.

This script creates the "Star Chart" described in the project docs:
- Historical regimes appear as colored clouds
- The current market state is a bright star
- Trajectories show how the market moved recently

Usage:
    python scripts/visualize_regimes.py
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import umap
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import torch

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from inference.pipeline import InferencePipeline, get_latest_model_path

def visualize_regimes():
    print("=" * 70)
    print("REGIME VISUALIZATION (Star Chart)")
    print("=" * 70)

    # 1. Load Model
    model_path = get_latest_model_path()
    if not model_path:
        print("No model found!")
        return
        
    print(f"\n[1/5] Loading model from: {model_path.name}")
    pipeline = InferencePipeline(model_path)
    
    # 2. Fetch & Prep All Data
    print("[2/5] Processing historical data (last 2 years)...")
    data = pipeline.fetch_data(lookback_days=730)
    features = pipeline.compute_features(data)
    
    # Generate latent vectors (Batch process)
    print(f"  Generating latent vectors for {len(features)} days...")
    
    latent_vectors = []
    dates = []
    regimes = []
    log_returns = []
    
    seq_len = pipeline.model_config.sequence_length
    
    # Pre-calculate stats for normalization
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
    
    # Process in chunks to avoid memory issues (though unlikely with 2 years)
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
            latent_np = latent.numpy()[0]
            latent_vectors.append(latent_np)
            
            # Predict regime
            regime = pipeline.gmm.predict([latent_np])[0]
            regimes.append(regime)
            
            dates.append(features.index[i])
            log_returns.append(features.iloc[i]['log_return'])

    X = np.array(latent_vectors)
    
    # 3. UMAP Projection (8D -> 3D)
    print("\n[3/5] Projecting to 3D with UMAP...")
    reducer_3d = umap.UMAP(
        n_components=3, 
        random_state=42,
        n_neighbors=30,  # Larger neighbors = more global structure
        min_dist=0.1
    )
    embedding_3d = reducer_3d.fit_transform(X)
    
    # 4. UMAP Projection (8D -> 2D)
    print("[4/5] Projecting to 2D with UMAP...")
    reducer_2d = umap.UMAP(
        n_components=2, 
        random_state=42,
        n_neighbors=30,
        min_dist=0.1
    )
    embedding_2d = reducer_2d.fit_transform(X)
    
    # Create DataFrame for plotting
    df_viz = pd.DataFrame({
        'date': dates,
        'regime': regimes,
        'return': log_returns,
        'x_2d': embedding_2d[:, 0],
        'y_2d': embedding_2d[:, 1],
        'x_3d': embedding_3d[:, 0],
        'y_3d': embedding_3d[:, 1],
        'z_3d': embedding_3d[:, 2],
    })
    
    # Label Regimes
    # Sort regimes by Sharpe to give them meaningful colors
    # (We'll trust the pipeline's signal mapper has this info, or just use raw ID)
    df_viz['Regime'] = df_viz['regime'].astype(str)
    
    # Identify "Current" state (last point)
    current_state = df_viz.iloc[-1]
    
    # 5. Generate Plots
    print("\n[5/5] Generating interactive plots...")
    output_dir = Path("./output")
    output_dir.mkdir(exist_ok=True)
    
    # --- 2D Plot ---
    fig_2d = px.scatter(
        df_viz, 
        x='x_2d', 
        y='y_2d', 
        color='Regime',
        hover_data=['date', 'return'],
        title='Market Regimes (2D UMAP Projection)',
        color_discrete_sequence=px.colors.qualitative.Bold,
        opacity=0.6
    )
    
    # Add Current Star
    fig_2d.add_trace(
        go.Scatter(
            x=[current_state['x_2d']],
            y=[current_state['y_2d']],
            mode='markers',
            marker=dict(size=20, color='yellow', symbol='star', line=dict(width=2, color='black')),
            name='TODAY'
        )
    )
    
    # Add Trajectory (Last 30 days)
    recent = df_viz.iloc[-30:]
    fig_2d.add_trace(
        go.Scatter(
            x=recent['x_2d'],
            y=recent['y_2d'],
            mode='lines',
            line=dict(color='white', width=2, dash='dot'),
            name='30-Day Path'
        )
    )
    
    fig_2d.update_layout(template="plotly_dark")
    fig_2d.write_html(output_dir / "regimes_2d.html")
    
    # --- 3D Plot ---
    fig_3d = px.scatter_3d(
        df_viz, 
        x='x_3d', 
        y='y_3d', 
        z='z_3d',
        color='Regime',
        hover_data=['date', 'return'],
        title='Market Regimes (3D UMAP Projection)',
        color_discrete_sequence=px.colors.qualitative.Bold,
        opacity=0.5
    )
    
    # Add Current Star (3D)
    fig_3d.add_trace(
        go.Scatter3d(
            x=[current_state['x_3d']],
            y=[current_state['y_3d']],
            z=[current_state['z_3d']],
            mode='markers',
            marker=dict(size=10, color='yellow', symbol='diamond', line=dict(width=2, color='black')),
            name='TODAY'
        )
    )
    
    # Add Trajectory (Last 30 days)
    fig_3d.add_trace(
        go.Scatter3d(
            x=recent['x_3d'],
            y=recent['y_3d'],
            z=recent['z_3d'],
            mode='lines',
            line=dict(color='white', width=4),
            name='30-Day Path'
        )
    )
    
    fig_3d.update_layout(template="plotly_dark")
    fig_3d.write_html(output_dir / "regimes_3d.html")
    
    print(f"\nSaved interactive visualizations:")
    print(f"  - {output_dir / 'regimes_2d.html'}")
    print(f"  - {output_dir / 'regimes_3d.html'}")
    print("\nOpen these files in your browser to interact.")

if __name__ == "__main__":
    visualize_regimes()

