#!/usr/bin/env python3
"""
Visualization Script for Figure 2: Noise Robustness Experiment (Corrected)

This script generates Figure 2 matching the paper layout:
- Shows Noiseless vs Noisy waveforms on top
- Shows representation trajectories for CoInception and TS2Vec below
- Displays correlation values for each comparison
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.spatial.distance import cosine
from analysis_preset import get_vis_settings, PATH_CONFIG
import os
import pickle
import json
import torch
import sys
import glob

# Add TS2Vec directory to path to import TS2Vec modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'ts2vec'))

try:
    from modules.coinception import CoInception
    from ts2vec import TS2Vec
    model_import_success = True
except ImportError as e:
    print(f"Warning: Could not import model classes: {e}")
    model_import_success = False


def load_latest_model(model_type, dataset=None):
    """Load the latest trained model for a given model type.
    
    Args:
        model_type (str): Model type ('CoInception' or 'TS2Vec')
        dataset (str, optional): Dataset name, defaults to None
        
    Returns:
        tuple: (model, device, config) or (None, None, None) if no model found
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    if model_type == 'CoInception':
        # Find latest CoInception model
        model_dir = os.path.join(os.path.dirname(__file__), 'training')
        model_files = glob.glob(os.path.join(model_dir, '**', 'model.pkl'), recursive=True)
    else:  # TS2Vec
        # Find latest TS2Vec model - search recursively under training directory
        model_dir = os.path.join(os.path.dirname(__file__), 'ts2vec', 'training')
        model_files = glob.glob(os.path.join(model_dir, '**', 'model_*.pkl'), recursive=True) + \
                      glob.glob(os.path.join(model_dir, '**', 'model.pkl'), recursive=True)
    
    if not model_files:
        print(f"No {model_type} models found in {model_dir}")
        return None, None, None
    
    # Prioritize models trained on UCR datasets (which are univariate)
    ucr_model_files = [f for f in model_files if 'UCR' in f or 'ucr' in f]
    if ucr_model_files:
        model_files = ucr_model_files
    
    # Sort by modification time (newest first)
    model_files.sort(key=os.path.getmtime, reverse=True)
    latest_model = model_files[0]
    
    print(f"Loading {model_type} model from: {latest_model}")
    
    try:
        if model_type == 'CoInception':
            # Load CoInception model with dynamic input dimension detection
            # First load state_dict to get input dimensions
            state_dict = torch.load(latest_model, map_location=device)
            
            # Extract input and output dimensions from state_dict
            input_dims = state_dict['module.input_fc.weight'].shape[1]
            output_dims = state_dict['module.feature_extractor.net.3.residual.weight'].shape[0]
            
            # Create config with correct dimensions
            config = {
                "batch_size": 8,
                "lr": 0.001,
                "output_dims": output_dims,  # Match pre-trained model's output dimensions
                "max_train_length": 3000
            }
            
            # Create a dummy input with correct dimensions
            input_len = 1000  # Fixed input length for visualization
            model = CoInception(
                input_len=input_len,
                input_dims=input_dims,  # Dynamic input dimensions
                device=device,
                **config
            )
            
            # Load the model weights
            model.load(latest_model)
        else:  # TS2Vec
            # Load TS2Vec model - first we need to find its config file
            # Look for config.json in the same directory as the model
            model_dir = os.path.dirname(latest_model)
            config_file = os.path.join(model_dir, 'config.json')
            
            if os.path.exists(config_file):
                # Load config from JSON file
                with open(config_file, 'r') as f:
                    ts2vec_config = json.load(f)
                
                # Extract input_dims from config
                input_dims = ts2vec_config.get('input_dims', 1)
                output_dims = ts2vec_config.get('output_dims', 320)
                
                # Create TS2Vec instance
                model = TS2Vec(
                    input_dims=input_dims,
                    output_dims=output_dims,
                    device=device
                )
                
                # Load the model weights
                model.load(latest_model)
                config = {
                    "output_dims": output_dims
                }
            else:
                # Try with default config if no config file found
                model = TS2Vec(
                    input_dims=1,
                    output_dims=320,
                    device=device
                )
                model.load(latest_model)
                config = {
                    "output_dims": 320
                }
        
        if model_type == 'CoInception':
            # Add input_dims to config for later use
            config['input_dims'] = input_dims
        
        print(f"Successfully loaded {model_type} model")
        return model, device, config
    except Exception as e:
        print(f"Error loading {model_type} model: {e}")
        return None, None, None


def generate_figure2():
    """Generate Figure 2 matching the paper's format."""
    
    # Set font sizes
    plt.rcParams.update({
        'axes.labelsize': 12,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'axes.titlesize': 14,
        'font.size': 11,
    })
    
    # Define time parameters
    T = 20
    L = 1000  # Length of time series
    time = np.arange(L)
    
    # Generate synthetic noiseless signal (multi-frequency sine wave)
    noiseless = np.sin(2 * np.pi * time / 100) + 0.5 * np.sin(2 * np.pi * time / 50)
    
    # Fixed noise level based on paper results
    noise_level = 0.3
    
    # Generate noisy signal with two high-frequency noise components as specified in paper
    def add_high_frequency_noise(signal, noise_level=0.3):
        t = np.linspace(0, len(signal), len(signal))
        # Two high-frequency noise signals as specified in the paper
        noise1 = noise_level * np.sin(50*t/len(t)*2*np.pi)
        noise2 = noise_level * np.random.randn(len(signal)) * 0.5
        return signal + noise1 + noise2
    
    noisy = add_high_frequency_noise(noiseless, noise_level)
    
    # Calculate correlation between noiseless and noisy
    corr_signals = np.corrcoef(noiseless, noisy)[0, 1]
    
    # Prepare input data for models (add channel dimension and batch dimension)
    noiseless_input = noiseless[np.newaxis, :, np.newaxis]  # Shape: (1, L, 1)
    noisy_input = noisy[np.newaxis, :, np.newaxis]  # Shape: (1, L, 1)
    
    # Default representation dimensions
    n_dims = 10
    
    # Load models - remove dataset parameter, function will find latest models automatically
    coin_model, coin_device, coin_config = load_latest_model('CoInception')
    ts2vec_model, ts2vec_device, ts2vec_config = load_latest_model('TS2Vec')
    
    # Initialize representations as None, will only use real model results
    coin_repr_noiseless = None
    coin_repr_noisy = None
    ts2vec_repr_noiseless = None
    ts2vec_repr_noisy = None
    
    # Use real models if available
    if model_import_success:
        if coin_model is not None:
            print("Using CoInception model to generate representations")
            try:
                # Generate representations for CoInception using original input
                # CoInception expects input shape: (batch, timesteps, features)
                coin_repr_noiseless = coin_model.encode(noiseless_input)
                coin_repr_noisy = coin_model.encode(noisy_input)
                
                # Reshape representations to (n_dims, L) for visualization
                coin_repr_noiseless = coin_repr_noiseless.squeeze().T[:n_dims, :]
                coin_repr_noisy = coin_repr_noisy.squeeze().T[:n_dims, :]
            except Exception as e:
                print(f"Error generating CoInception representations: {e}")
                import traceback
                traceback.print_exc()
        
        if ts2vec_model is not None:
            print("Using TS2Vec model to generate representations")
            try:
                # Generate representations for TS2Vec
                ts2vec_repr_noiseless = ts2vec_model.encode(noiseless_input)
                ts2vec_repr_noisy = ts2vec_model.encode(noisy_input)
                
                # Reshape representations to (n_dims, L) for visualization
                ts2vec_repr_noiseless = ts2vec_repr_noiseless.squeeze().T[:n_dims, :]
                ts2vec_repr_noisy = ts2vec_repr_noisy.squeeze().T[:n_dims, :]
            except Exception as e:
                print(f"Error generating TS2Vec representations: {e}")
    
    # Skip figure generation if no real representations were generated
    if (coin_repr_noiseless is None or coin_repr_noisy is None) and (ts2vec_repr_noiseless is None or ts2vec_repr_noisy is None):
        print("Skipping figure generation: no real representations generated from models")
        return
    
    # Calculate correlations between noiseless and noisy representations
    def compute_representation_correlation(repr1, repr2):
        """Compute cosine similarity per timestep and average"""
        corr_per_timestep = []
        for t in range(repr1.shape[1]):
            # Calculate cosine similarity between representations at each timestep
            corr = 1 - cosine(repr1[0, t, :], repr2[0, t, :])
            corr_per_timestep.append(corr)
        return np.mean(corr_per_timestep)
    
    corr_coinception = compute_representation_correlation(coin_repr_noiseless[np.newaxis, :, :].transpose(0, 2, 1), coin_repr_noisy[np.newaxis, :, :].transpose(0, 2, 1)) if (coin_repr_noiseless is not None and coin_repr_noisy is not None) else ''
    corr_ts2vec = compute_representation_correlation(ts2vec_repr_noiseless[np.newaxis, :, :].transpose(0, 2, 1), ts2vec_repr_noisy[np.newaxis, :, :].transpose(0, 2, 1)) if (ts2vec_repr_noiseless is not None and ts2vec_repr_noisy is not None) else ''
    
    # Create figure with 3 rows, 3 columns (left, center for corr, right)
    fig = plt.figure(figsize=(16, 5))
    
    # Create grid specification: 3 rows, 3 columns with middle column for correlations
    # width_ratios: left plots, center correlations, right plots
    gs = fig.add_gridspec(3, 3, height_ratios=[1, 1, 1], width_ratios=[4, 1, 4], 
                          hspace=0.15, wspace=0.05)
    
    # Row 1: Noiseless and Noisy waveforms
    ax_noiseless = fig.add_subplot(gs[0, 0])
    ax_noisy = fig.add_subplot(gs[0, 2])
    
    # Plot waveforms
    ax_noiseless.plot(time, noiseless, 'b-', linewidth=1)
    ax_noiseless.set_title('Noiseless', fontsize=14, fontweight='bold')
    ax_noiseless.set_ylabel('Amp', fontsize=11)
    ax_noiseless.set_xlim(0, L)
    ax_noiseless.set_ylim(-2, 2)
    ax_noiseless.set_xticklabels([])
    
    ax_noisy.plot(time, noisy, 'b-', linewidth=1)
    ax_noisy.set_title('Noisy', fontsize=14, fontweight='bold')
    ax_noisy.set_ylabel('Amp', fontsize=11)
    ax_noisy.set_xlim(0, L)
    ax_noisy.set_ylim(-2, 2)
    ax_noisy.set_xticklabels([])
    
    # Add correlation in the center with arrows
    ax_corr1 = fig.add_subplot(gs[0, 1])
    ax_corr1.axis('off')
    ax_corr1.annotate('', xy=(0.95, 0.35), xytext=(0.05, 0.35),
                      arrowprops=dict(arrowstyle='<->', color='black', lw=1.5),
                      xycoords='axes fraction')
    ax_corr1.text(0.5, 0.65, f'Corr: {corr_signals:.3f}', transform=ax_corr1.transAxes,
                  fontsize=11, verticalalignment='center', horizontalalignment='center',
                  fontweight='bold')
    
    # Row 2: CoInception representations
    ax_coin_noiseless = fig.add_subplot(gs[1, 0])
    ax_coin_noisy = fig.add_subplot(gs[1, 2])
    
    # Plot CoInception heatmaps without colorbar
    sns.heatmap(coin_repr_noiseless, ax=ax_coin_noiseless, cmap='magma', 
                cbar=False, xticklabels=False)
    ax_coin_noiseless.set_ylabel('Dim', fontsize=11)
    ax_coin_noiseless.set_xlabel('')
    ax_coin_noiseless.set_yticks([0, 3, 6, 9])
    ax_coin_noiseless.set_yticklabels(['0', '3', '6', '9'], fontsize=10)
    
    sns.heatmap(coin_repr_noisy, ax=ax_coin_noisy, cmap='magma',
                cbar=False, xticklabels=False)
    ax_coin_noisy.set_ylabel('Dim', fontsize=11)
    ax_coin_noisy.set_xlabel('')
    ax_coin_noisy.set_yticks([0, 3, 6, 9])
    ax_coin_noisy.set_yticklabels(['0', '3', '6', '9'], fontsize=10)
    
    # Add correlation in the center with arrows for CoInception
    ax_corr2 = fig.add_subplot(gs[1, 1])
    ax_corr2.axis('off')
    ax_corr2.annotate('', xy=(0.95, 0.35), xytext=(0.05, 0.35),
                      arrowprops=dict(arrowstyle='<->', color='black', lw=1.5),
                      xycoords='axes fraction')
    ax_corr2.text(0.5, 0.65, f'Corr: {corr_coinception:.3f}', transform=ax_corr2.transAxes,
                  fontsize=11, verticalalignment='center', horizontalalignment='center',
                  fontweight='bold')
    
    # Row 3: TS2Vec representations
    ax_ts2vec_noiseless = fig.add_subplot(gs[2, 0])
    ax_ts2vec_noisy = fig.add_subplot(gs[2, 2])
    
    # Plot TS2Vec heatmaps without colorbar
    sns.heatmap(ts2vec_repr_noiseless, ax=ax_ts2vec_noiseless, cmap='magma',
                cbar=False, xticklabels=False)
    ax_ts2vec_noiseless.set_ylabel('Dim', fontsize=11)
    ax_ts2vec_noiseless.set_yticks([0, 3, 6, 9])
    ax_ts2vec_noiseless.set_yticklabels(['0', '3', '6', '9'], fontsize=10)
    
    sns.heatmap(ts2vec_repr_noisy, ax=ax_ts2vec_noisy, cmap='magma',
                cbar=False, xticklabels=False)
    ax_ts2vec_noisy.set_ylabel('Dim', fontsize=11)
    ax_ts2vec_noisy.set_yticks([0, 3, 6, 9])
    ax_ts2vec_noisy.set_yticklabels(['0', '3', '6', '9'], fontsize=10)
    
    # Add correlation in the center with arrows for TS2Vec
    ax_corr3 = fig.add_subplot(gs[2, 1])
    ax_corr3.axis('off')
    ax_corr3.annotate('', xy=(0.95, 0.35), xytext=(0.05, 0.35),
                      arrowprops=dict(arrowstyle='<->', color='black', lw=1.5),
                      xycoords='axes fraction')
    ax_corr3.text(0.5, 0.65, f'Corr: {corr_ts2vec:.3f}', transform=ax_corr3.transAxes,
                  fontsize=11, verticalalignment='center', horizontalalignment='center',
                  fontweight='bold')
    
    # Add shared x-axis label at the bottom
    ax_ts2vec_noiseless.set_xlabel('Time Step', fontsize=11)
    ax_ts2vec_noisy.set_xlabel('Time Step', fontsize=11)
    
    # Set x-axis ticks for bottom row
    xtick_positions = np.linspace(0, L, 11)
    xtick_labels = [str(int(x)) for x in np.linspace(0, L, 11)]
    ax_ts2vec_noiseless.set_xticks(xtick_positions)
    ax_ts2vec_noiseless.set_xticklabels(xtick_labels, fontsize=9)
    ax_ts2vec_noisy.set_xticks(xtick_positions)
    ax_ts2vec_noisy.set_xticklabels(xtick_labels, fontsize=9)
    
    # Adjust layout
    plt.tight_layout()
    
    # Save the figure
    os.makedirs(PATH_CONFIG["vis_dir"], exist_ok=True)
    save_path = os.path.join(PATH_CONFIG["vis_dir"], "figure2.png")
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    
    print(f"Generated Figure 2: Noise Robustness Experiment")
    print(f"Signal correlation: {corr_signals:.3f}")
    print(f"CoInception correlation: {corr_coinception:.3f}")
    print(f"TS2Vec correlation: {corr_ts2vec:.3f}")
    print(f"Figure saved to: {save_path}")


if __name__ == "__main__":
    os.makedirs(PATH_CONFIG["vis_dir"], exist_ok=True)
    generate_figure2()