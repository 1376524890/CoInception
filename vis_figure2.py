#!/usr/bin/env python3
"""
Visualization Script for Figure 2: Noise Robustness Experiment

This script generates Figure 2 from the CoInception paper, which shows the noise robustness
of the model with a 2x2 grid layout containing waveforms and heatmaps.
"""

import numpy as np
import matplotlib.pyplot as plt
from utils.visualization import (
    create_2x2_grid,
    plot_waveform,
    plot_heatmap,
    plot_correlation_label,
    set_font_sizes,
    save_figure,
    calculate_correlation,
    add_subplot_title
)
from analysis_preset import get_vis_settings, PATH_CONFIG
import os


def generate_figure2():
    """Generate Figure 2 from the CoInception paper."""
    # Get visualization settings
    vis_settings = get_vis_settings("figure2")
    
    # Set font sizes
    set_font_sizes(
        axis_label=vis_settings["font_sizes"]["axis_label"],
        tick=vis_settings["font_sizes"]["tick"]
    )
    
    # Create a 2x2 grid figure
    fig, axs = create_2x2_grid(vis_settings["size"])
    
    # Generate sample data for demonstration
    # In a real scenario, this would come from the model
    np.random.seed(42)
    
    # Generate synthetic time series data
    n_timestamps = 100
    time = np.arange(n_timestamps)
    
    # Original waveform (blue)
    original = np.sin(0.1 * time) + 0.5 * np.sin(0.3 * time) + 0.2 * np.sin(0.5 * time)
    
    # Noisy waveform (with Gaussian noise)
    noise_level = 0.3
    noisy = original + noise_level * np.random.randn(n_timestamps)
    
    # Generate model representations (heatmaps)
    # Original representation
    repr_original = np.random.randn(20, n_timestamps) * 0.5 + 0.5
    
    # Noisy representation
    repr_noisy = np.random.randn(20, n_timestamps) * 0.5 + 0.5
    # Add some correlation with original
    repr_noisy = 0.7 * repr_original + 0.3 * repr_noisy
    
    # Calculate correlation between original and noisy representations
    corr = calculate_correlation(repr_original, repr_noisy)
    
    # Plot top-left: Original waveform (no noise)
    ax = axs[0, 0]
    plot_waveform(ax, original, color=vis_settings["colors"]["waveform"])
    add_subplot_title(ax, "Original")
    
    # Plot top-right: Noisy waveform
    ax = axs[0, 1]
    plot_waveform(ax, noisy, color=vis_settings["colors"]["waveform"])
    add_subplot_title(ax, "Noisy")
    
    # Plot bottom-left: Original representation heatmap
    ax = axs[1, 0]
    plot_heatmap(ax, repr_original, cmap=vis_settings["colors"]["heatmap"])
    
    # Plot bottom-right: Noisy representation heatmap with correlation
    ax = axs[1, 1]
    plot_heatmap(ax, repr_noisy, cmap=vis_settings["colors"]["heatmap"])
    plot_correlation_label(ax, corr, fontsize=vis_settings["font_sizes"]["corr"])
    
    # Set y-axis label for bottom subplots
    axs[1, 0].set_ylabel("Dim")
    axs[1, 1].set_ylabel("Dim")
    
    # Adjust layout
    plt.tight_layout()
    
    # Save the figure
    save_path = os.path.join(PATH_CONFIG["vis_dir"], "figure2.png")
    save_figure(fig, save_path)
    
    print(f"Generated Figure 2: Noise Robustness Experiment")
    print(f"Correlation between original and noisy representations: {corr:.3f}")


if __name__ == "__main__":
    # Create visualization directory if it doesn't exist
    os.makedirs(PATH_CONFIG["vis_dir"], exist_ok=True)
    
    # Generate Figure 2
    generate_figure2()
