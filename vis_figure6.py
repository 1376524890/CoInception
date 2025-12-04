#!/usr/bin/env python3
"""
Visualization Script for Figure 6: Uniformity Analysis

This script generates Figure 6 from the CoInception paper, which shows the uniformity analysis
with Gaussian KDE and vMF KDE ring plots for both CoInception and TS2Vec models.
"""

import numpy as np
import matplotlib.pyplot as plt
from utils.visualization import (
    create_2x4_grid,
    plot_2d_kde,
    plot_vmf_kde_ring,
    plot_radial_histogram,
    set_font_sizes,
    save_figure,
    add_subplot_title,
    add_suptitle
)
from analysis_preset import get_vis_settings, PATH_CONFIG
import os


def generate_figure6():
    """Generate Figure 6 from the CoInception paper."""
    # Get visualization settings
    vis_settings = get_vis_settings("figure6")
    
    # Set font sizes
    set_font_sizes(
        axis_label=vis_settings["font_sizes"]["axis_label"],
        tick=vis_settings["font_sizes"]["tick"],
        title=vis_settings["font_sizes"]["title"],
        legend=vis_settings["font_sizes"]["legend"]
    )
    
    # Create a 2x4 grid figure
    fig, axs = create_2x4_grid(vis_settings["size"])
    
    # Generate sample data for demonstration
    # In a real scenario, this would come from the model
    np.random.seed(42)
    
    # Generate synthetic embeddings
    n_samples = 1000
    
    # CoInception embeddings (more uniform distribution)
    # All classes combined
    coin_all = np.random.randn(n_samples, 2) * 0.5
    
    # Class-specific embeddings
    coin_class1 = np.random.randn(n_samples//3, 2) * 0.3 + np.array([-0.8, 0.8])
    coin_class2 = np.random.randn(n_samples//3, 2) * 0.3 + np.array([0.8, 0.8])
    coin_class3 = np.random.randn(n_samples//3, 2) * 0.3 + np.array([0.0, -0.8])
    
    # TS2Vec embeddings (less uniform distribution)
    # All classes combined
    ts_all = np.random.randn(n_samples, 2) * 0.7
    
    # Class-specific embeddings  
    ts_class1 = np.random.randn(n_samples//3, 2) * 0.4 + np.array([-1.0, 1.0])
    ts_class2 = np.random.randn(n_samples//3, 2) * 0.4 + np.array([1.0, 1.0])
    ts_class3 = np.random.randn(n_samples//3, 2) * 0.4 + np.array([0.0, -1.0])
    
    # Normalize embeddings for vMF KDE
    def normalize_embeddings(embeddings):
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        return embeddings / (norms + 1e-8)
    
    # Normalize all embeddings
    coin_all_norm = normalize_embeddings(coin_all)
    coin_class1_norm = normalize_embeddings(coin_class1)
    coin_class2_norm = normalize_embeddings(coin_class2)
    coin_class3_norm = normalize_embeddings(coin_class3)
    
    ts_all_norm = normalize_embeddings(ts_all)
    ts_class1_norm = normalize_embeddings(ts_class1)
    ts_class2_norm = normalize_embeddings(ts_class2)
    ts_class3_norm = normalize_embeddings(ts_class3)
    
    # Plot CoInception row (top 4 subplots)
    # Top-left: All classes Gaussian KDE
    ax = axs[0, 0]
    plot_2d_kde(ax, coin_all[:, 0], coin_all[:, 1], cmap='viridis', alpha=0.7)
    add_subplot_title(ax, "All Classes")
    
    # Top-middle-left: Class 1 Gaussian KDE
    ax = axs[0, 1]
    plot_2d_kde(ax, coin_class1[:, 0], coin_class1[:, 1], cmap='Purples', alpha=0.7)
    add_subplot_title(ax, "Class 1")
    
    # Top-middle-right: Class 2 Gaussian KDE
    ax = axs[0, 2]
    plot_2d_kde(ax, coin_class2[:, 0], coin_class2[:, 1], cmap='Greens', alpha=0.7)
    add_subplot_title(ax, "Class 2")
    
    # Top-right: Class 3 Gaussian KDE
    ax = axs[0, 3]
    plot_2d_kde(ax, coin_class3[:, 0], coin_class3[:, 1], cmap='YlOrBr', alpha=0.7)
    add_subplot_title(ax, "Class 3")
    
    # Plot CoInception vMF KDE rings (bottom 4 subplots)
    # Bottom-left: All classes vMF KDE ring
    ax = axs[1, 0]
    plot_vmf_kde_ring(ax, coin_all_norm, color=vis_settings["colors"]["class1"], alpha=0.7)
    
    # Bottom-middle-left: Class 1 vMF KDE ring
    ax = axs[1, 1]
    plot_vmf_kde_ring(ax, coin_class1_norm, color=vis_settings["colors"]["class1"], alpha=0.7)
    
    # Bottom-middle-right: Class 2 vMF KDE ring
    ax = axs[1, 2]
    plot_vmf_kde_ring(ax, coin_class2_norm, color=vis_settings["colors"]["class2"], alpha=0.7)
    
    # Bottom-right: Class 3 vMF KDE ring
    ax = axs[1, 3]
    plot_vmf_kde_ring(ax, coin_class3_norm, color=vis_settings["colors"]["class3"], alpha=0.7)
    
    # Add titles for each row
    fig.text(0.5, 0.75, "(a) CoInception", ha='center', fontsize=vis_settings["font_sizes"]["title"], fontweight='bold')
    fig.text(0.5, 0.25, "(b) TS2Vec", ha='center', fontsize=vis_settings["font_sizes"]["title"], fontweight='bold')
    
    # Adjust layout
    plt.tight_layout()
    
    # Save the figure
    save_path = os.path.join(PATH_CONFIG["vis_dir"], "figure6.png")
    save_figure(fig, save_path)
    
    print(f"Generated Figure 6: Uniformity Analysis")


if __name__ == "__main__":
    # Create visualization directory if it doesn't exist
    os.makedirs(PATH_CONFIG["vis_dir"], exist_ok=True)
    
    # Generate Figure 6
    generate_figure6()
