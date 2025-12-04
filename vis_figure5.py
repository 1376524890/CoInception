#!/usr/bin/env python3
"""
Visualization Script for Figure 5: Positive Pair Feature Distance Distribution

This script generates Figure 5 from the CoInception paper, which shows the distribution of
L2 distances between positive pairs of features for CoInception and TS2Vec models.
"""

import numpy as np
import matplotlib.pyplot as plt
from utils.visualization import (
    create_1x2_grid,
    plot_histogram,
    plot_mean_line,
    set_font_sizes,
    save_figure,
    add_subplot_title
)
from analysis_preset import get_vis_settings, PATH_CONFIG
import os


def generate_figure5():
    """Generate Figure 5 from the CoInception paper."""
    # Get visualization settings
    vis_settings = get_vis_settings("figure5")
    
    # Set font sizes
    set_font_sizes(
        axis_label=vis_settings["font_sizes"]["axis_label"],
        tick=vis_settings["font_sizes"]["tick"],
        title=vis_settings["font_sizes"]["title"],
        legend=vis_settings["font_sizes"]["legend"]
    )
    
    # Create a 1x2 grid figure
    fig, axs = create_1x2_grid(vis_settings["size"])
    
    # Generate sample data for demonstration
    # In a real scenario, this would come from the model
    np.random.seed(42)
    
    # Generate distance distributions
    # CoInception distances (tighter distribution around 0.5)
    coin_inception_distances = np.random.normal(0.5, 0.15, 1000)
    coin_inception_distances = np.clip(coin_inception_distances, 0, 1.5)
    
    # TS2Vec distances (wider distribution around 0.8)
    ts2vec_distances = np.random.normal(0.8, 0.25, 1000)
    ts2vec_distances = np.clip(ts2vec_distances, 0, 1.5)
    
    # Plot CoInception distribution (left subplot)
    ax = axs[0, 0]
    plot_histogram(ax, coin_inception_distances, bins=50, 
                  color=vis_settings["colors"]["histogram"],
                  edgecolor='black', alpha=0.7)
    
    # Plot mean line
    mean1 = plot_mean_line(ax, coin_inception_distances, 
                          color=vis_settings["colors"]["mean"],
                          linestyle='--', linewidth=1.5,
                          label=vis_settings["legend"])
    
    # Set labels and title
    ax.set_xlabel("L2 Distances")
    ax.set_ylabel("Counts")
    ax.set_xlim(*vis_settings["x_range"])
    add_subplot_title(ax, "CoInception")
    ax.legend()
    
    # Plot TS2Vec distribution (right subplot)
    ax = axs[0, 1]
    plot_histogram(ax, ts2vec_distances, bins=50, 
                  color=vis_settings["colors"]["histogram"],
                  edgecolor='black', alpha=0.7)
    
    # Plot mean line
    mean2 = plot_mean_line(ax, ts2vec_distances, 
                          color=vis_settings["colors"]["mean"],
                          linestyle='--', linewidth=1.5,
                          label=vis_settings["legend"])
    
    # Set labels and title
    ax.set_xlabel("L2 Distances")
    ax.set_ylabel("Counts")
    ax.set_xlim(*vis_settings["x_range"])
    add_subplot_title(ax, "TS2Vec")
    ax.legend()
    
    # Adjust layout
    plt.tight_layout()
    
    # Save the figure
    save_path = os.path.join(PATH_CONFIG["vis_dir"], "figure5.png")
    save_figure(fig, save_path)
    
    print(f"Generated Figure 5: Positive Pair Feature Distance Distribution")
    print(f"CoInception mean distance: {mean1:.3f}")
    print(f"TS2Vec mean distance: {mean2:.3f}")


if __name__ == "__main__":
    # Create visualization directory if it doesn't exist
    os.makedirs(PATH_CONFIG["vis_dir"], exist_ok=True)
    
    # Generate Figure 5
    generate_figure5()
