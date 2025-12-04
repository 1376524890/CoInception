#!/usr/bin/env python3
"""
Visualization Script for Figures 7-8: Noise Ratio Analysis

This script generates Figures 7 and 8 from the CoInception paper, which show the
noise ratio analysis results in table form and hexagonal plot form.
"""

import numpy as np
import matplotlib.pyplot as plt
from utils.visualization import (
    create_figure,
    plot_hexagon_3d,
    set_font_sizes,
    save_figure,
    add_subplot_title
)
from analysis_preset import get_vis_settings, PATH_CONFIG, TABLE_SETTINGS
import os


def generate_figure7():
    """Generate Figure 7 from the CoInception paper."""
    # Get visualization settings
    vis_settings = get_vis_settings("figure7_8")
    table_settings = TABLE_SETTINGS
    
    # Set font sizes
    set_font_sizes(
        axis_label=vis_settings["font_sizes"]["axis_label"],
        tick=vis_settings["font_sizes"]["tick"],
        title=vis_settings["font_sizes"]["title"]
    )
    
    # Create figure for table
    fig = create_figure(vis_settings["figure7_size"])
    
    # Sample data for the table
    # In a real scenario, this would come from evaluation results
    noise_levels = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
    coin_inception_acc = [0.92, 0.89, 0.85, 0.81, 0.76, 0.70]
    ts2vec_acc = [0.88, 0.83, 0.77, 0.70, 0.63, 0.55]
    
    # Create table data
    table_data = [
        ["Noise Level", "CoInception", "TS2Vec"],
        *[[f"{nl:.1f}", f"{ci:.2f}", f"{ts:.2f}"] for nl, ci, ts in zip(noise_levels, coin_inception_acc, ts2vec_acc)]
    ]
    
    # Create table
    ax = fig.add_subplot(111)
    ax.axis('tight')
    ax.axis('off')
    
    # Create the table
    table = ax.table(
        cellText=table_data,
        colLabels=None,
        cellLoc='center',
        loc='center'
    )
    
    # Set table properties to match paper requirements
    table.auto_set_font_size(False)
    table.set_fontsize(table_settings["font_size"])
    table.scale(1.2, 1.5)
    
    # Add title
    ax.set_title("Noise Robustness Comparison", fontsize=vis_settings["font_sizes"]["title"])
    
    # Adjust layout
    plt.tight_layout()
    
    # Save the figure
    save_path = os.path.join(PATH_CONFIG["vis_dir"], "figure7.png")
    save_figure(fig, save_path)
    
    print(f"Generated Figure 7: Noise Robustness Table")


def generate_figure8():
    """Generate Figure 8 from the CoInception paper."""
    # Get visualization settings
    vis_settings = get_vis_settings("figure7_8")
    
    # Set font sizes
    set_font_sizes(
        axis_label=vis_settings["font_sizes"]["axis_label"],
        tick=vis_settings["font_sizes"]["tick"],
        title=vis_settings["font_sizes"]["title"]
    )
    
    # Create figure for hexagonal plots
    fig = create_figure(vis_settings["figure8_size"])
    
    # Create 1x2 grid for hexagonal plots
    ax1 = fig.add_subplot(121, projection='3d')
    ax2 = fig.add_subplot(122, projection='3d')
    
    # Generate sample data for hexagonal plots
    # In a real scenario, this would come from evaluation results
    np.random.seed(42)
    
    # CoInception hexagonal data
    n_points = 100
    x1 = np.random.rand(n_points) * 5
    y1 = np.random.rand(n_points) * 5
    z1 = np.random.rand(n_points) * 2 + 3  # Higher performance
    
    # TS2Vec hexagonal data
    x2 = np.random.rand(n_points) * 5
    y2 = np.random.rand(n_points) * 5  
    z2 = np.random.rand(n_points) * 2 + 1  # Lower performance
    
    # Plot CoInception hexagonal data
    plot_hexagon_3d(ax1, x1, y1, z1, color=vis_settings["colors"]["coinception"], alpha=0.7)
    add_subplot_title(ax1, "CoInception")
    ax1.set_xlabel("Noise Ratio")
    ax1.set_ylabel("Dataset")
    ax1.set_zlabel("Accuracy")
    
    # Plot TS2Vec hexagonal data
    plot_hexagon_3d(ax2, x2, y2, z2, color=vis_settings["colors"]["ts2vec"], alpha=0.7)
    add_subplot_title(ax2, "TS2Vec")
    ax2.set_xlabel("Noise Ratio")
    ax2.set_ylabel("Dataset")
    ax2.set_zlabel("Accuracy")
    
    # Adjust layout
    plt.tight_layout()
    
    # Save the figure
    save_path = os.path.join(PATH_CONFIG["vis_dir"], "figure8.png")
    save_figure(fig, save_path)
    
    print(f"Generated Figure 8: Hexagonal Noise Analysis")


def generate_figure7_8():
    """Generate both Figure 7 and Figure 8 from the CoInception paper."""
    generate_figure7()
    generate_figure8()


if __name__ == "__main__":
    # Create visualization directory if it doesn't exist
    os.makedirs(PATH_CONFIG["vis_dir"], exist_ok=True)
    
    # Generate Figures 7 and 8
    generate_figure7_8()
