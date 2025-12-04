#!/usr/bin/env python3
"""
Visualization Script for Figure 4: Critical Difference Diagram

This script generates Figure 4 from the CoInception paper, which shows the critical difference
of classification performance across different models.
"""

import numpy as np
import matplotlib.pyplot as plt
from utils.visualization import (
    create_figure,
    plot_critical_difference,
    set_font_sizes,
    save_figure,
    add_subplot_title
)
from analysis_preset import get_vis_settings, PATH_CONFIG
import os


def generate_figure4():
    """Generate Figure 4 from the CoInception paper."""
    # Get visualization settings
    vis_settings = get_vis_settings("figure4")
    
    # Set font sizes
    set_font_sizes(
        axis_label=vis_settings["font_sizes"]["classifier"],
        tick=vis_settings["font_sizes"]["tick"]
    )
    
    # Create figure
    fig = create_figure(vis_settings["size"])
    ax = fig.add_subplot(111)
    
    # Sample data for critical difference diagram
    # In a real scenario, this would come from evaluation results
    ranks = {
        "CoInception": 1.2,
        "TS2Vec": 1.8,
        "InceptionTime": 2.5,
        "ResNet": 3.1,
        "Transformer": 3.8,
        "LSTM": 4.2,
        "GRU": 4.6,
        "CNN": 5.1
    }
    
    # Critical difference value
    cd = 1.5
    
    # Plot critical difference diagram
    plot_critical_difference(ax, ranks, cd)
    
    # Adjust layout
    plt.tight_layout()
    
    # Save the figure
    save_path = os.path.join(PATH_CONFIG["vis_dir"], "figure4.png")
    save_figure(fig, save_path)
    
    print(f"Generated Figure 4: Critical Difference Diagram")


if __name__ == "__main__":
    # Create visualization directory if it doesn't exist
    os.makedirs(PATH_CONFIG["vis_dir"], exist_ok=True)
    
    # Generate Figure 4
    generate_figure4()
