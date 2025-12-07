#!/usr/bin/env python3
"""
Main Visualization Script for CoInception (Corrected)

This script generates all corrected figures from the CoInception paper.
"""

import os
import sys
from analysis_preset import PATH_CONFIG


def generate_all_figures():
    """Generate all figures from the CoInception paper."""
    print("=" * 60)
    print("Generating all CoInception figures")
    print("=" * 60)
    
    # Create visualization directory if it doesn't exist
    os.makedirs(PATH_CONFIG["vis_dir"], exist_ok=True)
    
    # Figure 2: Noise Robustness Experiment
    print("\n[1/5] Generating Figure 2: Noise Robustness Experiment")
    print("-" * 40)
    from vis_figure2 import generate_figure2
    generate_figure2()
    
    # Figure 4: Critical Difference Diagram
    print("\n[2/5] Generating Figure 4: Critical Difference Diagram")
    print("-" * 40)
    from vis_figure4 import generate_figure4
    generate_figure4()
    
    # Figure 5: Positive Pair Feature Distance Distribution
    print("\n[3/5] Generating Figure 5: Positive Pair Feature Distance")
    print("-" * 40)
    from vis_figure5 import generate_figure5
    generate_figure5()
    
    # Figure 6: Uniformity Analysis
    print("\n[4/5] Generating Figure 6: Uniformity Analysis")
    print("-" * 40)
    from vis_figure6 import generate_figure6
    generate_figure6()
    
    # Figures 7-8: Noise Ratio Analysis
    print("\n[5/5] Generating Figures 7-8: Noise Ratio Analysis")
    print("-" * 40)
    from vis_figure7_8 import generate_figure7_8
    generate_figure7_8()
    
    print("\n" + "=" * 60)
    print("All figures generated successfully!")
    print(f"Figures saved to: {PATH_CONFIG['vis_dir']}")
    print("=" * 60)


if __name__ == "__main__":
    generate_all_figures()