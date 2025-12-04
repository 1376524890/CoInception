#!/usr/bin/env python3
"""
Main Visualization Script for CoInception

This script generates all figures from the CoInception paper by calling the individual visualization scripts.
"""

import os
import sys
import argparse
from analysis_preset import PATH_CONFIG


def generate_all_figures():
    """Generate all figures from the CoInception paper."""
    print("Generating all CoInception figures...")
    print("=" * 50)
    
    # Create visualization directory if it doesn't exist
    os.makedirs(PATH_CONFIG["vis_dir"], exist_ok=True)
    
    # Import and run individual visualization scripts
    
    # Figure 2: Noise Robustness Experiment
    print("\nGenerating Figure 2: Noise Robustness Experiment")
    from vis_figure2 import generate_figure2
    generate_figure2()
    
    # Figure 4: Critical Difference Diagram
    print("\nGenerating Figure 4: Critical Difference Diagram")
    from vis_figure4 import generate_figure4
    generate_figure4()
    
    # Figure 5: Positive Pair Feature Distance Distribution
    print("\nGenerating Figure 5: Positive Pair Feature Distance Distribution")
    from vis_figure5 import generate_figure5
    generate_figure5()
    
    # Figure 6: Uniformity Analysis
    print("\nGenerating Figure 6: Uniformity Analysis")
    from vis_figure6 import generate_figure6
    generate_figure6()
    
    # Figures 7-8: Noise Ratio Analysis
    print("\nGenerating Figures 7-8: Noise Ratio Analysis")
    from vis_figure7_8 import generate_figure7_8
    generate_figure7_8()
    
    print("\n" + "=" * 50)
    print("All figures generated successfully!")
    print(f"Figures saved to: {PATH_CONFIG['vis_dir']}")


def generate_specific_figure(figure_name):
    """Generate a specific figure from the CoInception paper.
    
    Args:
        figure_name (str): Name of the figure to generate (e.g., "figure2", "figure4", etc.).
    """
    print(f"Generating {figure_name}...")
    print("=" * 50)
    
    # Create visualization directory if it doesn't exist
    os.makedirs(PATH_CONFIG["vis_dir"], exist_ok=True)
    
    if figure_name == "figure2":
        from vis_figure2 import generate_figure2
        generate_figure2()
    elif figure_name == "figure4":
        from vis_figure4 import generate_figure4
        generate_figure4()
    elif figure_name == "figure5":
        from vis_figure5 import generate_figure5
        generate_figure5()
    elif figure_name == "figure6":
        from vis_figure6 import generate_figure6
        generate_figure6()
    elif figure_name in ["figure7", "figure8", "figure7_8"]:
        from vis_figure7_8 import generate_figure7_8
        generate_figure7_8()
    else:
        print(f"Error: Unknown figure name '{figure_name}'")
        print("Available figures: figure2, figure4, figure5, figure6, figure7, figure8, figure7_8")
        sys.exit(1)
    
    print("\n" + "=" * 50)
    print(f"{figure_name} generated successfully!")
    print(f"Figure saved to: {PATH_CONFIG['vis_dir']}")


def main():
    """Main function to parse arguments and generate figures."""
    parser = argparse.ArgumentParser(description="Generate CoInception paper figures")
    parser.add_argument('--all', action='store_true', help='Generate all figures')
    parser.add_argument('--figure', type=str, help='Generate a specific figure (e.g., figure2, figure4, etc.)')
    
    args = parser.parse_args()
    
    if args.all:
        generate_all_figures()
    elif args.figure:
        generate_specific_figure(args.figure)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
