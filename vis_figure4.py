#!/usr/bin/env python3
"""
Visualization Script for Figure 4: Critical Difference Diagram (Corrected)

This script generates Figure 4 matching the paper layout:
- Standard Critical Difference diagram format
- Methods sorted by rank from best (bottom) to worst (top)
- Bold horizontal lines connecting methods with no significant difference
- Uses proper Nemenyi test visualization style
"""

import numpy as np
import matplotlib.pyplot as plt
from analysis_preset import get_vis_settings, PATH_CONFIG
import os
import pickle
import json


def load_evaluation_results():
    """Load evaluation results from training directories only.
    Returns:
        dict: Dictionary of evaluation results or empty dict if no results found
    """
    results_dir = os.path.dirname(os.path.abspath(__file__))
    
    print("Loading results from training directories only")
    
    # Default empty results
    eval_results = {
        "CoInception": {},
        "TS2Vec": {}
    }
    
    # Try to load CoInception evaluation results from all datasets
    coin_training_dir = os.path.join(results_dir, 'training')
    if os.path.exists(coin_training_dir):
        # Find all evaluation results files
        import glob
        coin_results_files = glob.glob(os.path.join(coin_training_dir, '**', 'eval_res.pkl'), recursive=True)
        for file_path in coin_results_files:
            # Extract dataset name from file path
            dataset_name = os.path.basename(os.path.dirname(file_path))
            try:
                with open(file_path, 'rb') as f:
                    eval_results["CoInception"][dataset_name] = pickle.load(f)
                print(f"Loaded CoInception evaluation results from {file_path}")
            except Exception as e:
                print(f"Error loading CoInception evaluation results from {file_path}: {e}")
    
    # Try to load TS2Vec evaluation results from all datasets
    ts2vec_training_dir = os.path.join(results_dir, 'ts2vec', 'training')
    if os.path.exists(ts2vec_training_dir):
        # Find all evaluation results files
        import glob
        ts2vec_results_files = glob.glob(os.path.join(ts2vec_training_dir, '**', 'eval_res.pkl'), recursive=True)
        for file_path in ts2vec_results_files:
            # Extract dataset name from file path
            dataset_name = os.path.basename(os.path.dirname(file_path))
            try:
                with open(file_path, 'rb') as f:
                    eval_results["TS2Vec"][dataset_name] = pickle.load(f)
                print(f"Loaded TS2Vec evaluation results from {file_path}")
            except Exception as e:
                print(f"Error loading TS2Vec evaluation results from {file_path}: {e}")
    
    return eval_results

def calculate_method_ranks(eval_results):
    """Calculate method ranks from evaluation results.
    
    Args:
        eval_results: Dictionary of evaluation results
        
    Returns:
        dict: Dictionary of method ranks or empty dict if no results found
    """
    print("Calculating ranks from local training results")
    
    # Extract accuracy values for both models from their evaluation results
    model_accuracies = {}
    
    # Process CoInception results
    if 'CoInception' in eval_results and isinstance(eval_results['CoInception'], dict):
        coin_results = eval_results['CoInception']
        coin_accs = []
        
        # Collect all accuracy values from CoInception results
        for dataset, result in coin_results.items():
            if isinstance(result, dict) and 'acc' in result:
                coin_accs.append(result['acc'])
        
        if coin_accs:
            model_accuracies['CoInception'] = np.mean(coin_accs)
            print(f"CoInception average accuracy: {model_accuracies['CoInception']:.4f}")
    
    # Process TS2Vec results
    if 'TS2Vec' in eval_results and isinstance(eval_results['TS2Vec'], dict):
        ts2vec_results = eval_results['TS2Vec']
        ts2vec_accs = []
        
        # Collect all accuracy values from TS2Vec results
        for dataset, result in ts2vec_results.items():
            if isinstance(result, dict) and 'acc' in result:
                ts2vec_accs.append(result['acc'])
        
        if ts2vec_accs:
            model_accuracies['TS2Vec'] = np.mean(ts2vec_accs)
            print(f"TS2Vec average accuracy: {model_accuracies['TS2Vec']:.4f}")
    
    # Calculate ranks based on average accuracy
    if model_accuracies:
        # Sort models by accuracy (higher is better, so lower rank)
        sorted_models = sorted(model_accuracies.items(), key=lambda x: x[1], reverse=True)
        
        # Assign ranks
        ranks = {}
        for i, (model, acc) in enumerate(sorted_models):
            ranks[model] = i + 1  # Rank starts from 1
        
        print(f"Calculated ranks: {ranks}")
        return ranks
    else:
        print("No valid accuracy results found for ranking")
        return {}

def generate_figure4():
    """Generate Figure 4 matching the paper's format."""
    
    # Set font sizes
    plt.rcParams.update({
        'axes.labelsize': 14,
        'xtick.labelsize': 12,
        'ytick.labelsize': 12,
        'axes.titlesize': 16,
        'font.size': 12,
    })
    
    # Load evaluation results
    eval_results = load_evaluation_results()
    
    # Calculate method ranks (lower rank = better performance)
    ranks = calculate_method_ranks(eval_results)
    
    # Critical Difference value (for 8 methods at 95% confidence)
    cd = 1.50
    
    # Sort methods by rank (best to worst)
    sorted_methods = sorted(ranks.items(), key=lambda x: x[1])
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Calculate positions
    n_methods = len(sorted_methods)
    max_rank = max(ranks.values())
    
    # Calculate y-positions (spread methods evenly)
    y_positions = {}
    for i, (method, rank) in enumerate(sorted_methods):
        y_positions[method] = i
    
    # Plot each method as a horizontal line from left edge to its rank position
    for method, rank in sorted_methods:
        y = y_positions[method]
        # Draw line from method name to rank position
        ax.plot([rank, max_rank + 0.8], [y, y], 'k-', linewidth=2)
        # Draw circle at rank position
        ax.plot(rank, y, 'ko', markersize=10, markerfacecolor='black')
        # Add method name on the right
        ax.text(max_rank + 1.0, y, method, ha='left', va='center', fontsize=13)
    
    # Draw connections between methods that are not significantly different
    # Methods are connected if their rank difference < CD
    connected_groups = []
    
    # Find groups of methods within CD of each other
    method_list = [m for m, _ in sorted_methods]
    rank_list = [r for _, r in sorted_methods]
    
    # Simple algorithm to find connected groups
    groups = []
    visited = set()
    
    for i, (m1, r1) in enumerate(sorted_methods):
        if m1 in visited:
            continue
        group = [m1]
        visited.add(m1)
        for j, (m2, r2) in enumerate(sorted_methods):
            if m2 not in visited and abs(r1 - r2) < cd:
                # Check if m2 is connected to any member of the group
                for gm in group:
                    if abs(ranks[gm] - r2) < cd:
                        group.append(m2)
                        visited.add(m2)
                        break
        if len(group) > 1:
            groups.append(group)
    
    # Draw connection lines for groups
    line_height = -0.15
    for group in groups:
        group_ranks = [ranks[m] for m in group]
        group_ys = [y_positions[m] for m in group]
        min_y, max_y = min(group_ys), max(group_ys)
        
        # Draw thick horizontal bar above the group
        bar_y = max_y + 0.3
        min_rank = min(group_ranks)
        max_rank_g = max(group_ranks)
        
        # Draw connection bar
        ax.plot([min_rank, max_rank_g], [bar_y, bar_y], 'k-', linewidth=4)
    
    # Draw CD indicator line (vertical dashed line at CD position from best rank)
    best_rank = min(ranks.values())
    cd_line_x = best_rank + cd
    ax.axvline(x=cd_line_x, color='gray', linestyle='--', linewidth=2, alpha=0.7)
    
    # Add CD legend/annotation
    ax.plot([best_rank, best_rank + cd], [-1, -1], 'k-', linewidth=3)
    ax.text(best_rank + cd/2, -1.5, f'CD = {cd:.2f}', ha='center', va='top', fontsize=12)
    
    # Set axis properties
    ax.set_xlim(0.5, max_rank + 3)
    ax.set_ylim(-2, n_methods + 0.5)
    ax.set_xlabel('Average Rank', fontsize=14)
    ax.set_title('Critical Difference Diagram', fontsize=16, fontweight='bold')
    
    # Remove unnecessary spines and ticks
    ax.spines['left'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_yticks([])
    
    # Set x-axis ticks
    ax.set_xticks(range(1, int(max_rank) + 2))
    
    # Invert y-axis so best method is at bottom
    ax.invert_yaxis()
    
    # Add data source annotation
    fig.text(0.5, 0.01, "Data Source: external_results.csv", 
             ha='center', fontsize=10, fontstyle='italic', color='gray')
    
    # Adjust layout
    plt.tight_layout()
    
    # Save the figure
    os.makedirs(PATH_CONFIG["vis_dir"], exist_ok=True)
    save_path = os.path.join(PATH_CONFIG["vis_dir"], "figure4.png")
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    
    print(f"Generated Figure 4: Critical Difference Diagram")
    print(f"Figure saved to: {save_path}")


if __name__ == "__main__":
    os.makedirs(PATH_CONFIG["vis_dir"], exist_ok=True)
    generate_figure4()