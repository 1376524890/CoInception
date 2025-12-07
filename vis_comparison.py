#!/usr/bin/env python3
"""
Visualization Script for Comprehensive Comparison between CoInception and TS2Vec

This script generates comprehensive comparison charts between CoInception and TS2Vec models.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from analysis_preset import get_vis_settings, PATH_CONFIG
import os
import pickle
import json
import glob


def load_model_results():
    """Load evaluation results from both CoInception and TS2Vec training directories only.
    Returns:
        dict: Dictionary of evaluation results for both models
    """
    results_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Default empty results
    eval_results = {
        "CoInception": {},
        "TS2Vec": {}
    }
    
    print("Loading results from training directories only")
    
    # Try to load CoInception evaluation results from all datasets
    coin_training_dir = os.path.join(results_dir, 'training')
    if os.path.exists(coin_training_dir):
        # Find all evaluation results files
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


def generate_performance_comparison(eval_results):
    """Generate performance comparison charts between CoInception and TS2Vec.
    
    Args:
        eval_results (dict): Dictionary of evaluation results for both models
    """
    
    # Set font sizes
    plt.rcParams.update({
        'axes.labelsize': 12,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'axes.titlesize': 14,
        'font.size': 11,
    })
    
    # Extract datasets with results for both models
    coin_datasets = set(eval_results["CoInception"].keys())
    ts2vec_datasets = set(eval_results["TS2Vec"].keys())
    common_datasets = coin_datasets.intersection(ts2vec_datasets)
    
    # If no common datasets, use all available datasets
    if not common_datasets:
        common_datasets = coin_datasets.union(ts2vec_datasets)
    
    # Sort datasets for consistency
    common_datasets = sorted(common_datasets)
    
    model_names = ["CoInception", "TS2Vec"]
    model_colors = {"CoInception": "#E74C3C", "TS2Vec": "#27AE60"}
    
    # Create figure with multiple subplots
    fig, axs = plt.subplots(2, 2, figsize=(14, 10))
    axs = axs.flatten()
    
    # Define metrics to plot
    metrics = ["accuracy", "rank"]
    
    # Plot each metric
    for i, (metric_name, ax) in enumerate(zip(metrics, axs)):
        # Prepare data for plotting
        coin_values = []
        ts2vec_values = []
        valid_datasets = []
        
        for dataset in common_datasets:
            # Get CoInception value if available
            coin_val = None
            if dataset in eval_results["CoInception"]:
                coin_res = eval_results["CoInception"][dataset]
                if isinstance(coin_res, dict):
                    if metric_name in coin_res:
                        coin_val = coin_res[metric_name]
                    elif metric_name == "accuracy" and "acc" in coin_res:
                        coin_val = coin_res["acc"]
            
            # Get TS2Vec value if available
            ts2vec_val = None
            if dataset in eval_results["TS2Vec"]:
                ts2vec_res = eval_results["TS2Vec"][dataset]
                if isinstance(ts2vec_res, dict):
                    if metric_name in ts2vec_res:
                        ts2vec_val = ts2vec_res[metric_name]
                    elif metric_name == "accuracy" and "acc" in ts2vec_res:
                        ts2vec_val = ts2vec_res["acc"]
            
            # Only include dataset if both models have the metric
            if coin_val is not None and ts2vec_val is not None:
                coin_values.append(coin_val)
                ts2vec_values.append(ts2vec_val)
                valid_datasets.append(dataset)
        
        # If no data for this metric, skip
        if not valid_datasets:
            ax.set_title(f"{metric_name.capitalize()} - No Data Available")
            ax.axis('off')
            continue
        
        # Set up bar positions
        x = np.arange(len(valid_datasets))
        width = 0.35
        
        # Plot bars
        ax.bar(x - width/2, coin_values, width, label="CoInception", color=model_colors["CoInception"])
        ax.bar(x + width/2, ts2vec_values, width, label="TS2Vec", color=model_colors["TS2Vec"])
        
        # Set title and labels
        ax.set_title(f"{metric_name.capitalize()} Comparison")
        ax.set_xlabel("Dataset")
        ax.set_ylabel(metric_name.capitalize())
        
        # Set x-tick labels with rotation
        ax.set_xticks(x)
        ax.set_xticklabels(valid_datasets, rotation=45, ha='right')
        
        # Add legend
        ax.legend()
        
        # Add grid
        ax.grid(axis='y', alpha=0.3)
    
    # Hide unused subplots
    for j in range(len(metrics), 4):
        axs[j].axis('off')
    
    # Add data source annotation
    fig.text(0.5, 0.01, "Data Source: external_results.csv", 
             ha='center', fontsize=10, fontstyle='italic', color='gray')
    
    # Adjust layout
    plt.tight_layout()
    
    # Save the figure
    os.makedirs(PATH_CONFIG["vis_dir"], exist_ok=True)
    save_path = os.path.join(PATH_CONFIG["vis_dir"], "comparison_performance.png")
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    
    print(f"Generated performance comparison chart")
    print(f"Figure saved to: {save_path}")


def generate_overall_comparison(eval_results):
    """Generate overall comparison chart between CoInception and TS2Vec.
    
    Args:
        eval_results (dict): Dictionary of evaluation results for both models
    """
    
    # Set font sizes
    plt.rcParams.update({
        'axes.labelsize': 12,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'axes.titlesize': 14,
        'font.size': 11,
    })
    
    # Calculate overall performance scores
    model_scores = {
        "CoInception": {
            "accuracy": 0,
            "rank": 0,
            "count": {
                "accuracy": 0,
                "rank": 0
            }
        },
        "TS2Vec": {
            "accuracy": 0,
            "rank": 0,
            "count": {
                "accuracy": 0,
                "rank": 0
            }
        }
    }
    
    # Calculate scores for each model and metric
    for model_name in model_scores.keys():
        for dataset_name, dataset_results in eval_results[model_name].items():
            if isinstance(dataset_results, dict):
                # Process accuracy
                if "accuracy" in dataset_results:
                    model_scores[model_name]["accuracy"] += dataset_results["accuracy"]
                    model_scores[model_name]["count"]["accuracy"] += 1
                elif "acc" in dataset_results:
                    model_scores[model_name]["accuracy"] += dataset_results["acc"]
                    model_scores[model_name]["count"]["accuracy"] += 1
                
                # Process rank
                if "rank" in dataset_results:
                    model_scores[model_name]["rank"] += dataset_results["rank"]
                    model_scores[model_name]["count"]["rank"] += 1
    
    # Calculate average scores
    for model_name in model_scores.keys():
        for metric in ["accuracy", "rank"]:
            if model_scores[model_name]["count"][metric] > 0:
                model_scores[model_name][metric] /= model_scores[model_name]["count"][metric]
    
    # Create radar chart for overall comparison
    metrics = ["accuracy", "rank"]
    model_names = ["CoInception", "TS2Vec"]
    
    # For radar chart, we need to normalize metrics so higher is better for all
    # We'll invert rank since lower values are better
    normalized_scores = {}
    for model_name in model_names:
        normalized_scores[model_name] = []
        for metric in metrics:
            score = model_scores[model_name][metric]
            if metric == "rank" and score > 0:
                # Normalize rank: lower rank is better, so invert it
                max_score = max(model_scores["CoInception"][metric], model_scores["TS2Vec"][metric])
                min_score = min(model_scores["CoInception"][metric], model_scores["TS2Vec"][metric])
                # Normalize to [0, 1] where 1 is best
                normalized_scores[model_name].append(1 - ((score - min_score) / (max_score - min_score + 1e-8)))
            else:
                normalized_scores[model_name].append(score)
    
    # Create radar chart
    fig, ax = plt.subplots(figsize=(10, 8), subplot_kw=dict(polar=True))
    
    # Number of variables
    N = len(metrics)
    
    # Calculate angles for each metric
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    
    # Close the polygon
    for model_name in model_names:
        normalized_scores[model_name] = normalized_scores[model_name] + [normalized_scores[model_name][0]]
    angles = angles + [angles[0]]
    
    # Plot data for each model
    for model_name in model_names:
        ax.plot(angles, normalized_scores[model_name], 'o-', linewidth=2, label=model_name)
        ax.fill(angles, normalized_scores[model_name], alpha=0.25)
    
    # Set metric labels
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(metrics)
    
    # Set title
    ax.set_title("Overall Performance Comparison", fontsize=16, fontweight='bold', pad=20)
    
    # Add legend
    ax.legend(loc='upper right', bbox_to_anchor=(1.15, 1.1), fontsize=12)
    
    # Add data source annotation
    fig.text(0.5, 0.01, "Data Source: external_results.csv", 
             ha='center', fontsize=10, fontstyle='italic', color='gray')
    
    # Adjust layout
    plt.tight_layout()
    
    # Save the figure
    os.makedirs(PATH_CONFIG["vis_dir"], exist_ok=True)
    save_path = os.path.join(PATH_CONFIG["vis_dir"], "comparison_overall.png")
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    
    print(f"Generated overall comparison chart")
    print(f"Figure saved to: {save_path}")


def generate_all_comparison_charts():
    """Generate all comparison charts between CoInception and TS2Vec."""
    print("=" * 60)
    print("Generating comprehensive comparison charts")
    print("=" * 60)
    
    # Load evaluation results
    eval_results = load_model_results()
    
    # Generate performance comparison chart
    print("\n[1/2] Generating performance comparison chart")
    print("-" * 40)
    generate_performance_comparison(eval_results)
    
    # Generate overall comparison chart
    print("\n[2/2] Generating overall comparison chart")
    print("-" * 40)
    generate_overall_comparison(eval_results)
    
    print("\n" + "=" * 60)
    print("All comparison charts generated successfully!")
    print("=" * 60)


if __name__ == "__main__":
    os.makedirs(PATH_CONFIG["vis_dir"], exist_ok=True)
    generate_all_comparison_charts()
