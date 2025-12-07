#!/usr/bin/env python3
"""
Visualization Script for Figures 7-8: Noise Ratio Analysis (Corrected)

This script generates Figures 7 and 8 matching the paper layout:
- Figure 7: Table showing noise ratio comparison (Noise Ratio, CoInception, TS2Vec for MSE/MAE)
- Figure 8: Radar charts comparing MSE and MAE across different noise ratios
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from analysis_preset import get_vis_settings, PATH_CONFIG
import os
import pickle
import json


def load_evaluation_results(dataset='ETTm1'):
    """Load evaluation results from noise ratio analysis experiment.
    Returns:
        dict: Dictionary of evaluation results or empty dict if no results found
    """
    results_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Default empty results
    eval_results = {
        "CoInception": {},
        "TS2Vec": {}
    }
    
    import pickle
    
    # 尝试加载噪声比率分析实验结果
    noise_results_file = os.path.join(results_dir, 'results', f'all_noise_results_{dataset}.pkl')
    
    if os.path.exists(noise_results_file):
        print(f"Found noise ratio analysis results file: {noise_results_file}")
        try:
            with open(noise_results_file, 'rb') as f:
                all_results = pickle.load(f)
            
            print(f"Loaded noise ratio analysis results. Available models: {list(all_results.keys())}")
            
            # 将结果转换为预期格式
            if 'coinception' in all_results:
                coin_results = all_results['coinception']
                # 提取MSE和MAE结果，只使用可用的噪声比率
                coin_mse = []
                coin_mae = []
                for ratio in [0, 10, 20, 30, 40, 50]:
                    if ratio in coin_results:
                        coin_mse.append(coin_results[ratio]['MSE'])
                        coin_mae.append(coin_results[ratio]['MAE'])
                    else:
                        # 使用默认值填充缺失的噪声比率
                        coin_mse.append(0.0)
                        coin_mae.append(0.0)
                
                eval_results['CoInception']['noise_robustness'] = {
                    'mse': coin_mse,
                    'mae': coin_mae
                }
                print("Loaded CoInception noise robustness results")
            
            if 'vs2rec' in all_results:
                vs2rec_results = all_results['vs2rec']
                # 提取MSE和MAE结果，只使用可用的噪声比率
                vs2rec_mse = []
                vs2rec_mae = []
                for ratio in [0, 10, 20, 30, 40, 50]:
                    if ratio in vs2rec_results:
                        vs2rec_mse.append(vs2rec_results[ratio]['MSE'])
                        vs2rec_mae.append(vs2rec_results[ratio]['MAE'])
                    else:
                        # 使用默认值填充缺失的噪声比率
                        vs2rec_mse.append(0.0)
                        vs2rec_mae.append(0.0)
                
                eval_results['TS2Vec']['noise_robustness'] = {
                    'mse': vs2rec_mse,
                    'mae': vs2rec_mae
                }
                print("Loaded vs2rec/TS2Vec noise robustness results")
            
        except Exception as e:
            print(f"Error loading noise ratio analysis results: {e}")
    else:
        print(f"Noise ratio analysis results file not found: {noise_results_file}")
        print("Please run noise_ratio_analysis.py first to generate the results")
    
    # Print final results structure for debugging
    print(f"Final evaluation results structure: {eval_results}")
    
    return eval_results

def generate_figure7(dataset='ETTm1'):
    """Generate Figure 7: Noise Ratio Table matching the paper's format."""
    
    # Set font sizes
    plt.rcParams.update({
        'axes.labelsize': 12,
        'xtick.labelsize': 11,
        'ytick.labelsize': 11,
        'axes.titlesize': 14,
        'font.size': 11,
    })
    
    # Load evaluation results
    eval_results = load_evaluation_results(dataset)
    
    # Noise ratios for the table
    noise_ratios = ['0%', '10%', '20%', '30%', '40%', '50%']
    
    # Initialize empty lists for noise ratio results
    coin_mse = ['', '', '', '', '', '']
    coin_mae = ['', '', '', '', '', '']
    ts2vec_mse = ['', '', '', '', '', '']
    ts2vec_mae = ['', '', '', '', '', '']
    
    # Try to get actual results from evaluation data
    if eval_results:
        # Look for CoInception noise robustness results
        if 'CoInception' in eval_results and isinstance(eval_results['CoInception'], dict):
            coin_result = eval_results['CoInception']
            # Check for noise robustness data in various locations
            noise_data_coin = None
            if 'noise_robustness' in coin_result:
                noise_data_coin = coin_result['noise_robustness']
            elif 'noise' in coin_result:
                noise_data_coin = coin_result['noise']
            elif 'noise_ratio' in coin_result:
                noise_data_coin = coin_result['noise_ratio']
            
            if noise_data_coin:
                print(f"Found CoInception noise data: {noise_data_coin}")
                # Update CoInception values if available
                if isinstance(noise_data_coin, dict):
                    # Try different formats for noise data
                    if 'mse' in noise_data_coin and len(noise_data_coin['mse']) >= 6:
                        coin_mse = noise_data_coin['mse'][:6]
                        print("Updated CoInception MSE values from evaluation results")
                    elif isinstance(noise_data_coin.get('mse'), (list, tuple)) and len(noise_data_coin['mse']) >= 6:
                        coin_mse = noise_data_coin['mse'][:6]
                        print("Updated CoInception MSE values from list format")
                    
                    if 'mae' in noise_data_coin and len(noise_data_coin['mae']) >= 6:
                        coin_mae = noise_data_coin['mae'][:6]
                        print("Updated CoInception MAE values from evaluation results")
                    elif isinstance(noise_data_coin.get('mae'), (list, tuple)) and len(noise_data_coin['mae']) >= 6:
                        coin_mae = noise_data_coin['mae'][:6]
                        print("Updated CoInception MAE values from list format")
        
        # Look for TS2Vec noise robustness results
        if 'TS2Vec' in eval_results and isinstance(eval_results['TS2Vec'], dict):
            ts2vec_result = eval_results['TS2Vec']
            # Check for noise robustness data in various locations
            noise_data_ts2vec = None
            if 'noise_robustness' in ts2vec_result:
                noise_data_ts2vec = ts2vec_result['noise_robustness']
            elif 'noise' in ts2vec_result:
                noise_data_ts2vec = ts2vec_result['noise']
            elif 'noise_ratio' in ts2vec_result:
                noise_data_ts2vec = ts2vec_result['noise_ratio']
            
            if noise_data_ts2vec:
                print(f"Found TS2Vec noise data: {noise_data_ts2vec}")
                # Update TS2Vec values if available
                if isinstance(noise_data_ts2vec, dict):
                    # Try different formats for noise data
                    if 'mse' in noise_data_ts2vec and len(noise_data_ts2vec['mse']) >= 6:
                        ts2vec_mse = noise_data_ts2vec['mse'][:6]
                        print("Updated TS2Vec MSE values from evaluation results")
                    elif isinstance(noise_data_ts2vec.get('mse'), (list, tuple)) and len(noise_data_ts2vec['mse']) >= 6:
                        ts2vec_mse = noise_data_ts2vec['mse'][:6]
                        print("Updated TS2Vec MSE values from list format")
                    
                    if 'mae' in noise_data_ts2vec and len(noise_data_ts2vec['mae']) >= 6:
                        ts2vec_mae = noise_data_ts2vec['mae'][:6]
                        print("Updated TS2Vec MAE values from evaluation results")
                    elif isinstance(noise_data_ts2vec.get('mae'), (list, tuple)) and len(noise_data_ts2vec['mae']) >= 6:
                        ts2vec_mae = noise_data_ts2vec['mae'][:6]
                        print("Updated TS2Vec MAE values from list format")
    
    # If no actual noise robustness results found, keep lists empty
    print(f"Using real noise robustness results: {any(coin_mse) or any(ts2vec_mse) or any(coin_mae) or any(ts2vec_mae)}")
    
    # Create figure for table
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.axis('off')
    
    # Table data
    col_labels = ['Noise Ratio', 'CoInception\nMSE', 'CoInception\nMAE', 
                  'TS2Vec\nMSE', 'TS2Vec\nMAE']
    
    table_data = []
    for i in range(len(noise_ratios)):
        # Format values if they are numbers, otherwise use empty string
        coin_mse_val = f'{coin_mse[i]:.3f}' if i < len(coin_mse) and isinstance(coin_mse[i], (int, float)) else ''
        coin_mae_val = f'{coin_mae[i]:.3f}' if i < len(coin_mae) and isinstance(coin_mae[i], (int, float)) else ''
        ts2vec_mse_val = f'{ts2vec_mse[i]:.3f}' if i < len(ts2vec_mse) and isinstance(ts2vec_mse[i], (int, float)) else ''
        ts2vec_mae_val = f'{ts2vec_mae[i]:.3f}' if i < len(ts2vec_mae) and isinstance(ts2vec_mae[i], (int, float)) else ''
        
        row = [
            noise_ratios[i],
            coin_mse_val,
            coin_mae_val,
            ts2vec_mse_val,
            ts2vec_mae_val
        ]
        table_data.append(row)
    
    # Create table
    table = ax.table(
        cellText=table_data,
        colLabels=col_labels,
        cellLoc='center',
        loc='center',
        colWidths=[0.15, 0.2, 0.2, 0.2, 0.2]
    )
    
    # Style the table
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1.2, 1.8)
    
    # Color the header row
    for i in range(len(col_labels)):
        table[(0, i)].set_facecolor('#E6E6FA')
        table[(0, i)].set_text_props(weight='bold')
    
    # Highlight best results (CoInception columns)
    for i in range(1, len(noise_ratios) + 1):
        table[(i, 1)].set_facecolor('#E8F5E9')  # Light green for MSE
        table[(i, 2)].set_facecolor('#E8F5E9')  # Light green for MAE
    
    ax.set_title('Noise Ratio Analysis - ETTm1 Dataset', fontsize=14, fontweight='bold', pad=20)
    
    # Add data source annotation
    fig.text(0.5, 0.01, "Data Source: Based on noise robustness evaluation results", 
             ha='center', fontsize=10, fontstyle='italic', color='gray')
    
    # Adjust layout
    plt.tight_layout()
    
    # Save the figure
    os.makedirs(PATH_CONFIG["vis_dir"], exist_ok=True)
    save_path = os.path.join(PATH_CONFIG["vis_dir"], "figure7.png")
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    
    print(f"Generated Figure 7: Noise Ratio Table")
    print(f"Figure saved to: {save_path}")


def generate_figure8(dataset='ETTm1'):
    """Generate Figure 8: Radar Charts matching the paper's format."""
    
    # Set font sizes
    plt.rcParams.update({
        'axes.labelsize': 12,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'axes.titlesize': 14,
        'font.size': 10,
    })
    
    # Load evaluation results
    eval_results = load_evaluation_results(dataset)
    
    # Categories (noise ratios)
    categories = ['0%', '10%', '20%', '30%', '40%', '50%']
    N = len(categories)
    
    # Initialize empty lists for noise ratio results
    coin_mse = ['', '', '', '', '', '']
    coin_mae = ['', '', '', '', '', '']
    ts2vec_mse = ['', '', '', '', '', '']
    ts2vec_mae = ['', '', '', '', '', '']
    
    # Try to get actual results from evaluation data
    if eval_results:
        # Look for CoInception noise robustness results
        if 'CoInception' in eval_results and isinstance(eval_results['CoInception'], dict):
            coin_result = eval_results['CoInception']
            # Check for noise robustness data in various locations
            noise_data_coin = None
            if 'noise_robustness' in coin_result:
                noise_data_coin = coin_result['noise_robustness']
            elif 'noise' in coin_result:
                noise_data_coin = coin_result['noise']
            elif 'noise_ratio' in coin_result:
                noise_data_coin = coin_result['noise_ratio']
            
            if noise_data_coin:
                print(f"Found CoInception noise data for radar chart: {noise_data_coin}")
                # Update CoInception values if available
                if isinstance(noise_data_coin, dict):
                    # Try different formats for noise data
                    if 'mse' in noise_data_coin and len(noise_data_coin['mse']) >= 6:
                        coin_mse = noise_data_coin['mse'][:6]
                        print("Updated CoInception MSE values from evaluation results")
                    elif isinstance(noise_data_coin.get('mse'), (list, tuple)) and len(noise_data_coin['mse']) >= 6:
                        coin_mse = noise_data_coin['mse'][:6]
                        print("Updated CoInception MSE values from list format")
                    
                    if 'mae' in noise_data_coin and len(noise_data_coin['mae']) >= 6:
                        coin_mae = noise_data_coin['mae'][:6]
                        print("Updated CoInception MAE values from evaluation results")
                    elif isinstance(noise_data_coin.get('mae'), (list, tuple)) and len(noise_data_coin['mae']) >= 6:
                        coin_mae = noise_data_coin['mae'][:6]
                        print("Updated CoInception MAE values from list format")
        
        # Look for TS2Vec noise robustness results
        if 'TS2Vec' in eval_results and isinstance(eval_results['TS2Vec'], dict):
            ts2vec_result = eval_results['TS2Vec']
            # Check for noise robustness data in various locations
            noise_data_ts2vec = None
            if 'noise_robustness' in ts2vec_result:
                noise_data_ts2vec = ts2vec_result['noise_robustness']
            elif 'noise' in ts2vec_result:
                noise_data_ts2vec = ts2vec_result['noise']
            elif 'noise_ratio' in ts2vec_result:
                noise_data_ts2vec = ts2vec_result['noise_ratio']
            
            if noise_data_ts2vec:
                print(f"Found TS2Vec noise data for radar chart: {noise_data_ts2vec}")
                # Update TS2Vec values if available
                if isinstance(noise_data_ts2vec, dict):
                    # Try different formats for noise data
                    if 'mse' in noise_data_ts2vec and len(noise_data_ts2vec['mse']) >= 6:
                        ts2vec_mse = noise_data_ts2vec['mse'][:6]
                        print("Updated TS2Vec MSE values from evaluation results")
                    elif isinstance(noise_data_ts2vec.get('mse'), (list, tuple)) and len(noise_data_ts2vec['mse']) >= 6:
                        ts2vec_mse = noise_data_ts2vec['mse'][:6]
                        print("Updated TS2Vec MSE values from list format")
                    
                    if 'mae' in noise_data_ts2vec and len(noise_data_ts2vec['mae']) >= 6:
                        ts2vec_mae = noise_data_ts2vec['mae'][:6]
                        print("Updated TS2Vec MAE values from evaluation results")
                    elif isinstance(noise_data_ts2vec.get('mae'), (list, tuple)) and len(noise_data_ts2vec['mae']) >= 6:
                        ts2vec_mae = noise_data_ts2vec['mae'][:6]
                        print("Updated TS2Vec MAE values from list format")
    
    # If no actual noise robustness results found, skip radar chart generation
    print(f"Using real noise robustness results for radar charts: {any(coin_mse) or any(ts2vec_mse) or any(coin_mae) or any(ts2vec_mae)}")
    
    # Skip radar chart generation if no real results
    if not (any(coin_mse) and any(ts2vec_mse) and any(coin_mae) and any(ts2vec_mae)):
        print("Skipping radar chart generation: insufficient real noise robustness results")
        return
    
    # Create figure with 1x2 subplots for MSE and MAE radar charts
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6), subplot_kw=dict(polar=True))
    
    # Function to create radar chart
    def radar_chart(ax, categories, values1, values2, title, label1='CoInception', label2='TS2Vec'):
        # Calculate angles
        angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
        
        # Close the polygon
        values1 = values1 + [values1[0]]
        values2 = values2 + [values2[0]]
        angles = angles + [angles[0]]
        
        # Plot data
        ax.plot(angles, values1, 'o-', linewidth=2, label=label1, color='#E74C3C')
        ax.fill(angles, values1, alpha=0.25, color='#E74C3C')
        
        ax.plot(angles, values2, 's-', linewidth=2, label=label2, color='#27AE60')
        ax.fill(angles, values2, alpha=0.25, color='#27AE60')
        
        # Set category labels
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(categories, fontsize=11)
        
        # Set title
        ax.set_title(title, fontsize=14, fontweight='bold', pad=15)
        
        # Add legend
        ax.legend(loc='upper right', bbox_to_anchor=(1.15, 1.1), fontsize=10)
    
    # MSE Radar Chart
    radar_chart(ax1, categories, coin_mse, ts2vec_mse, 'MSE')
    
    # MAE Radar Chart
    radar_chart(ax2, categories, coin_mae, ts2vec_mae, 'MAE')
    
    # Add overall title
    fig.suptitle('CoInception vs TS2Vec Performance with Different Noise Ratios (ETTm1)', 
                 fontsize=14, fontweight='bold', y=1.02)
    
    # Add data source annotation
    fig.text(0.5, 0.01, "Data Source: Based on noise robustness evaluation results", 
             ha='center', fontsize=10, fontstyle='italic', color='gray')
    
    # Adjust layout
    plt.tight_layout()
    
    # Save the figure
    os.makedirs(PATH_CONFIG["vis_dir"], exist_ok=True)
    save_path = os.path.join(PATH_CONFIG["vis_dir"], "figure8.png")
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    
    print(f"Generated Figure 8: Radar Charts")
    print(f"Figure saved to: {save_path}")


def generate_figure7_8(dataset='ETTm1'):
    """Generate both Figures 7 and 8."""
    generate_figure7(dataset)
    generate_figure8(dataset)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Generate Figures 7 and 8 for noise ratio analysis')
    parser.add_argument('--dataset', type=str, default='ETTm1', help='Dataset name')
    args = parser.parse_args()
    
    os.makedirs(PATH_CONFIG["vis_dir"], exist_ok=True)
    generate_figure7_8(args.dataset)