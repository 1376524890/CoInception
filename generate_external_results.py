#!/usr/bin/env python3
"""
Generate external results CSV file using existing training data.
"""

import os
import sys
import numpy as np
import pandas as pd
import pickle
import glob

# Add the project root to the path
sys.path.append('.')

def generate_external_results():
    """Generate external results CSV file using only local training data."""
    print("Generating external results CSV using only local training data...")
    
    # Get project root directory
    project_root = os.path.dirname(os.path.abspath(__file__))
    
    # Get list of all training directories for CoInception using absolute paths
    coin_training_dirs = glob.glob(os.path.join(project_root, 'training', '*'))
    
    # Filter out non-classification directories (forecast datasets) and the reps directory
    classification_dirs = [d for d in coin_training_dirs if 'forecast' not in d and not d.endswith('reps')]
    
    print(f"Found {len(coin_training_dirs)} CoInception training directories")
    print(f"Found {len(classification_dirs)} classification training directories")
    
    # Calculate CoInception average accuracy from existing results
    coin_accuracies = []
    
    for training_dir in classification_dirs:
        try:
            # Extract dataset name
            dataset_name = training_dir.split('__')[0].split('/')[-1]
            
            # Check if eval_res.pkl exists
            eval_res_path = os.path.join(training_dir, 'eval_res.pkl')
            if os.path.exists(eval_res_path):
                with open(eval_res_path, 'rb') as f:
                    eval_res = pickle.load(f)
                
                if 'acc' in eval_res:
                    coin_accuracies.append(eval_res['acc'])
                    print(f"Loaded accuracy for {dataset_name}: {eval_res['acc']:.4f}")
        except Exception as e:
            print(f"Error loading results for {training_dir}: {e}")
            continue
    
    # Calculate average accuracy for CoInception
    if coin_accuracies:
        coin_avg_acc = np.mean(coin_accuracies)
        print(f"\nCoInception average accuracy: {coin_avg_acc:.4f} (n={len(coin_accuracies)} datasets)")
    else:
        print("\nNo CoInception accuracies found")
    
    # Now process TS2Vec results
    ts2vec_accuracies = []
    ts2vec_training_dirs = glob.glob(os.path.join(project_root, 'ts2vec', 'training', '*'))
    ts2vec_classification_dirs = [d for d in ts2vec_training_dirs if 'forecast' not in d]
    
    print(f"\nProcessing {len(ts2vec_classification_dirs)} TS2Vec classification directories")
    
    for training_dir in ts2vec_classification_dirs:
        try:
            # Extract dataset name
            dataset_name = training_dir.split('__')[0].split('/')[-1]
            
            # Check if eval_res.pkl exists
            eval_res_path = os.path.join(training_dir, 'eval_res.pkl')
            if os.path.exists(eval_res_path):
                with open(eval_res_path, 'rb') as f:
                    eval_res = pickle.load(f)
                
                if 'acc' in eval_res:
                    ts2vec_accuracies.append(eval_res['acc'])
                    print(f"Loaded TS2Vec accuracy for {dataset_name}: {eval_res['acc']:.4f}")
        except Exception as e:
            print(f"Error loading TS2Vec results for {training_dir}: {e}")
            continue
    
    # Calculate average accuracy for TS2Vec
    if ts2vec_accuracies:
        ts2vec_avg_acc = np.mean(ts2vec_accuracies)
        print(f"\nTS2Vec average accuracy: {ts2vec_avg_acc:.4f} (n={len(ts2vec_accuracies)} datasets)")
    else:
        print("\nNo TS2Vec accuracies found")
    
    # Create external results CSV with only local models
    external_results = []
    
    # Add CoInception results if available
    if coin_accuracies:
        external_results.append({
            'Method': 'CoInception',
            'UCR_Accuracy': coin_avg_acc,
            'UCR_Rank': None,  # No rank from local data
            'UCR_Params': '206K',  # Known model parameter
            'UEA_Accuracy': None,  # No UEA results from local data
            'UEA_Rank': None,  # No UEA rank from local data
            'UEA_Params': '206K'  # Same params for UEA
        })
    
    # Add TS2Vec results if available
    if ts2vec_accuracies:
        external_results.append({
            'Method': 'TS2Vec',
            'UCR_Accuracy': ts2vec_avg_acc,
            'UCR_Rank': None,  # No rank from local data
            'UCR_Params': '641K',  # Known model parameter
            'UEA_Accuracy': None,  # No UEA results from local data
            'UEA_Rank': None,  # No UEA rank from local data
            'UEA_Params': '641K'  # Same params for UEA
        })
    
    # Create DataFrame and save to CSV
    df = pd.DataFrame(external_results)
    csv_file = 'external_results.csv'
    df.to_csv(csv_file, index=False)
    
    print(f"\nGenerated external results CSV: {csv_file}")
    print("\nCSV file content:")
    print(df)
    
    return csv_file

if __name__ == "__main__":
    generate_external_results()
