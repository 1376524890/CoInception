#!/usr/bin/env python3
"""
Check the content of generated tables to identify missing data.
"""

import pandas as pd
import os
import pickle

# Paths to the tables
visualizations_dir = "/home/codeserver/CoInception/visualizations"
tables_dir = "/home/codeserver/CoInception/tables"

def check_table_content(table_file):
    """Check the content of a table file."""
    print(f"\n=== Checking {table_file} ===")
    
    if os.path.exists(table_file):
        # Check if it's an image file
        if table_file.endswith('.png') or table_file.endswith('.jpg') or table_file.endswith('.jpeg'):
            print(f"Image file, cannot read text content")
            # Get file size instead
            file_size = os.path.getsize(table_file)
            print(f"File size: {file_size} bytes")
        else:
            # Read the table content
            with open(table_file, 'r') as f:
                content = f.read()
            
            # Print the content (first 500 chars)
            print(f"Table content (first 500 chars):\n{content[:500]}...")
            
            # Count empty cells
            empty_cell_count = content.count('& &')
            print(f"Number of empty cells: {empty_cell_count}")
    else:
        print(f"File not found: {table_file}")

def check_ts2vec_models():
    """Check if TS2Vec models exist."""
    print(f"\n=== Checking TS2Vec models ===")
    
    # Check TS2Vec model directory
    ts2vec_model_dir = "/home/codeserver/CoInception/ts2vec/training"
    
    if os.path.exists(ts2vec_model_dir):
        # Find all TS2Vec model files
        import glob
        model_files = glob.glob(os.path.join(ts2vec_model_dir, '**', 'model*.pkl'), recursive=True)
        
        print(f"Found {len(model_files)} TS2Vec model files")
        for model_file in model_files:
            print(f"  - {model_file}")
    else:
        print(f"TS2Vec model directory not found: {ts2vec_model_dir}")

def check_eval_results():
    """Check evaluation results to identify missing data."""
    print(f"\n=== Checking evaluation results ===")
    
    # Load evaluation results
    eval_results_path = "/home/codeserver/CoInception/generate_tables.py"
    
    # Check if the file exists
    if os.path.exists(eval_results_path):
        # Import the evaluation results
        import sys
        sys.path.insert(0, "/home/codeserver/CoInception")
        
        from generate_tables import eval_results
        
        print(f"Loaded evaluation results for {len(eval_results)} datasets")
        print(f"Dataset keys: {list(eval_results.keys())}")
        
        # Check forecast datasets specifically
        forecast_datasets = [key for key in eval_results.keys() if '_forecast' in key]
        print(f"\nForecast datasets: {forecast_datasets}")
        
        # Check anomaly detection datasets
        anomaly_datasets = [key for key in eval_results.keys() if key.lower() in ['kpi', 'yahoo']]
        print(f"Anomaly detection datasets: {anomaly_datasets}")
        
        # Check content of a forecast dataset
        if forecast_datasets:
            sample_forecast = forecast_datasets[0]
            print(f"\nSample forecast dataset {sample_forecast} content:")
            print(f"  Keys: {list(eval_results[sample_forecast].keys())}")
            
            # Check if TS2Vec results exist for this dataset
            if 'TS2Vec' in eval_results[sample_forecast]:
                print(f"  TS2Vec results: {type(eval_results[sample_forecast]['TS2Vec'])}")
            if 'CoInception' in eval_results[sample_forecast]:
                print(f"  CoInception results: {type(eval_results[sample_forecast]['CoInception'])}")
    else:
        print(f"File not found: {eval_results_path}")

# Main execution
if __name__ == "__main__":
    # Check table_i (table1)
    check_table_content(os.path.join(tables_dir, 'table_i.tex'))
    
    # Check table_iii (table3)
    check_table_content(os.path.join(tables_dir, 'table_iii.tex'))
    
    # Check figure7
    check_table_content(os.path.join(visualizations_dir, 'figure7.png'))
    
    # Check TS2Vec models
    check_ts2vec_models()
    
    # Check evaluation results
    check_eval_results()
