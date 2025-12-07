#!/usr/bin/env python3
"""
Check TS2Vec forecast results structure to identify the correct keys.
"""

import pickle
import os
import glob

# Path to TS2Vec forecast results
forecast_results_path = "/home/codeserver/CoInception/ts2vec/training/ETT"

# Find all forecast eval_res.pkl files
forecast_files = glob.glob(os.path.join(forecast_results_path, '**', 'eval_res.pkl'), recursive=True)

print(f"Found {len(forecast_files)} TS2Vec forecast result files")

for forecast_file in forecast_files:
    print(f"\n=== Checking {forecast_file} ===")
    
    # Load the results
    with open(forecast_file, 'rb') as f:
        results = pickle.load(f)
    
    # Print the structure of the results
    print(f"Results type: {type(results)}")
    print(f"Results keys: {list(results.keys()) if isinstance(results, dict) else 'Not a dictionary'}")
    
    # If it's a dictionary, print more details
    if isinstance(results, dict):
        for key, value in results.items():
            print(f"\nKey: {key}")
            print(f"Value type: {type(value)}")
            if isinstance(value, dict):
                print(f"Value keys: {list(value.keys())}")
            elif isinstance(value, (list, tuple, set)):
                print(f"Value length: {len(value)}")
            elif isinstance(value, (int, float)):
                print(f"Value: {value}")
            else:
                print(f"Value sample: {str(value)[:100]}..." if len(str(value)) > 100 else f"Value: {value}")
