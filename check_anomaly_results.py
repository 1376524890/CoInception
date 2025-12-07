#!/usr/bin/env python3
"""
Check anomaly detection results from TS2Vec KPI training.
"""

import pickle
import os

# Path to the KPI anomaly detection results
kpi_results_path = "/home/codeserver/CoInception/ts2vec/training/kpi__anomaly_kpi_20251206_071904/eval_res.pkl"

# Check if the file exists
if os.path.exists(kpi_results_path):
    print(f"Loading KPI anomaly detection results from: {kpi_results_path}")
    
    # Load the results
    with open(kpi_results_path, 'rb') as f:
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
else:
    print(f"File not found: {kpi_results_path}")
