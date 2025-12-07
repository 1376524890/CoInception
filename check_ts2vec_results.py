#!/usr/bin/env python3
"""
Script to check TS2Vec evaluation results
"""

import pickle
import os

# Define the path to the TS2Vec KPI evaluation result file
ts2vec_kpi_file = '/home/codeserver/CoInception/ts2vec/training/kpi__anomaly_kpi_20251206_071904/eval_res.pkl'

print("Checking TS2Vec KPI evaluation results...")
print("=" * 50)

# Check if the file exists
if os.path.exists(ts2vec_kpi_file):
    print(f"\nReading TS2Vec KPI results from: {ts2vec_kpi_file}")
    with open(ts2vec_kpi_file, 'rb') as f:
        ts2vec_results = pickle.load(f)
    
    print("TS2Vec KPI results:")
    if isinstance(ts2vec_results, dict):
        for key, value in ts2vec_results.items():
            print(f"  {key}: {value}")
            # If the value is a dictionary, print its contents too
            if isinstance(value, dict):
                for subkey, subvalue in value.items():
                    print(f"    {subkey}: {subvalue}")
    else:
        print(f"  Results type: {type(ts2vec_results)}")
        print(f"  Results content: {ts2vec_results}")
        
    # Check if coldstart results are included
    has_coldstart = any('cold' in key.lower() for key in ts2vec_results.keys())
    print(f"\nHas coldstart results: {'✅ Yes' if has_coldstart else '❌ No'}")
else:
    print(f"TS2Vec KPI file not found: {ts2vec_kpi_file}")
