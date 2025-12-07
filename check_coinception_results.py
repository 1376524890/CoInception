#!/usr/bin/env python3
"""
Script to check the contents of CoInception evaluation result files
"""

import pickle
import os

# Define the paths to the evaluation result files
normal_file = '/home/codeserver/CoInception/training/kpi__anomaly_0/eval_res.pkl'
coldstart_file = '/home/codeserver/CoInception/training/kpi__anomaly_coldstart_0/eval_res.pkl'

print("Checking CoInception evaluation results...")
print("=" * 50)

# Check normal setting results
if os.path.exists(normal_file):
    print(f"\nReading Normal Setting results from: {normal_file}")
    with open(normal_file, 'rb') as f:
        normal_results = pickle.load(f)
    print("Normal Setting results:")
    for key, value in normal_results.items():
        print(f"  {key}: {value}")
else:
    print(f"Normal Setting file not found: {normal_file}")

# Check coldstart setting results
if os.path.exists(coldstart_file):
    print(f"\nReading Cold-start Setting results from: {coldstart_file}")
    with open(coldstart_file, 'rb') as f:
        coldstart_results = pickle.load(f)
    print("Cold-start Setting results:")
    for key, value in coldstart_results.items():
        print(f"  {key}: {value}")
else:
    print(f"Cold-start Setting file not found: {coldstart_file}")

# Compare the two results
if os.path.exists(normal_file) and os.path.exists(coldstart_file):
    print(f"\nComparing Normal vs Cold-start results:")
    print("=" * 50)
    
    # Check if dictionaries are identical
    if normal_results == coldstart_results:
        print("❌ WARNING: Normal and Cold-start results are IDENTICAL!")
        print("This indicates a problem with the coldstart training.")
    else:
        print("✅ Normal and Cold-start results are DIFFERENT!")
        
        # Show differences
        all_keys = set(normal_results.keys()).union(coldstart_results.keys())
        for key in all_keys:
            if key in normal_results and key in coldstart_results:
                if normal_results[key] != coldstart_results[key]:
                    print(f"  {key}: Normal={normal_results[key]}, Coldstart={coldstart_results[key]}")
            elif key in normal_results:
                print(f"  {key}: Normal={normal_results[key]}, Coldstart=NOT FOUND")
            else:
                print(f"  {key}: Normal=NOT FOUND, Coldstart={coldstart_results[key]}")
