#!/usr/bin/env python3
"""
Train TS2Vec model on all datasets.
"""

import os
import subprocess
import sys

# Define datasets to train on with their loader types
# Format: (dataset, loader_type, task_type)
datasets_to_train = [
    # ETT forecast datasets
    ("ETTh1", "forecast_csv", "forecast"),
    ("ETTh2", "forecast_csv", "forecast"),
    ("ETTm1", "forecast_csv", "forecast"),
    # Other forecast datasets
    ("electricity", "forecast_csv", "forecast"),
    # Anomaly detection datasets
    ("kpi", "anomaly", "anomaly")
]

# Define TS2Vec training parameters
train_params = {
    "repr_dims": 320,
    "max_threads": 8,
    "seed": 42,
    "eval": True
}

# Change to TS2Vec directory
os.chdir("ts2vec")

# Create training directory if it doesn't exist
os.makedirs("training", exist_ok=True)

# Train on each dataset
for dataset, loader, task_type in datasets_to_train:
    print(f"\nTraining TS2Vec on {dataset} with {loader} loader...")
    
    # Determine dataset path based on dataset and loader type
    if task_type == "forecast":
        if dataset in ["ETTh1", "ETTh2", "ETTm1"]:
            # For ETT datasets, we need to use the correct path
            dataset_path = f"ETT/{dataset}"
        else:
            dataset_path = dataset
    else:
        # For other tasks, use the dataset name directly
        dataset_path = dataset
    
    # Construct training command
    cmd = [
        "/home/codeserver/.conda/envs/ts2vec/bin/python", "train.py",
        dataset_path, f"{task_type}_{dataset.lower()}",
        "--loader", loader,
        "--repr-dims", str(train_params["repr_dims"]),
        "--max-threads", str(train_params["max_threads"]),
        "--seed", str(train_params["seed"])
    ]
    
    if train_params["eval"]:
        cmd.append("--eval")
    
    print(f"Running command: {' '.join(cmd)}")
    
    # Run training command
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    # Print output
    print("STDOUT:", result.stdout)
    if result.stderr:
        print("STDERR:", result.stderr)
    
    if result.returncode != 0:
        print(f"Error training TS2Vec on {dataset}")
    else:
        print(f"Successfully trained TS2Vec on {dataset}")

print("\nTS2Vec training completed!")
