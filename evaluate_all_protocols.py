#!/usr/bin/env python3
"""
Evaluate all protocols on existing training results.

This script will:
1. Iterate through all existing training results
2. For each dataset, load the saved model
3. Run all evaluation protocols on the loaded model
4. Generate a CSV file with the results for Table II
"""

import os
import sys
import numpy as np
import pandas as pd
import pickle
import glob

# Add the project root to the path
sys.path.append('.')

import datautils
from tasks.classification import eval_classification

def evaluate_all_protocols():
    """Evaluate all protocols on existing training results."""
    # List of evaluation protocols to run
    eval_protocols = ['linear', 'svm', 'knn', 'dtw', 'tnc', 'tst', 'tstcc', 'tloss', 'timesnet']
    
    # Get list of all training directories with saved models
    training_dirs = glob.glob('training/*__run_*')
    print(f"Found {len(training_dirs)} training directories")
    
    # Filter out non-classification directories (forecast datasets)
    classification_dirs = [d for d in training_dirs if 'forecast' not in d]
    print(f"Found {len(classification_dirs)} classification training directories")
    
    # Create results directory
    results_dir = 'evaluation_results'
    os.makedirs(results_dir, exist_ok=True)
    
    # Initialize results dictionary
    all_results = {}
    
    # Iterate through all classification training directories
    for training_dir in classification_dirs:
        # Extract dataset name from directory name
        dataset_name = training_dir.split('__')[0].split('/')[-1]
        print(f"\nProcessing dataset: {dataset_name}")
        
        try:
            # Check if model.pkl exists
            model_path = os.path.join(training_dir, 'model.pkl')
            if not os.path.exists(model_path):
                print(f"Model file not found for {dataset_name},