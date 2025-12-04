#!/usr/bin/env python3
"""
Full Dataset Analysis Script for CoInception

This script traverses all datasets defined in analysis_preset.py and runs the CoInception model
with the preset parameters to reproduce the paper results.
"""

import os
import sys
import subprocess
import logging
from analysis_preset import (
    PRESET_PARAMS,
    DATASET_LISTS,
    PATH_CONFIG,
    create_directories,
    get_preset_params
)

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('full_dataset_analysis.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

def run_coinception(dataset_name, dataset_type, variant=None):
    """
    Run CoInception model on a single dataset.
    
    Args:
        dataset_name (str): Name of the dataset
        dataset_type (str): Type of the dataset (classification, forecasting, anomaly_detection)
        variant (str, optional): Variant of the dataset (univariate, multivariate)
    """
    # Create necessary directories
    create_directories()
    
    # Prepare command arguments
    args = []
    
    # Build the command with required positional arguments first
    # Set run_name to dataset_name for simplicity
    run_name = f"coinception_{dataset_name}"
    
    # Determine loader based on dataset type
    if dataset_type == 'classification':
        if variant == 'univariate':
            loader = 'UCR'
        else:  # multivariate
            loader = 'UEA'
    elif dataset_type == 'forecasting':
        # Use appropriate loader for forecasting datasets
        loader = 'forecast_csv'
    elif dataset_type == 'anomaly_detection':
        # Use appropriate loader for anomaly detection datasets
        loader = 'anomaly'
    
    # Base command with required arguments
    cmd = [
        sys.executable, 'train.py',
        dataset_name,  # First positional argument: dataset
        run_name,  # Second positional argument: run_name
        '--loader', loader  # Required --loader argument
    ]
    
    # Add common arguments
    cmd.extend([
        '--batch-size', str(PRESET_PARAMS['batch_size']),
        '--repr-dims', str(PRESET_PARAMS['repr_dims']),
        '--max-threads', str(PRESET_PARAMS['max_threads']),
        '--seed', str(PRESET_PARAMS['seed']),
        '--lr', str(PRESET_PARAMS['lr']),
        '--eval',
        '--save_ckpt',
        '--max-train-length', str(PRESET_PARAMS['max_train_length'])
    ])
    
    # Add dataset-specific arguments if needed
    if dataset_type == 'classification' and variant == 'multivariate':
        cmd.append('--is_multivariate')
    elif dataset_type == 'anomaly_detection' and dataset_name in ['Yahoo', 'KPI']:
        # anomaly detection specific settings are handled by the loader
        pass
    
    logger.info(f"Running command: {' '.join(cmd)}")
    
    # Run the command
    try:
        result = subprocess.run(
            cmd,
            check=True,
            capture_output=True,
            text=True,
            cwd=os.getcwd()
        )
        logger.info(f"Successfully ran CoInception on {dataset_name}")
        logger.debug(f"Command output: {result.stdout}")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"Failed to run CoInception on {dataset_name}")
        logger.error(f"Return code: {e.returncode}")
        logger.error(f"Error output: {e.stderr}")
        logger.debug(f"Command output: {e.stdout}")
        return False
    except Exception as e:
        logger.error(f"Unexpected error running CoInception on {dataset_name}: {str(e)}")
        return False

def main():
    """Main function to run analysis on all datasets."""
    import argparse
    
    # Add argument parser for test mode
    parser = argparse.ArgumentParser(description="Full Dataset Analysis for CoInception")
    parser.add_argument('--test', type=str, help='Test with a single dataset')
    parser.add_argument('--category', type=str, help='Category of the test dataset')
    
    args = parser.parse_args()
    
    logger.info("Starting full dataset analysis for CoInception")
    logger.info(f"Using preset parameters: {PRESET_PARAMS}")
    
    # Create necessary directories
    create_directories()
    
    # Statistics
    total_datasets = 0
    successful_runs = 0
    failed_runs = 0
    failed_datasets = []
    
    # Check if we're in test mode
    if args.test and args.category:
        # Test mode: process only the specified dataset
        logger.info(f"\n=== TEST MODE: Processing single dataset {args.test} from category {args.category} ===")
        
        if args.category not in DATASET_LISTS:
            logger.error(f"Unknown category: {args.category}")
            return 1
        
        info = DATASET_LISTS[args.category]
        dataset_type = info['type']
        variant = info.get('variant', None)
        
        if args.test not in info['datasets']:
            logger.error(f"Dataset {args.test} not found in category {args.category}")
            return 1
        
        total_datasets = 1
        logger.info(f"Processing dataset 1: {args.test}")
        
        if run_coinception(args.test, dataset_type, variant):
            successful_runs = 1
        else:
            failed_runs = 1
            failed_datasets.append((args.category, args.test))
    else:
        # Normal mode: traverse all dataset categories
        for category, info in DATASET_LISTS.items():
            dataset_type = info['type']
            variant = info.get('variant', None)
            datasets = info['datasets']
            
            logger.info(f"\n=== Processing {info['name']} ({len(datasets)} datasets) ===")
            
            for dataset_name in datasets:
                total_datasets += 1
                logger.info(f"Processing dataset {total_datasets}: {dataset_name}")
                
                # Run CoInception on the dataset
                if run_coinception(dataset_name, dataset_type, variant):
                    successful_runs += 1
                else:
                    failed_runs += 1
                    failed_datasets.append((category, dataset_name))
    
    # Print summary
    logger.info("\n" + "="*60)
    logger.info("FULL DATASET ANALYSIS SUMMARY")
    logger.info("="*60)
    logger.info(f"Total datasets processed: {total_datasets}")
    logger.info(f"Successful runs: {successful_runs}")
    logger.info(f"Failed runs: {failed_runs}")
    
    if failed_datasets:
        logger.info("\nFailed datasets:")
        for category, dataset_name in failed_datasets:
            logger.info(f"  - {category}/{dataset_name}")
    
    logger.info("\nFull dataset analysis completed!")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
