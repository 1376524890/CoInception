#!/usr/bin/env python3
"""
Run All Datasets Script for CoInception

This script runs the CoInception model on all datasets specified in the analysis_preset.py file,
using the preset parameters. It supports running multiple datasets sequentially or in parallel.
"""

import os
import sys
import time
import argparse
import subprocess
import multiprocessing
from datetime import datetime
from analysis_preset import (
    get_all_datasets, 
    get_preset_params, 
    create_directories,
    PATH_CONFIG,
    ANOMALY_SETTINGS
)


def run_single_dataset(dataset_info, run_args):
    """Run the CoInception model on a single dataset.
    
    Args:
        dataset_info (dict): Dictionary containing dataset information.
        run_args (argparse.Namespace): Command-line arguments.
    
    Returns:
        dict: Results of the run.
    """
    dataset_name = dataset_info["name"]
    dataset_type = dataset_info["type"]
    variant = dataset_info.get("variant", "univariate")
    
    # Determine the loader based on dataset type
    if dataset_type == "classification":
        if variant == "univariate":
            loader = "UCR"
        else:
            loader = "UEA"
    elif dataset_type == "forecasting":
        loader = "forecast_csv_univar" if variant == "univariate" else "forecast_csv"
    elif dataset_type == "anomaly_detection":
        loader = "anomaly"
    else:
        raise ValueError(f"Unknown dataset type: {dataset_type}")
    
    # Get preset parameters
    preset_params = get_preset_params()
    
    # Create run name with timestamp
    run_name = f"run_{dataset_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    # Build the command to run train.py
    cmd = [
        sys.executable, "train.py",
        dataset_name,
        run_name,
        f"--loader={loader}",
        f"--gpu={run_args.gpu}",
        f"--batch-size={preset_params['batch_size']}",
        f"--lr={preset_params['lr']}",
        f"--repr-dims={preset_params['repr_dims']}",
        f"--max-train-length={preset_params['max_train_length']}",
        f"--max-threads={preset_params['max_threads']}",
        f"--seed={preset_params['seed']}"
    ]
    
    # Add eval and save_ckpt flags if specified
    if preset_params.get("eval", True):
        cmd.append("--eval")
    if preset_params.get("save_ckpt", True):
        cmd.append("--save_ckpt")
    
    print(f"\n{'='*60}")
    print(f"Running dataset: {dataset_name}")
    print(f"Dataset type: {dataset_type}")
    print(f"Loader: {loader}")
    print(f"Command: {' '.join(cmd)}")
    print(f"{'='*60}")
    
    # Create output directory
    os.makedirs(PATH_CONFIG["results_dir"], exist_ok=True)
    
    # Run the command and capture output
    output_file = os.path.join(PATH_CONFIG["results_dir"], f"{dataset_name}_output.txt")
    
    try:
        start_time = time.time()
        with open(output_file, "w") as f:
            result = subprocess.run(cmd, cwd=os.getcwd(), stdout=f, stderr=subprocess.STDOUT, check=True, text=True)
        end_time = time.time()
        
        run_time = end_time - start_time
        
        print(f"✅ Dataset {dataset_name} completed successfully in {run_time:.2f} seconds")
        print(f"Output saved to: {output_file}")
        
        return {
            "dataset": dataset_name,
            "status": "success",
            "run_time": run_time,
            "output_file": output_file,
            "command": " ".join(cmd)
        }
    except subprocess.CalledProcessError as e:
        print(f"❌ Dataset {dataset_name} failed with return code {e.returncode}")
        print(f"Output saved to: {output_file}")
        
        return {
            "dataset": dataset_name,
            "status": "failed",
            "return_code": e.returncode,
            "output_file": output_file,
            "command": " ".join(cmd)
        }
    except Exception as e:
        print(f"❌ Dataset {dataset_name} failed with exception: {e}")
        
        return {
            "dataset": dataset_name,
            "status": "error",
            "error": str(e),
            "command": " ".join(cmd)
        }


def run_anomaly_dataset(dataset_name, setting, run_args):
    """Run the CoInception model on an anomaly detection dataset with specific settings.
    
    Args:
        dataset_name (str): Name of the anomaly detection dataset.
        setting (str): Setting for anomaly detection (normal or coldstart).
        run_args (argparse.Namespace): Command-line arguments.
    
    Returns:
        dict: Results of the run.
    """
    # Get preset parameters
    preset_params = get_preset_params()
    
    # Determine the loader based on setting
    loader = "anomaly" if setting == "normal" else "anomaly_coldstart"
    
    # Create run name with timestamp
    run_name = f"run_{dataset_name}_{setting}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    # Build the command to run train.py
    cmd = [
        sys.executable, "train.py",
        dataset_name,
        run_name,
        f"--loader={loader}",
        f"--gpu={run_args.gpu}",
        f"--batch-size={preset_params['batch_size']}",
        f"--lr={preset_params['lr']}",
        f"--repr-dims={preset_params['repr_dims']}",
        f"--max-train-length={preset_params['max_train_length']}",
        f"--max-threads={preset_params['max_threads']}",
        f"--seed={preset_params['seed']}"
    ]
    
    # Add eval and save_ckpt flags if specified
    if preset_params.get("eval", True):
        cmd.append("--eval")
    if preset_params.get("save_ckpt", True):
        cmd.append("--save_ckpt")
    
    print(f"\n{'='*60}")
    print(f"Running anomaly dataset: {dataset_name}")
    print(f"Setting: {setting}")
    print(f"Loader: {loader}")
    print(f"Command: {' '.join(cmd)}")
    print(f"{'='*60}")
    
    # Create output directory
    os.makedirs(PATH_CONFIG["results_dir"], exist_ok=True)
    
    # Run the command and capture output
    output_file = os.path.join(PATH_CONFIG["results_dir"], f"{dataset_name}_{setting}_output.txt")
    
    try:
        start_time = time.time()
        with open(output_file, "w") as f:
            result = subprocess.run(cmd, cwd=os.getcwd(), stdout=f, stderr=subprocess.STDOUT, check=True, text=True)
        end_time = time.time()
        
        run_time = end_time - start_time
        
        print(f"✅ Anomaly dataset {dataset_name} ({setting}) completed successfully in {run_time:.2f} seconds")
        print(f"Output saved to: {output_file}")
        
        return {
            "dataset": dataset_name,
            "setting": setting,
            "status": "success",
            "run_time": run_time,
            "output_file": output_file,
            "command": " ".join(cmd)
        }
    except subprocess.CalledProcessError as e:
        print(f"❌ Anomaly dataset {dataset_name} ({setting}) failed with return code {e.returncode}")
        print(f"Output saved to: {output_file}")
        
        return {
            "dataset": dataset_name,
            "setting": setting,
            "status": "failed",
            "return_code": e.returncode,
            "output_file": output_file,
            "command": " ".join(cmd)
        }
    except Exception as e:
        print(f"❌ Anomaly dataset {dataset_name} ({setting}) failed with exception: {e}")
        
        return {
            "dataset": dataset_name,
            "setting": setting,
            "status": "error",
            "error": str(e),
            "command": " ".join(cmd)
        }


def process_dataset_type(dataset_type_info, run_args):
    """Process all datasets of a specific type.
    
    Args:
        dataset_type_info (dict): Dictionary containing dataset type information.
        run_args (argparse.Namespace): Command-line arguments.
    
    Returns:
        list: List of results for all datasets of this type.
    """
    results = []
    datasets = dataset_type_info["datasets"]
    dataset_type = dataset_type_info["type"]
    
    for dataset_name in datasets:
        # Create dataset info dict
        dataset_info = {
            "name": dataset_name,
            "type": dataset_type,
            "variant": dataset_type_info.get("variant", "univariate")
        }
        
        if dataset_type == "anomaly_detection":
            # Process anomaly detection datasets with different settings
            if dataset_name in ANOMALY_SETTINGS:
                for setting in ANOMALY_SETTINGS[dataset_name]["settings"]:
                    result = run_anomaly_dataset(dataset_name, setting, run_args)
                    results.append(result)
            else:
                # Default settings for anomaly detection
                for setting in ["normal"]:
                    result = run_anomaly_dataset(dataset_name, setting, run_args)
                    results.append(result)
        else:
            # Process other dataset types
            result = run_single_dataset(dataset_info, run_args)
            results.append(result)
        
        # Add a delay between runs if specified
        if run_args.delay > 0:
            time.sleep(run_args.delay)
    
    return results


def main():
    """Main function to run all datasets."""
    parser = argparse.ArgumentParser(description="Run CoInception on all datasets")
    parser.add_argument('--gpu', type=int, default=0, help='The GPU ID to use (default: 0)')
    parser.add_argument('--parallel', type=int, default=1, help='Number of parallel processes to use (default: 1, sequential)')
    parser.add_argument('--delay', type=int, default=0, help='Delay between runs in seconds (default: 0)')
    parser.add_argument('--dataset-type', type=str, default=None, 
                        help='Run only specific dataset type (ucr, uea, forecasting, anomaly_detection)')
    parser.add_argument('--dry-run', action='store_true', help='Print commands without executing them')
    
    args = parser.parse_args()
    
    # Set CUDA visible devices if GPU is specified
    if args.gpu >= 0:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    
    print("CoInception All Datasets Runner")
    print("=" * 60)
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"GPU: {args.gpu}")
    print(f"Parallel processes: {args.parallel}")
    print(f"Delay between runs: {args.delay}s")
    
    # Create necessary directories
    create_directories()
    
    # Get all datasets
    all_datasets = get_all_datasets()
    
    # Filter datasets if dataset-type is specified
    if args.dataset_type:
        if args.dataset_type not in all_datasets:
            print(f"Error: Unknown dataset type '{args.dataset_type}'")
            print(f"Available dataset types: {list(all_datasets.keys())}")
            sys.exit(1)
        datasets_to_run = {args.dataset_type: all_datasets[args.dataset_type]}
        print(f"Running only dataset type: {args.dataset_type}")
    else:
        datasets_to_run = all_datasets
        print(f"Running all dataset types: {list(datasets_to_run.keys())}")
    
    print(f"Total dataset categories: {len(datasets_to_run)}")
    
    # Calculate total number of datasets to run
    total_datasets = 0
    for dataset_type, info in datasets_to_run.items():
        count = info["count"]
        if dataset_type == "anomaly_detection":
            # Anomaly detection has multiple settings per dataset
            for dataset_name in info["datasets"]:
                if dataset_name in ANOMALY_SETTINGS:
                    total_datasets += len(ANOMALY_SETTINGS[dataset_name]["settings"])
                else:
                    total_datasets += 1
        else:
            total_datasets += count
    
    print(f"Total datasets to run: {total_datasets}")
    print("=" * 60)
    
    # Run datasets
    start_time = time.time()
    all_results = []
    
    for dataset_type, dataset_type_info in datasets_to_run.items():
        print(f"\n{'='*60}")
        print(f"Processing dataset type: {dataset_type}")
        print(f"Description: {dataset_type_info['description']}")
        print(f"Number of datasets: {dataset_type_info['count']}")
        
        # Process datasets of this type
        type_results = process_dataset_type(dataset_type_info, args)
        all_results.extend(type_results)
    
    end_time = time.time()
    total_time = end_time - start_time
    
    # Print summary
    print(f"\n{'='*60}")
    print("Run Summary")
    print("=" * 60)
    print(f"Total runtime: {time.strftime('%H:%M:%S', time.gmtime(total_time))}")
    print(f"Total datasets: {len(all_results)}")
    
    # Count results
    success_count = sum(1 for r in all_results if r["status"] == "success")
    failed_count = sum(1 for r in all_results if r["status"] == "failed")
    error_count = sum(1 for r in all_results if r["status"] == "error")
    
    print(f"Success: {success_count}")
    print(f"Failed: {failed_count}")
    print(f"Error: {error_count}")
    
    # Save results to file
    results_file = os.path.join(PATH_CONFIG["results_dir"], f"all_datasets_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt")
    with open(results_file, "w") as f:
        f.write("CoInception All Datasets Results\n")
        f.write("=" * 60 + "\n")
        f.write(f"Run time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Total runtime: {time.strftime('%H:%M:%S', time.gmtime(total_time))}\n")
        f.write(f"Total datasets: {len(all_results)}\n")
        f.write(f"Success: {success_count}, Failed: {failed_count}, Error: {error_count}\n\n")
        
        for i, result in enumerate(all_results, 1):
            f.write(f"Result {i}: {result['status']}\n")
            if "setting" in result:
                f.write(f"  Dataset: {result['dataset']} ({result['setting']})\n")
            else:
                f.write(f"  Dataset: {result['dataset']}\n")
            f.write(f"  Command: {result['command']}\n")
            if "output_file" in result:
                f.write(f"  Output: {result['output_file']}\n")
            if result["status"] == "failed":
                f.write(f"  Return code: {result['return_code']}\n")
            elif result["status"] == "error":
                f.write(f"  Error: {result['error']}\n")
            if "run_time" in result:
                f.write(f"  Runtime: {result['run_time']:.2f}s\n")
            f.write("\n")
    
    print(f"Results saved to: {results_file}")
    print("=" * 60)
    print("All datasets processed.")


if __name__ == "__main__":
    main()
