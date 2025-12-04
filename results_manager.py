#!/usr/bin/env python3
"""
Results Manager for CoInception

This module provides a ResultsManager class to manage, query, and export evaluation results
from the CoInception model.
"""

import numpy as np
import pandas as pd
import os
import pickle
import json
import csv
from typing import Dict, List, Tuple, Optional, Union
from datetime import datetime
from analysis_preset import PATH_CONFIG


class ResultsManager:
    """Results manager for CoInception evaluation results."""
    
    def __init__(self, results_dir: str = None):
        """Initialize the results manager.
        
        Args:
            results_dir (str): Directory to store results. If None, uses default from analysis_preset.
        """
        self.results_dir = results_dir or PATH_CONFIG["results_dir"]
        self.results = []
        self.results_df = None
        
        # Create results directory if it doesn't exist
        os.makedirs(self.results_dir, exist_ok=True)
    
    def load_results(self, filename: str = None) -> None:
        """Load results from a file.
        
        Args:
            filename (str): Filename to load results from. If None, loads all results from directory.
        """
        if filename:
            # Load specific results file
            filepath = os.path.join(self.results_dir, filename)
            if os.path.exists(filepath):
                with open(filepath, "rb") as f:
                    result = pickle.load(f)
                self.results.append(result)
            else:
                raise FileNotFoundError(f"Results file not found: {filepath}")
        else:
            # Load all results files from directory
            self.results = []
            for file in os.listdir(self.results_dir):
                if file.endswith(".pkl"):
                    filepath = os.path.join(self.results_dir, file)
                    with open(filepath, "rb") as f:
                        result = pickle.load(f)
                    self.results.append(result)
        
        # Create dataframe for easy querying
        self._create_results_dataframe()
    
    def save_results(self, result: Dict, filename: str = None) -> None:
        """Save a result to a file.
        
        Args:
            result (Dict): Result to save.
            filename (str): Filename to save as. If None, generates a filename.
        """
        # Add timestamp if not present
        if "timestamp" not in result:
            result["timestamp"] = datetime.now().isoformat()
        
        # Generate filename if not provided
        if not filename:
            dataset_name = result.get("dataset", "unknown")
            task_type = result.get("task_type", "unknown")
            timestamp = result["timestamp"].replace(":", "-")
            filename = f"{dataset_name}_{task_type}_{timestamp}.pkl"
        
        # Save result
        filepath = os.path.join(self.results_dir, filename)
        with open(filepath, "wb") as f:
            pickle.dump(result, f)
        
        # Add to results list
        self.results.append(result)
        
        # Update dataframe
        self._create_results_dataframe()
        
        print(f"Saved results to: {filepath}")
    
    def _create_results_dataframe(self) -> None:
        """Create a dataframe from the results list for easy querying."""
        if not self.results:
            self.results_df = None
            return
        
        # Flatten results into a list of dictionaries suitable for dataframe
        flat_results = []
        for result in self.results:
            # Create a copy of the result
            flat_result = result.copy()
            
            # Flatten metrics if present
            if "metrics" in flat_result:
                metrics = flat_result.pop("metrics")
                for metric_name, metric_value in metrics.items():
                    flat_result[f"metric_{metric_name}"] = metric_value
            
            # Flatten params if present
            if "params" in flat_result:
                params = flat_result.pop("params")
                for param_name, param_value in params.items():
                    flat_result[f"param_{param_name}"] = param_value
            
            flat_results.append(flat_result)
        
        # Create dataframe
        self.results_df = pd.DataFrame(flat_results)
    
    def query_results(self, **kwargs) -> pd.DataFrame:
        """Query results based on conditions.
        
        Args:
            **kwargs: Conditions to filter results. Keys are column names, values are filter values.
            
        Returns:
            pd.DataFrame: Filtered results dataframe.
        """
        if self.results_df is None:
            return pd.DataFrame()
        
        # Apply filters
        filtered_df = self.results_df.copy()
        
        for key, value in kwargs.items():
            if key in filtered_df.columns:
                filtered_df = filtered_df[filtered_df[key] == value]
        
        return filtered_df
    
    def get_results_by_dataset(self, dataset_name: str) -> pd.DataFrame:
        """Get results for a specific dataset.
        
        Args:
            dataset_name (str): Name of the dataset.
            
        Returns:
            pd.DataFrame: Results for the dataset.
        """
        return self.query_results(dataset=dataset_name)
    
    def get_results_by_task(self, task_type: str) -> pd.DataFrame:
        """Get results for a specific task type.
        
        Args:
            task_type (str): Type of task (classification, forecasting, anomaly_detection).
            
        Returns:
            pd.DataFrame: Results for the task type.
        """
        return self.query_results(task_type=task_type)
    
    def get_results_by_metric(self, metric_name: str, threshold: Optional[float] = None,
                            operator: str = ">=") -> pd.DataFrame:
        """Get results based on a metric value.
        
        Args:
            metric_name (str): Name of the metric.
            threshold (float): Threshold value for the metric.
            operator (str): Comparison operator (<, <=, ==, >=, >).
            
        Returns:
            pd.DataFrame: Filtered results dataframe.
        """
        if self.results_df is None:
            return pd.DataFrame()
        
        metric_col = f"metric_{metric_name}"
        if metric_col not in self.results_df.columns:
            return pd.DataFrame()
        
        if threshold is None:
            return self.results_df[[col for col in self.results_df.columns if col != metric_col] + [metric_col]]
        
        # Apply threshold filter
        if operator == ">=":
            filtered_df = self.results_df[self.results_df[metric_col] >= threshold]
        elif operator == ">":
            filtered_df = self.results_df[self.results_df[metric_col] > threshold]
        elif operator == "<=":
            filtered_df = self.results_df[self.results_df[metric_col] <= threshold]
        elif operator == "<":
            filtered_df = self.results_df[self.results_df[metric_col] < threshold]
        elif operator == "==":
            filtered_df = self.results_df[self.results_df[metric_col] == threshold]
        else:
            raise ValueError(f"Unknown operator: {operator}")
        
        return filtered_df
    
    def export_results(self, filename: str, format: str = "csv", **kwargs) -> None:
        """Export results to a file.
        
        Args:
            filename (str): Filename to export to.
            format (str): Export format (csv, json, latex).
            **kwargs: Additional parameters for filtering results.
        """
        # Get filtered results
        filtered_df = self.query_results(**kwargs)
        
        if filtered_df.empty:
            print("No results to export.")
            return
        
        # Export based on format
        if format == "csv":
            filtered_df.to_csv(filename, index=False)
        elif format == "json":
            filtered_df.to_json(filename, orient="records", indent=2)
        elif format == "latex":
            latex_table = filtered_df.to_latex(index=False, float_format="%.4f")
            with open(filename, "w") as f:
                f.write(latex_table)
        else:
            raise ValueError(f"Unknown export format: {format}")
        
        print(f"Exported {len(filtered_df)} results to: {filename}")
    
    def get_best_results(self, metric_name: str, dataset_name: Optional[str] = None,
                       task_type: Optional[str] = None) -> pd.DataFrame:
        """Get the best results based on a metric.
        
        Args:
            metric_name (str): Name of the metric to optimize.
            dataset_name (str): Optional dataset filter.
            task_type (str): Optional task type filter.
            
        Returns:
            pd.DataFrame: Best results dataframe.
        """
        # Get filtered results
        filters = {}
        if dataset_name:
            filters["dataset"] = dataset_name
        if task_type:
            filters["task_type"] = task_type
        
        filtered_df = self.query_results(**filters)
        
        if filtered_df.empty:
            return pd.DataFrame()
        
        # Get metric column
        metric_col = f"metric_{metric_name}"
        if metric_col not in filtered_df.columns:
            return pd.DataFrame()
        
        # Get best results (highest value)
        best_idx = filtered_df[metric_col].idxmax()
        best_result = filtered_df.loc[[best_idx]]
        
        return best_result
    
    def get_summary_statistics(self, metric_name: str, group_by: Optional[str] = None) -> pd.DataFrame:
        """Get summary statistics for a metric.
        
        Args:
            metric_name (str): Name of the metric.
            group_by (str): Column to group by.
            
        Returns:
            pd.DataFrame: Summary statistics dataframe.
        """
        if self.results_df is None:
            return pd.DataFrame()
        
        metric_col = f"metric_{metric_name}"
        if metric_col not in self.results_df.columns:
            return pd.DataFrame()
        
        if group_by:
            if group_by not in self.results_df.columns:
                return pd.DataFrame()
            
            # Group by and calculate statistics
            summary = self.results_df.groupby(group_by)[metric_col].agg(["mean", "std", "min", "max", "count"])
        else:
            # Calculate overall statistics
            summary = self.results_df[metric_col].agg(["mean", "std", "min", "max", "count"])
            summary = pd.DataFrame(summary).T
        
        return summary
    
    def compare_models(self, metric_name: str, model_names: List[str] = None) -> pd.DataFrame:
        """Compare performance of different models.
        
        Args:
            metric_name (str): Name of the metric to compare.
            model_names (List[str]): List of model names to compare.
            
        Returns:
            pd.DataFrame: Model comparison dataframe.
        """
        if self.results_df is None:
            return pd.DataFrame()
        
        metric_col = f"metric_{metric_name}"
        if metric_col not in self.results_df.columns:
            return pd.DataFrame()
        
        if "model" not in self.results_df.columns:
            return pd.DataFrame()
        
        # Filter by model names if provided
        if model_names:
            comparison_df = self.results_df[self.results_df["model"].isin(model_names)]
        else:
            comparison_df = self.results_df.copy()
        
        # Group by model and calculate statistics
        comparison = comparison_df.groupby("model")[metric_col].agg(["mean", "std", "min", "max", "count"])
        
        return comparison
    
    def list_datasets(self) -> List[str]:
        """List all datasets in the results.
        
        Returns:
            List[str]: List of dataset names.
        """
        if self.results_df is None or "dataset" not in self.results_df.columns:
            return []
        
        return sorted(self.results_df["dataset"].unique().tolist())
    
    def list_task_types(self) -> List[str]:
        """List all task types in the results.
        
        Returns:
            List[str]: List of task types.
        """
        if self.results_df is None or "task_type" not in self.results_df.columns:
            return []
        
        return sorted(self.results_df["task_type"].unique().tolist())
    
    def list_metrics(self) -> List[str]:
        """List all metrics in the results.
        
        Returns:
            List[str]: List of metric names.
        """
        if self.results_df is None:
            return []
        
        # Extract metric columns
        metric_columns = [col.replace("metric_", "") for col in self.results_df.columns if col.startswith("metric_")]
        
        return sorted(metric_columns)
    
    def list_params(self) -> List[str]:
        """List all parameters in the results.
        
        Returns:
            List[str]: List of parameter names.
        """
        if self.results_df is None:
            return []
        
        # Extract parameter columns
        param_columns = [col.replace("param_", "") for col in self.results_df.columns if col.startswith("param_")]
        
        return sorted(param_columns)
    
    def clear_results(self) -> None:
        """Clear all results from memory."""
        self.results = []
        self.results_df = None
        print("Cleared all results from memory.")
    
    def delete_results(self, **kwargs) -> int:
        """Delete results based on conditions.
        
        Args:
            **kwargs: Conditions to filter results for deletion.
            
        Returns:
            int: Number of results deleted.
        """
        # Get filtered results
        filtered_df = self.query_results(**kwargs)
        
        if filtered_df.empty:
            print("No results to delete.")
            return 0
        
        # Get indices of results to delete
        delete_indices = filtered_df.index.tolist()
        
        # Delete results from list
        self.results = [result for i, result in enumerate(self.results) if i not in delete_indices]
        
        # Update dataframe
        self._create_results_dataframe()
        
        print(f"Deleted {len(delete_indices)} results.")
        return len(delete_indices)
    
    def get_results_count(self, **kwargs) -> int:
        """Get the number of results matching conditions.
        
        Args:
            **kwargs: Conditions to filter results.
            
        Returns:
            int: Number of matching results.
        """
        filtered_df = self.query_results(**kwargs)
        return len(filtered_df)


# Example usage
if __name__ == "__main__":
    # Initialize results manager
    rm = ResultsManager()
    
    # Print available methods
    print("Results Manager Methods:")
    print("=" * 50)
    methods = [
        ("load_results", "Load results from files"),
        ("save_results", "Save results to file"),
        ("query_results", "Query results with filters"),
        ("get_results_by_dataset", "Get results for a specific dataset"),
        ("get_results_by_task", "Get results for a specific task type"),
        ("get_results_by_metric", "Get results based on a metric value"),
        ("export_results", "Export results to file"),
        ("get_best_results", "Get best results for a metric"),
        ("get_summary_statistics", "Get summary statistics for a metric"),
        ("get_best_results", "Get best results for a metric"),
        ("compare_models", "Compare model performance"),
        ("list_datasets", "List all datasets"),
        ("list_task_types", "List all task types"),
        ("list_metrics", "List all metrics"),
        ("list_params", "List all parameters"),
        ("clear_results", "Clear all results from memory"),
        ("delete_results", "Delete results based on conditions"),
        ("get_results_count", "Get count of matching results")
    ]
    
    for method_name, description in methods:
        print(f"  {method_name}: {description}")
    
    # Example: Create sample results
    sample_results = [
        {
            "dataset": "FordA",
            "task_type": "classification",
            "model": "CoInception",
            "metrics": {
                "accuracy": 0.98,
                "f1": 0.97,
                "precision": 0.98,
                "recall": 0.97
            },
            "params": {
                "batch_size": 8,
                "lr": 0.001,
                "repr_dims": 320
            }
        },
        {
            "dataset": "FordA",
            "task_type": "classification",
            "model": "TS2Vec",
            "metrics": {
                "accuracy": 0.96,
                "f1": 0.95,
                "precision": 0.96,
                "recall": 0.94
            },
            "params": {
                "batch_size": 8,
                "lr": 0.001,
                "repr_dims": 320
            }
        }
    ]
    
    # Save sample results
    print("\nSaving sample results...")
    for result in sample_results:
        rm.save_results(result)
    
    # Query results
    print("\nQuerying results:")
    ford_results = rm.get_results_by_dataset("FordA")
    print(f"FordA results shape: {ford_results.shape}")
    
    # Get best results
    print("\nBest results for accuracy:")
    best_results = rm.get_best_results("accuracy")
    print(best_results[["dataset", "model", "metric_accuracy"]])
    
    # Compare models
    print("\nModel comparison:")
    comparison = rm.compare_models("accuracy")
    print(comparison)
    
    print("\nResults manager initialized successfully!")
