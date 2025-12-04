#!/usr/bin/env python3
"""
Dataset Utility Functions for CoInception

This module provides utilities for dataset management, loading, preprocessing, and statistics.
"""

import os
import numpy as np
import pickle
from typing import Dict, List, Tuple, Optional, Union
from pathlib import Path


class DatasetManager:
    """Dataset manager for handling dataset loading, preprocessing, and caching."""
    
    def __init__(self, data_dir: str = "data"):
        """Initialize the dataset manager.
        
        Args:
            data_dir (str): Directory where datasets are stored.
        """
        self.data_dir = data_dir
        self.dataset_cache = {}
        self.dataset_info = {}
        self.cache_dir = os.path.join(os.getcwd(), "dataset_cache")
        os.makedirs(self.cache_dir, exist_ok=True)
    
    def get_dataset_path(self, dataset_name: str, dataset_type: str) -> str:
        """Get the path to a dataset.
        
        Args:
            dataset_name (str): Name of the dataset.
            dataset_type (str): Type of the dataset (UCR, UEA, forecast, anomaly).
            
        Returns:
            str: Path to the dataset.
        """
        if dataset_type in ["UCR", "UEA"]:
            return os.path.join(self.data_dir, dataset_type, dataset_name)
        elif dataset_type == "forecast":
            return os.path.join(self.data_dir, "forecast", f"{dataset_name}.csv")
        elif dataset_type == "anomaly":
            return os.path.join(self.data_dir, "anomaly", dataset_name)
        else:
            raise ValueError(f"Unknown dataset type: {dataset_type}")
    
    def load_dataset(self, dataset_name: str, dataset_type: str, loader_func: callable, *args, **kwargs) -> Tuple:
        """Load a dataset, using cache if available.
        
        Args:
            dataset_name (str): Name of the dataset.
            dataset_type (str): Type of the dataset.
            loader_func (callable): Function to load the dataset if not in cache.
            *args: Arguments to pass to the loader function.
            **kwargs: Keyword arguments to pass to the loader function.
            
        Returns:
            Tuple: Loaded dataset.
        """
        cache_key = f"{dataset_type}_{dataset_name}_{'_'.join([str(a) for a in args])}_{'_'.join([f'{k}={v}' for k, v in kwargs.items()])}"
        cache_file = os.path.join(self.cache_dir, f"{cache_key}.pkl")
        
        # Check if dataset is in memory cache
        if cache_key in self.dataset_cache:
            return self.dataset_cache[cache_key]
        
        # Check if dataset is in file cache
        if os.path.exists(cache_file):
            with open(cache_file, "rb") as f:
                data = pickle.load(f)
            self.dataset_cache[cache_key] = data
            return data
        
        # Load dataset using provided function
        data = loader_func(*args, **kwargs)
        
        # Cache dataset to file and memory
        with open(cache_file, "wb") as f:
            pickle.dump(data, f)
        self.dataset_cache[cache_key] = data
        
        return data
    
    def clear_cache(self, dataset_name: Optional[str] = None) -> None:
        """Clear the dataset cache.
        
        Args:
            dataset_name (Optional[str]): Name of the dataset to clear cache for. If None, clear all cache.
        """
        if dataset_name is None:
            # Clear all memory cache
            self.dataset_cache.clear()
            # Clear all file cache
            for file in os.listdir(self.cache_dir):
                os.remove(os.path.join(self.cache_dir, file))
        else:
            # Clear memory cache for specific dataset
            keys_to_remove = [k for k in self.dataset_cache if dataset_name in k]
            for key in keys_to_remove:
                del self.dataset_cache[key]
            # Clear file cache for specific dataset
            for file in os.listdir(self.cache_dir):
                if dataset_name in file:
                    os.remove(os.path.join(self.cache_dir, file))
    
    def get_dataset_statistics(self, data: np.ndarray) -> Dict:
        """Get statistics for a dataset.
        
        Args:
            data (np.ndarray): Dataset data.
            
        Returns:
            Dict: Dictionary containing dataset statistics.
        """
        stats = {
            "shape": data.shape,
            "n_instances": data.shape[0],
            "n_timestamps": data.shape[1],
            "n_features": data.shape[2] if len(data.shape) > 2 else 1,
            "mean": np.mean(data),
            "std": np.std(data),
            "min": np.min(data),
            "max": np.max(data),
            "nan_count": np.sum(np.isnan(data)),
            "nan_percentage": np.sum(np.isnan(data)) / data.size * 100,
            "positive_count": np.sum(data > 0),
            "negative_count": np.sum(data < 0),
            "zero_count": np.sum(data == 0)
        }
        
        return stats
    
    def normalize_dataset(self, data: np.ndarray, method: str = "zscore") -> Tuple[np.ndarray, Dict]:
        """Normalize a dataset.
        
        Args:
            data (np.ndarray): Dataset data.
            method (str): Normalization method (zscore, minmax, robust).
            
        Returns:
            Tuple[np.ndarray, Dict]: Normalized data and normalization parameters.
        """
        if method == "zscore":
            mean = np.nanmean(data, axis=(0, 1))
            std = np.nanstd(data, axis=(0, 1))
            normalized = (data - mean) / (std + 1e-8)
            params = {"method": "zscore", "mean": mean, "std": std}
        elif method == "minmax":
            min_val = np.nanmin(data, axis=(0, 1))
            max_val = np.nanmax(data, axis=(0, 1))
            normalized = (data - min_val) / (max_val - min_val + 1e-8)
            params = {"method": "minmax", "min": min_val, "max": max_val}
        elif method == "robust":
            median = np.nanmedian(data, axis=(0, 1))
            iqr = np.nanpercentile(data, 75, axis=(0, 1)) - np.nanpercentile(data, 25, axis=(0, 1))
            normalized = (data - median) / (iqr + 1e-8)
            params = {"method": "robust", "median": median, "iqr": iqr}
        else:
            raise ValueError(f"Unknown normalization method: {method}")
        
        return normalized, params
    
    def denormalize_dataset(self, data: np.ndarray, params: Dict) -> np.ndarray:
        """Denormalize a dataset.
        
        Args:
            data (np.ndarray): Normalized data.
            params (Dict): Normalization parameters.
            
        Returns:
            np.ndarray: Denormalized data.
        """
        method = params["method"]
        
        if method == "zscore":
            mean = params["mean"]
            std = params["std"]
            return data * (std + 1e-8) + mean
        elif method == "minmax":
            min_val = params["min"]
            max_val = params["max"]
            return data * (max_val - min_val + 1e-8) + min_val
        elif method == "robust":
            median = params["median"]
            iqr = params["iqr"]
            return data * (iqr + 1e-8) + median
        else:
            raise ValueError(f"Unknown normalization method: {method}")
    
    def split_dataset(self, data: np.ndarray, labels: Optional[np.ndarray] = None, 
                     train_ratio: float = 0.7, val_ratio: float = 0.15, 
                     test_ratio: float = 0.15, shuffle: bool = True, 
                     seed: Optional[int] = None) -> Tuple:
        """Split a dataset into train, validation, and test sets.
        
        Args:
            data (np.ndarray): Dataset data.
            labels (Optional[np.ndarray]): Dataset labels.
            train_ratio (float): Ratio of training data.
            val_ratio (float): Ratio of validation data.
            test_ratio (float): Ratio of test data.
            shuffle (bool): Whether to shuffle the data.
            seed (Optional[int]): Random seed for shuffling.
            
        Returns:
            Tuple: Split datasets (train_data, val_data, test_data, [train_labels, val_labels, test_labels]).
        """
        if not np.isclose(train_ratio + val_ratio + test_ratio, 1.0):
            raise ValueError("Train, validation, and test ratios must sum to 1")
        
        n_instances = data.shape[0]
        indices = np.arange(n_instances)
        
        if shuffle:
            if seed is not None:
                np.random.seed(seed)
            np.random.shuffle(indices)
        
        # Calculate split indices
        train_end = int(n_instances * train_ratio)
        val_end = train_end + int(n_instances * val_ratio)
        
        # Split data
        train_indices = indices[:train_end]
        val_indices = indices[train_end:val_end]
        test_indices = indices[val_end:]
        
        train_data = data[train_indices]
        val_data = data[val_indices]
        test_data = data[test_indices]
        
        if labels is not None:
            train_labels = labels[train_indices]
            val_labels = labels[val_indices]
            test_labels = labels[test_indices]
            return train_data, val_data, test_data, train_labels, val_labels, test_labels
        else:
            return train_data, val_data, test_data
    
    def add_noise(self, data: np.ndarray, noise_type: str = "gaussian", 
                 noise_level: float = 0.1, seed: Optional[int] = None) -> np.ndarray:
        """Add noise to a dataset.
        
        Args:
            data (np.ndarray): Dataset data.
            noise_type (str): Type of noise (gaussian, uniform, salt_pepper).
            noise_level (float): Level of noise to add.
            seed (Optional[int]): Random seed for noise generation.
            
        Returns:
            np.ndarray: Noisy data.
        """
        if seed is not None:
            np.random.seed(seed)
        
        noise = np.zeros_like(data)
        
        if noise_type == "gaussian":
            # Gaussian noise with standard deviation based on noise_level
            std = noise_level * np.std(data)
            noise = np.random.normal(0, std, data.shape)
        elif noise_type == "uniform":
            # Uniform noise in range [-noise_level*max_abs, noise_level*max_abs]
            max_abs = np.max(np.abs(data))
            noise = np.random.uniform(-noise_level * max_abs, noise_level * max_abs, data.shape)
        elif noise_type == "salt_pepper":
            # Salt and pepper noise
            salt_prob = noise_level / 2
            pepper_prob = noise_level / 2
            salt = np.random.rand(*data.shape) < salt_prob
            pepper = np.random.rand(*data.shape) < pepper_prob
            noise = np.zeros_like(data)
            noise[salt] = np.max(data)
            noise[pepper] = np.min(data)
        else:
            raise ValueError(f"Unknown noise type: {noise_type}")
        
        return data + noise


def get_dataset_info(dataset_name: str, dataset_type: str) -> Dict:
    """Get information about a dataset.
    
    Args:
        dataset_name (str): Name of the dataset.
        dataset_type (str): Type of the dataset.
        
    Returns:
        Dict: Dictionary containing dataset information.
    """
    from analysis_preset import DATASET_LISTS
    
    for category, info in DATASET_LISTS.items():
        if dataset_name in info["datasets"]:
            return {
                "name": dataset_name,
                "type": info["type"],
                "category": category,
                "variant": info.get("variant", "univariate"),
                "description": info["description"]
            }
    
    # Default information if not found
    return {
        "name": dataset_name,
        "type": dataset_type,
        "category": "unknown",
        "variant": "univariate",
        "description": "Unknown dataset"
    }


def get_loader_type(dataset_name: str, dataset_type: str) -> str:
    """Get the loader type for a dataset.
    
    Args:
        dataset_name (str): Name of the dataset.
        dataset_type (str): Type of the dataset.
        
    Returns:
        str: Loader type.
    """
    from analysis_preset import ANOMALY_SETTINGS
    
    if dataset_type == "classification":
        from analysis_preset import DATASET_LISTS
        for category, info in DATASET_LISTS.items():
            if dataset_name in info["datasets"]:
                if info.get("variant", "univariate") == "univariate":
                    return "UCR"
                else:
                    return "UEA"
        return "UCR"
    elif dataset_type == "forecasting":
        return "forecast_csv_univar"  # Default to univariate
    elif dataset_type == "anomaly_detection":
        return "anomaly"
    else:
        raise ValueError(f"Unknown dataset type: {dataset_type}")


def generate_dataset_id(dataset_name: str, dataset_type: str, 
                        setting: Optional[str] = None) -> str:
    """Generate a unique ID for a dataset run.
    
    Args:
        dataset_name (str): Name of the dataset.
        dataset_type (str): Type of the dataset.
        setting (Optional[str]): Additional setting information.
        
    Returns:
        str: Unique dataset ID.
    """
    if setting:
        return f"{dataset_type}_{dataset_name}_{setting}"
    else:
        return f"{dataset_type}_{dataset_name}"


def save_dataset_metadata(dataset_id: str, metadata: Dict, 
                          output_dir: str = "dataset_metadata") -> None:
    """Save dataset metadata to a file.
    
    Args:
        dataset_id (str): Unique dataset ID.
        metadata (Dict): Dataset metadata.
        output_dir (str): Directory to save metadata.
    """
    os.makedirs(output_dir, exist_ok=True)
    metadata_file = os.path.join(output_dir, f"{dataset_id}.pkl")
    
    with open(metadata_file, "wb") as f:
        pickle.dump(metadata, f)


def load_dataset_metadata(dataset_id: str, 
                          output_dir: str = "dataset_metadata") -> Dict:
    """Load dataset metadata from a file.
    
    Args:
        dataset_id (str): Unique dataset ID.
        output_dir (str): Directory containing metadata.
        
    Returns:
        Dict: Dataset metadata.
    """
    metadata_file = os.path.join(output_dir, f"{dataset_id}.pkl")
    
    if not os.path.exists(metadata_file):
        raise FileNotFoundError(f"Metadata file not found: {metadata_file}")
    
    with open(metadata_file, "rb") as f:
        return pickle.load(f)


def list_available_datasets() -> Dict:
    """List all available datasets from analysis_preset.
    
    Returns:
        Dict: Dictionary containing available datasets grouped by category.
    """
    from analysis_preset import DATASET_LISTS
    return DATASET_LISTS.copy()


def filter_datasets_by_type(dataset_type: str) -> List[str]:
    """Filter datasets by type.
    
    Args:
        dataset_type (str): Type of dataset to filter.
        
    Returns:
        List[str]: List of dataset names.
    """
    from analysis_preset import DATASET_LISTS
    
    datasets = []
    for category, info in DATASET_LISTS.items():
        if info["type"] == dataset_type:
            datasets.extend(info["datasets"])
    
    return datasets


def filter_datasets_by_variant(variant: str) -> List[str]:
    """Filter datasets by variant (univariate or multivariate).
    
    Args:
        variant (str): Variant to filter.
        
    Returns:
        List[str]: List of dataset names.
    """
    from analysis_preset import DATASET_LISTS
    
    datasets = []
    for category, info in DATASET_LISTS.items():
        if info.get("variant", "univariate") == variant:
            datasets.extend(info["datasets"])
    
    return datasets


def get_dataset_count() -> Dict[str, int]:
    """Get the count of datasets by category.
    
    Returns:
        Dict[str, int]: Dictionary containing dataset counts by category.
    """
    from analysis_preset import DATASET_LISTS
    
    counts = {}
    for category, info in DATASET_LISTS.items():
        counts[category] = info["count"]
    
    return counts
