#!/usr/bin/env python3
"""
Analysis Preset Configuration for CoInception

This file defines the preset parameters and dataset lists for reproducing the CoInception paper results.
"""

import os

# Preset parameters based on README guidelines
PRESET_PARAMS = {
    "batch_size": 8,
    "repr_dims": 320,
    "max_threads": 8,
    "seed": 42,
    "lr": 0.001,
    "eval": True,
    "save_ckpt": True,
    "input_len": 512,  # Default input length
    "hidden_dims": 64,  # Default hidden dimension
    "depth": 3,  # Default depth
    "temporal_unit": 0,  # Default temporal unit
    "max_train_length": 512,  # Default max train length
    "encoding_window": "full_series",  # Default encoding window
}

# Dataset categories and lists
DATASET_LISTS = {
    "ucr": {
        "name": "UCR Datasets",
        "description": "128 single-variable time series classification datasets",
        "count": 128,
        "type": "classification",
        "variant": "univariate",
        "datasets": [
            # UCR datasets (128 in total)
            "ACSF1", "Adiac", "AllGestureWiimoteX", "AllGestureWiimoteY", 
            "AllGestureWiimoteZ", "ArrowHead", "Beef", "BeetleFly", 
            "BirdChicken", "BME", "Car", "CBF", 
            "Chinatown", "ChlorineConcentration", "CinC_ECG_torso", "Coffee", 
            "Computers", "Cricket_X", "Cricket_Y", "Cricket_Z", 
            "Crop", "DiatomSizeReduction", "DistalPhalanxOutlineAgeGroup", 
            "DistalPhalanxOutlineCorrect", "DistalPhalanxTW", "DodgerLoopDay", 
            "DodgerLoopGame", "DodgerLoopWeekend", "Earthquakes", "ECG200", 
            "ECG5000", "ECGFiveDays", "ElectricDevices", "EOGHorizontalSignal", 
            "EOGVerticalSignal", "EthanolLevel", "FaceAll", "FaceFour", 
            "FacesUCR", "FiftyWords", "Fish", "FordA", "FordB", "FreezerRegularTrain", 
            "FreezerSmallTrain", "Fungi", "GestureMidAirD1", "GestureMidAirD2", 
            "GestureMidAirD3", "GesturePebbleZ1", "GesturePebbleZ2", "GunPoint", 
            "GunPointAgeSpan", "GunPointMaleVersusFemale", "GunPointOldVersusYoung", 
            "Ham", "HandOutlines", "Haptics", "Herring", "HouseTwenty", "InlineSkate", 
            "InsectEPGRegularTrain", "InsectEPGSmallTrain", "InsectWingbeatSound", 
            "ItalyPowerDemand", "LargeKitchenAppliances", "Lightning2", "Lightning7", 
            "Mallat", "Meat", "MedicalImages", "MiddlePhalanxOutlineAgeGroup", 
            "MiddlePhalanxOutlineCorrect", "MiddlePhalanxTW", "MixedShapesRegularTrain", 
            "MixedShapesSmallTrain", "MoteStrain", "NonInvasiveFetalECGThorax1", 
            "NonInvasiveFetalECGThorax2", "OliveOil", "OSULeaf", "PhalangesOutlinesCorrect", 
            "Phoneme", "PickupGestureWiimoteZ", "PigAirwayPressure", "PigArtPressure", 
            "PigCVP", "PLAID", "Plane", "PowerCons", "ProximalPhalanxOutlineAgeGroup", 
            "ProximalPhalanxOutlineCorrect", "ProximalPhalanxTW", "RefrigerationDevices", 
            "Rock", "ScreenType", "SemgHandGenderCh2", "SemgHandMovementCh2", 
            "SemgHandSubjectCh2", "ShakeGestureWiimoteZ", "ShapeletSim", "ShapesAll", 
            "SmallKitchenAppliances", "SmoothSubspace", "SonyAIBORobotSurface1", 
            "SonyAIBORobotSurface2", "StarLightCurves", "Strawberry", "SwedishLeaf", 
            "Symbols", "SyntheticControl", "ToeSegmentation1", "ToeSegmentation2", 
            "Trace", "TwoLeadECG", "TwoPatterns", "UMD", "UWaveGestureLibraryAll", 
            "UWaveGestureLibraryX", "UWaveGestureLibraryY", "UWaveGestureLibraryZ", 
            "Wafer", "Wine", "WordSynonyms", "Worms", "WormsTwoClass", "Yoga"
        ]
    },
    "uea": {
        "name": "UEA Datasets",
        "description": "30 multivariate time series classification datasets",
        "count": 30,
        "type": "classification",
        "variant": "multivariate",
        "datasets": [
            "ArticularyWordRecognition", "AtrialFibrillation", "BasicMotions", 
            "CharacterTrajectories", "Cricket", "DuckDuckGeese", "EigenWorms", 
            "Epilepsy", "ERing", "EthanolConcentration", "FaceDetection", 
            "FingerMovements", "HandMovementDirection", "Handwriting", "Heartbeat", 
            "InsectWingbeat", "JapaneseVowels", "Libras", "LSST", "MotorImagery", 
            "NATOPS", "PEMS-SF", "PenDigits", "PhonemeSpectra", "RacketSports", 
            "SelfRegulationSCP1", "SelfRegulationSCP2", "SpokenArabicDigits", 
            "StandWalkJump", "UWaveGestureLibrary"
        ]
    },
    "forecasting": {
        "name": "Forecasting Datasets",
        "description": "Time series forecasting datasets",
        "count": 4,
        "type": "forecasting",
        "datasets": ["ETTh1", "ETTh2", "ETTm1", "Electricity"]
    },
    "anomaly_detection": {
        "name": "Anomaly Detection Datasets",
        "description": "Time series anomaly detection datasets",
        "count": 2,
        "type": "anomaly_detection",
        "datasets": ["Yahoo", "KPI"]
    }
}

# Anomaly detection settings
ANOMALY_SETTINGS = {
    "Yahoo": {
        "settings": ["normal", "coldstart"]
    },
    "KPI": {
        "settings": ["normal", "coldstart"]
    }
}

# Path configurations
PATH_CONFIG = {
    "data_dir": os.path.join(os.getcwd(), "data"),
    "ckpt_dir": os.path.join(os.getcwd(), "checkpoints"),
    "results_dir": os.path.join(os.getcwd(), "results"),
    "vis_dir": os.path.join(os.getcwd(), "visualizations"),
    "repr_dir": os.path.join(os.getcwd(), "representations"),
    "table_dir": os.path.join(os.getcwd(), "tables")
}

# Visualization settings for reproducing paper figures
VIS_SETTINGS = {
    "figure2": {
        "name": "Figure 2: Noise Robustness Experiment",
        "size": (7, 3),
        "layout": (2, 2),
        "font_sizes": {
            "axis_label": 15,
            "tick": 10,
            "corr": 12
        },
        "colors": {
            "waveform": "blue",
            "heatmap": "RdPu"
        },
        "grid": False,
        "corr_label": "Corr: {:.3f}"
    },
    "figure4": {
        "name": "Figure 4: Critical Difference Diagram",
        "size": (5, 3),
        "font_sizes": {
            "classifier": 10,
            "tick": 8
        },
        "colors": {
            "line": "black"
        }
    },
    "figure5": {
        "name": "Figure 5: Positive Pair Feature Distance Distribution",
        "size": (6, 3),
        "font_sizes": {
            "title": 12,
            "axis_label": 10,
            "tick": 8,
            "legend": 10
        },
        "colors": {
            "histogram": "blue",
            "mean": "black"
        },
        "x_range": (0, 1.5),
        "legend": "--- mean"
    },
    "figure6": {
        "name": "Figure 6: Uniformity Analysis",
        "size": (7, 6),
        "layout": (2, 4),
        "font_sizes": {
            "title": 14,
            "sub_title": 12,
            "axis_label": 10,
            "tick": 8
        },
        "colors": {
            "class1": "purple",
            "class2": "green",
            "class3": "yellow"
        },
        "grid": False,
        "radial_histogram": True
    },
    "figure7_8": {
        "name": "Figures 7-8: Noise Ratio Analysis",
        "figure7_size": (4, 2),
        "figure8_size": (7, 3),
        "font_sizes": {
            "title": 12,
            "axis_label": 10,
            "tick": 8,
            "legend": 10
        },
        "colors": {
            "coinception": "red",
            "ts2vec": "green"
        },
        "noise_range": (0, 0.5),
        "hexagon_3d": True
    }
}

# Table generation settings
TABLE_SETTINGS = {
    "font_size": 10,
    "font_family": "Times New Roman",
    "border_style": {
        "header": "double",
        "rows": "single"
    },
    "highlight_best": True,
    "highlight_color": "red",
    "highlight_weight": "bold",
    "column_width": "uniform",
    "title_alignment": "center",
    "row_height": 1.5
}

# Experiment configurations
EXPERIMENT_CONFIGS = {
    "noise_robustness": {
        "name": "Noise Robustness Experiment",
        "noise_levels": [0.0, 0.1, 0.2, 0.3, 0.4, 0.5],
        "noise_types": ["gaussian", "uniform", "salt_pepper"],
        "metrics": ["correlation", "mse", "mae", "ssim"]
    },
    "classification_performance": {
        "name": "Classification Performance",
        "metrics": ["accuracy", "f1", "precision", "recall"],
        "cross_validation": True,
        "cv_folds": 5
    },
    "forecasting_performance": {
        "name": "Forecasting Performance",
        "metrics": ["mae", "mse", "rmse", "mape"],
        "horizons": [24, 48, 96]
    },
    "anomaly_detection_performance": {
        "name": "Anomaly Detection Performance",
        "metrics": ["auc_roc", "auc_pr", "precision", "recall", "f1"],
        "threshold_methods": ["otsu", "quantile", "dynamic"]
    }
}

# Model comparison settings
MODEL_COMPARISON = {
    "baselines": ["TS2Vec", "InceptionTime", "ResNet", "Transformer", "LSTM", "GRU"],
    "metrics": ["accuracy", "f1", "auc_roc", "mae", "mse", "rmse"],
    "ranking_method": "critical_difference"
}


def get_preset_params():
    """Get the preset parameters as a dictionary."""
    return PRESET_PARAMS.copy()


def get_dataset_list(dataset_type):
    """Get the dataset list for a specific dataset type.
    
    Args:
        dataset_type (str): Dataset type, can be 'ucr', 'uea', 'forecasting', or 'anomaly_detection'.
    
    Returns:
        list: List of datasets for the specified type.
    """
    if dataset_type not in DATASET_LISTS:
        raise ValueError(f"Unknown dataset type: {dataset_type}")
    return DATASET_LISTS[dataset_type]["datasets"].copy()


def get_all_datasets():
    """Get all datasets across all categories.
    
    Returns:
        dict: Dictionary containing all datasets grouped by category.
    """
    return DATASET_LISTS.copy()


def get_path_config():
    """Get the path configuration.
    
    Returns:
        dict: Path configuration dictionary.
    """
    return PATH_CONFIG.copy()


def create_directories():
    """Create all necessary directories for the experiment."""
    for path in PATH_CONFIG.values():
        os.makedirs(path, exist_ok=True)


def get_vis_settings(figure_name):
    """Get visualization settings for a specific figure.
    
    Args:
        figure_name (str): Figure name, can be 'figure2', 'figure4', 'figure5', 'figure6', or 'figure7_8'.
    
    Returns:
        dict: Visualization settings for the specified figure.
    """
    if figure_name not in VIS_SETTINGS:
        raise ValueError(f"Unknown figure name: {figure_name}")
    return VIS_SETTINGS[figure_name].copy()


if __name__ == "__main__":
    # Print preset information
    print("CoInception Analysis Preset Configuration")
    print("=" * 50)
    print("\nPreset Parameters:")
    for key, value in PRESET_PARAMS.items():
        print(f"  {key}: {value}")
    
    print("\nDataset Categories:")
    for category, info in DATASET_LISTS.items():
        print(f"  {category}: {info['name']} ({info['count']} datasets)")
    
    print("\nPath Configuration:")
    for key, path in PATH_CONFIG.items():
        print(f"  {key}: {path}")
    
    print("\nVisualization Figures:")
    for figure_name, info in VIS_SETTINGS.items():
        print(f"  {figure_name}: {info['name']}")
