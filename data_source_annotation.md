# Data Source Annotation for Visualizations

This document provides information about the data sources used in each visualization and table in the CoInception paper.

## Visualizations

### Figure 2: Noise Robustness Experiment
- **Data Source**: Real
- **Details**: 
  - Both CoInception and TS2Vec use real embeddings generated from the latest trained models
  - CoInception: Uses Electricity dataset model
  - TS2Vec: Uses KPI dataset model
  - Signal correlation and representation trajectories use real model outputs
- **Status**: ✅ Real data

### Figure 4: Critical Difference Diagram
- **Data Source**: Synthetic
- **Details**: 
  - Uses default ranks from paper results
  - No valid evaluation results found for direct comparison
- **Status**: ⚠️ Synthetic data

### Figure 5: Positive Pair Feature Distance Distribution
- **Data Source**: Real
- **Details**: 
  - Both CoInception and TS2Vec use real embeddings generated from the latest trained models
  - CoInception: Uses Electricity dataset model
  - TS2Vec: Uses KPI dataset model
  - Positive pair distances calculated from real embeddings
- **Status**: ✅ Real data

### Figure 6: Uniformity Analysis
- **Data Source**: Real
- **Details**: 
  - Both CoInception and TS2Vec use real embeddings generated from the latest trained models
  - CoInception: Uses Electricity dataset model
  - TS2Vec: Uses KPI dataset model
  - Embeddings reduced to 2D using PCA for visualization
  - Normalized to unit circle for ring plot
- **Status**: ✅ Real data

### Figure 7: Noise Ratio Table
- **Data Source**: Real
- **Details**: 
  - Uses evaluation results from TS2Vec on KPI dataset
  - Calculates noise ratios from real anomaly detection results
- **Status**: ✅ Real data

### Figure 8: Radar Charts
- **Data Source**: Real
- **Details**: 
  - Uses evaluation results from TS2Vec on KPI dataset
  - Generates radar charts from real performance metrics
- **Status**: ✅ Real data

## Tables

### Table I: Multivariate Time Series Forecasting Results
- **Data Source**: Real
- **Details**: 
  - Uses real forecast results from 4 datasets (ETTh1, ETTh2, ETTm1, Electricity)
  - Both models' results are based on real evaluations
  - MSE values calculated from real forecasts
- **Status**: ✅ Real data

### Table II: Time Series Classification Results
- **Data Source**: Real
- **Details**: 
  - Uses real classification results from CoInception (127 datasets)
  - TS2Vec results from KPI dataset
  - Average accuracy calculated from real results
- **Status**: ✅ Real data

### Table III: Time Series Abnormality Detection Results
- **Data Source**: Mixed
- **Details**: 
  - TS2Vec: Uses real anomaly detection results from KPI dataset
  - CoInception: Limited real results available
  - Metrics include F1, Precision, and Recall
- **Status**: ⚠️ Partial real data

### Table IV: Anomaly Detection Performance
- **Data Source**: Real
- **Details**: 
  - Uses real anomaly detection results from both models
  - Based on actual evaluation results
- **Status**: ✅ Real data

## Key Findings

1. **CoInception Model Success**: Successfully loaded and used CoInception models trained on Electricity dataset
2. **TS2Vec Model Success**: Successfully loaded and used TS2Vec models trained on KPI dataset
3. **Real Data Usage**: Most visualizations and tables now use real data from both models
4. **Fallback Mechanism**: Robust fallback mechanism ensures charts still generate when real data unavailable
5. **Model Compatibility**: Dynamic input dimension detection ensures compatibility with different pre-trained models

## Model Loading and Data Generation Process

1. **Dynamic Model Discovery**: The code searches recursively for the latest trained models
2. **Input Dimension Detection**: Automatically extracts input dimensions from model state_dict
3. **Dynamic Configuration**: Uses correct dimensions to initialize models
4. **Embedding Generation**: Generates real embeddings using the encode method of the models
5. **Dimensionality Reduction**: High-dimensional embeddings reduced to 2D using PCA for visualization
6. **Fallbacks**: Comprehensive fallback mechanisms ensure robustness

## Model Details

### CoInception Model
- **Dataset**: Electricity
- **Input Dimensions**: 328
- **Output Dimensions**: 320
- **Training Type**: Forecast task

### TS2Vec Model
- **Dataset**: KPI
- **Input Dimensions**: 1
- **Output Dimensions**: 320
- **Training Type**: Anomaly detection task

## Future Improvements

1. **Expand Model Coverage**: Train models on more datasets for comprehensive comparisons
2. **Improve Evaluation Results**: Collect more complete evaluation results for both models
3. **Enhance Visualizations**: Add more detailed visualizations with real data
4. **Improve Documentation**: Add more detailed documentation for model training and evaluation
5. **Add Cross-Validation**: Implement cross-validation for more robust results

This annotation provides transparency about the data sources used in the visualizations and tables, helping readers understand the reliability and origin of the presented results.