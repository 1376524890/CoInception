#!/bin/bash

# CoInception Complete Analysis Run Script
# This script executes the full analysis pipeline as specified in the plan document.

set -e

# Color definitions
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Log file
LOG_FILE="analysis_run_$(date +%Y%m%d_%H%M%S).log"

# Project directories
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TRAINING_DIR="$PROJECT_ROOT/training"
RESULTS_DIR="$PROJECT_ROOT/results"
TABLES_DIR="$PROJECT_ROOT/tables"

# Function to log messages
log() {
    local level=$1
    local message=$2
    local timestamp=$(date +"%Y-%m-%d %H:%M:%S")
    echo -e "[$timestamp] [$level] $message" | tee -a "$LOG_FILE"
}

# Function to run a command with logging
run_command() {
    local command=$1
    local description=$2
    
    log "${BLUE}INFO${NC}" "Starting: $description"
    log "${BLUE}INFO${NC}" "Command: $command"
    
    if eval "$command"; then
        log "${GREEN}SUCCESS${NC}" "Completed: $description"
        return 0
    else
        log "${RED}ERROR${NC}" "Failed: $description"
        return 1
    fi
}

# Function to check if training is completed for a dataset
check_training_completion() {
    local dataset_name=$1
    local training_subdir="$TRAINING_DIR/${dataset_name}__"
    
    # Check if any training directory exists for this dataset
    local training_dirs=($(find "$TRAINING_DIR" -maxdepth 1 -name "${dataset_name}__*" -type d 2>/dev/null || true))
    
    if [ ${#training_dirs[@]} -eq 0 ]; then
        log "${BLUE}INFO${NC}" "No training directory found for dataset: $dataset_name"
        return 1  # Not completed
    fi
    
    # Check for model.pkl and eval_res.pkl in the most recent training directory
    local latest_dir="${training_dirs[-1]}"
    local model_file="$latest_dir/model.pkl"
    local eval_file="$latest_dir/eval_res.pkl"
    
    if [ -f "$model_file" ] && [ -f "$eval_file" ]; then
        log "${GREEN}SUCCESS${NC}" "Training completed for dataset: $dataset_name (found: $latest_dir)"
        return 0  # Completed
    else
        log "${YELLOW}WARNING${NC}" "Incomplete training for dataset: $dataset_name (missing files in $latest_dir)"
        return 1  # Not completed
    fi
}

# Function to check if anomaly detection is completed
check_anomaly_completion() {
    local dataset_name=$1
    local setting=$2  # normal or coldstart
    local training_pattern="${dataset_name}__*${setting}*"
    
    # Find training directories matching the pattern
    local training_dirs=($(find "$TRAINING_DIR" -maxdepth 1 -name "${training_pattern}" -type d 2>/dev/null || true))
    
    if [ ${#training_dirs[@]} -eq 0 ]; then
        log "${BLUE}INFO${NC}" "No training directory found for anomaly dataset: $dataset_name ($setting)"
        return 1  # Not completed
    fi
    
    # Check for model.pkl and eval_res.pkl in the most recent training directory
    local latest_dir="${training_dirs[-1]}"
    local model_file="$latest_dir/model.pkl"
    local eval_file="$latest_dir/eval_res.pkl"
    
    if [ -f "$model_file" ] && [ -f "$eval_file" ]; then
        log "${GREEN}SUCCESS${NC}" "Anomaly training completed for: $dataset_name ($setting) (found: $latest_dir)"
        return 0  # Completed
    else
        log "${YELLOW}WARNING${NC}" "Incomplete anomaly training for: $dataset_name ($setting) (missing files in $latest_dir)"
        return 1  # Not completed
    fi
}

# Function to check if all datasets are completed
check_all_training_completion() {
    log "${BLUE}INFO${NC}" "Checking training completion status..."
    
    # Check classification datasets (UCR)
    local ucr_datasets=("FordA" "ECG200" "Lightning7" "Earthquakes" "Wine" "Wafer" "TwoLeadECG" "FaceDetection" "UWaveGestureLibrary" "Mallat")
    local uea_datasets=("EthanolConcentration" "FaceDetection" "CharacterTrajectories" "StandWalkJump" "Heartbeat" "SpokenArabicDigits" "UWaveGestureLibrary" "NATOPS" "JapaneseVowels" "PenDigits")
    local forecasting_datasets=("electricity" "kpi" "yahoo")
    
    local incomplete_count=0
    
    # Check UCR datasets
    for dataset in "${ucr_datasets[@]}"; do
        if ! check_training_completion "$dataset"; then
            ((incomplete_count++))
        fi
    done
    
    # Check UEA datasets  
    for dataset in "${uea_datasets[@]}"; do
        if ! check_training_completion "$dataset"; then
            ((incomplete_count++))
        fi
    done
    
    # Check forecasting datasets
    for dataset in "${forecasting_datasets[@]}"; do
        if ! check_training_completion "$dataset"; then
            ((incomplete_count++))
        fi
    done
    
    # Check anomaly detection datasets
    local anomaly_datasets=("SMD" "SWaT" "WADI" "NIPS" "ISIS")
    local anomaly_settings=("normal" "coldstart")
    
    for dataset in "${anomaly_datasets[@]}"; do
        for setting in "${anomaly_settings[@]}"; do
            if ! check_anomaly_completion "$dataset" "$setting"; then
                ((incomplete_count++))
            fi
        done
    done
    
    if [ $incomplete_count -eq 0 ]; then
        log "${GREEN}SUCCESS${NC}" "All training completed! ($incomplete_count incomplete)"
        return 0  # All completed
    else
        log "${YELLOW}WARNING${NC}" "Training incomplete: $incomplete_count datasets/settings need training"
        return 1  # Some incomplete
    fi
}

# Parse command line arguments
GPU=0
PARALLEL=1
DELAY=0
SKIP_DATASETS=false
SKIP_TABLES=false
SKIP_FIGURES=false
AUTO_SKIP_COMPLETED=false
FORCE_RETRAIN=false

while [[ $# -gt 0 ]]; do
    case $1 in
        -g|--gpu)
            GPU="$2"
            shift 2
            ;;
        -p|--parallel)
            PARALLEL="$2"
            shift 2
            ;;
        -d|--delay)
            DELAY="$2"
            shift 2
            ;;
        --skip-datasets)
            SKIP_DATASETS=true
            shift 1
            ;;
        --skip-tables)
            SKIP_TABLES=true
            shift 1
            ;;
        --skip-figures)
            SKIP_FIGURES=true
            shift 1
            ;;
        --auto-skip-completed)
            AUTO_SKIP_COMPLETED=true
            shift 1
            ;;
        --force-retrain)
            FORCE_RETRAIN=true
            shift 1
            ;;
        -h|--help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Complete Analysis Run Script for CoInception"
            echo ""
            echo "Options:"
            echo "  -g, --gpu GPU_ID            GPU ID to use (default: 0)"
            echo "  -p, --parallel NUM          Number of parallel processes (default: 1)"
            echo "  -d, --delay SECONDS         Delay between runs in seconds (default: 0)"
            echo "  --skip-datasets             Skip running datasets (use existing results)"
            echo "  --skip-tables               Skip generating tables"
            echo "  --skip-figures              Skip generating figures"
            echo "  --auto-skip-completed       Auto-detect and skip completed training"
            echo "  --force-retrain             Force retraining even if completed"
            echo "  -h, --help                  Show this help message and exit"
            echo ""
            exit 0
            ;;
        *)
            log "${RED}ERROR${NC}" "Unknown option: $1"
            log "${BLUE}INFO${NC}" "Use --help for usage information"
            exit 1
            ;;
    esac
done

# Main function
main() {
    log "${BLUE}INFO${NC}" "Starting CoInception Complete Analysis Pipeline"
    log "${BLUE}INFO${NC}" "Analysis Run Log: $LOG_FILE"
    log "${BLUE}INFO${NC}" "="
    log "${BLUE}INFO${NC}" "Command line arguments:"
    log "${BLUE}INFO${NC}" "  GPU: $GPU"
    log "${BLUE}INFO${NC}" "  Parallel: $PARALLEL"
    log "${BLUE}INFO${NC}" "  Delay: $DELAY"
    log "${BLUE}INFO${NC}" "  Skip datasets: $SKIP_DATASETS"
    log "${BLUE}INFO${NC}" "  Skip tables: $SKIP_TABLES"
    log "${BLUE}INFO${NC}" "  Skip figures: $SKIP_FIGURES"
    log "${BLUE}INFO${NC}" "  Auto-skip completed: $AUTO_SKIP_COMPLETED"
    log "${BLUE}INFO${NC}" "  Force retrain: $FORCE_RETRAIN"
    log "${BLUE}INFO${NC}" "="
    
    # Change to project root directory
    cd "$PROJECT_ROOT"
    log "${BLUE}INFO${NC}" "Working directory: $(pwd)"
    
    # Create necessary directories
    mkdir -p "$TRAINING_DIR" "$RESULTS_DIR" "$TABLES_DIR"
    
    # Step 0: Check training completion status
    if [ "$SKIP_DATASETS" = false ] && [ "$AUTO_SKIP_COMPLETED" = true ]; then
        log "${BLUE}INFO${NC}" "Step 0: Checking training completion status..."
        if check_all_training_completion; then
            log "${GREEN}SUCCESS${NC}" "All training already completed! Skipping dataset training."
            SKIP_DATASETS=true
        else
            log "${YELLOW}WARNING${NC}" "Some training incomplete. Proceeding with training."
        fi
    fi
    
    # Step 1: Run all datasets with preset parameters
    if [ "$SKIP_DATASETS" = false ]; then
        log "${BLUE}INFO${NC}" "Step 1: Running all datasets with preset parameters"
        
        if [ "$FORCE_RETRAIN" = true ]; then
            log "${YELLOW}WARNING${NC}" "Force retrain enabled - will retrain all datasets"
        fi
        
        run_command "python run_all_datasets.py --gpu $GPU --parallel $PARALLEL --delay $DELAY --force-retrain $FORCE_RETRAIN" "Running all datasets"
    else
        log "${YELLOW}WARNING${NC}" "Step 1 skipped: Running datasets disabled"
    fi
    
    # Step 2: Generate tables from results
    if [ "$SKIP_TABLES" = false ]; then
        log "${BLUE}INFO${NC}" "Step 2: Generating tables from results"
        run_command "python generate_tables.py" "Generating all tables"
    else
        log "${YELLOW}WARNING${NC}" "Step 2 skipped: Generating tables disabled by --skip-tables flag"
    fi
    
    # Step 3: Generate visualizations
    if [ "$SKIP_FIGURES" = false ]; then
        log "${BLUE}INFO${NC}" "Step 3: Generating visualizations"
        run_command "python visualize_figures.py" "Generating all figures"
        
        # Step 4: Generate individual figures with specific scripts
        log "${BLUE}INFO${NC}" "Step 4: Generating individual figures with specific scripts"
        run_command "python vis_figure2.py" "Generating Figure 2: Noise Robustness"
        run_command "python vis_figure4.py" "Generating Figure 4: Critical Difference"
        run_command "python vis_figure5.py" "Generating Figure 5: Feature Distance Distribution"
        run_command "python vis_figure6.py" "Generating Figure 6: Uniformity Analysis"
        run_command "python vis_figure7_8.py" "Generating Figures 7-8: Noise Ratio Analysis"
        run_command "python vis_trajectory.py" "Generating Trajectory Visualizations"
    else
        log "${YELLOW}WARNING${NC}" "Step 3-4 skipped: Generating figures disabled by --skip-figures flag"
    fi
    
    # Step 5: Final summary
    log "${BLUE}INFO${NC}" "="
    log "${GREEN}SUCCESS${NC}" "CoInception Complete Analysis Pipeline Finished"
    log "${BLUE}INFO${NC}" "Log File: $LOG_FILE"
    log "${BLUE}INFO${NC}" "Check the results, visualizations, and tables directories for output files"
    
    # Print summary to console
    echo -e "\n${GREEN}=== Analysis Pipeline Complete ===${NC}"
    echo -e "Log File: ${BLUE}$LOG_FILE${NC}"
    echo -e "Results Directory: ${BLUE}$RESULTS_DIR/${NC}"
    echo -e "Training Directory: ${BLUE}$TRAINING_DIR/${NC}"
    echo -e "Visualizations Directory: ${BLUE}visualizations/${NC}"
    echo -e "Tables Directory: ${BLUE}$TABLES_DIR/${NC}"
    echo -e "${GREEN}=================================${NC}"
}

# Run the main function
main