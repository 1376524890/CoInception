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

# Parse command line arguments
GPU=0
PARALLEL=1
DELAY=0
SKIP_DATASETS=false
SKIP_TABLES=false
SKIP_FIGURES=false

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
        -h|--help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Complete Analysis Run Script for CoInception"
            echo ""
            echo "Options:"
            echo "  -g, --gpu GPU_ID        GPU ID to use (default: 0)"
            echo "  -p, --parallel NUM      Number of parallel processes (default: 1)"
            echo "  -d, --delay SECONDS     Delay between runs in seconds (default: 0)"
            echo "  --skip-datasets         Skip running datasets (use existing results)"
            echo "  --skip-tables           Skip generating tables"
            echo "  --skip-figures          Skip generating figures"
            echo "  -h, --help              Show this help message and exit"
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
    log "${BLUE}INFO${NC}" "="
    
    # Change to project root directory
    SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
    cd "$PROJECT_ROOT"
    log "${BLUE}INFO${NC}" "Working directory: $(pwd)"
    
    # Step 1: Run all datasets with preset parameters
    if [ "$SKIP_DATASETS" = false ]; then
        log "${BLUE}INFO${NC}" "Step 1: Running all datasets with preset parameters"
        run_command "python run_all_datasets.py --gpu $GPU --parallel $PARALLEL --delay $DELAY" "Running all datasets"
    else
        log "${YELLOW}WARNING${NC}" "Step 1 skipped: Running datasets disabled by --skip-datasets flag"
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
    echo -e "Results Directory: ${BLUE}results/${NC}"
    echo -e "Visualizations Directory: ${BLUE}visualizations/${NC}"
    echo -e "Tables Directory: ${BLUE}tables/${NC}"
    echo -e "${GREEN}=================================${NC}"
}

# Run the main function
main