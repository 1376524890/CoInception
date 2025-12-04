#!/usr/bin/env python3
"""
Table Generation Script for CoInception

This script generates tables from the CoInception paper, matching the exact format requirements.
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import argparse
from analysis_preset import PATH_CONFIG, TABLE_SETTINGS, get_preset_params
from utils.visualization import set_font_sizes, save_figure


def generate_table_i():
    """Generate Table I from the CoInception paper."""
    # Sample data for Table I
    # In a real scenario, this would come from evaluation results
    datasets = [
        "FordA", "FordB", "Coffee", "ECGFiveDays", "ElectricDevices",
        "Beef", "CBF", "Ham", "Plane", "OliveOil", "Meat", "GunPoint"
    ]
    
    coin_inception = [0.98, 0.97, 1.00, 0.99, 0.95, 0.98, 1.00, 0.98, 1.00, 1.00, 0.99, 0.99]
    ts2vec = [0.96, 0.94, 0.99, 0.97, 0.92, 0.96, 0.99, 0.96, 0.99, 0.99, 0.97, 0.98]
    inception_time = [0.95, 0.93, 0.99, 0.96, 0.91, 0.95, 0.99, 0.95, 0.99, 0.99, 0.96, 0.97]
    resnet = [0.94, 0.92, 0.98, 0.95, 0.90, 0.94, 0.98, 0.94, 0.98, 0.98, 0.95, 0.96]
    
    return {
        "title": "Table I: Classification Accuracy on Selected UCR Datasets",
        "columns": ["Dataset", "CoInception", "TS2Vec", "InceptionTime", "ResNet"],
        "data": list(zip(datasets, coin_inception, ts2vec, inception_time, resnet)),
        "highlight_column": 1  # Highlight best result (CoInception)
    }


def generate_table_ii():
    """Generate Table II from the CoInception paper."""
    # Sample data for Table II
    datasets = ["ArticularyWordRecognition", "AtrialFibrillation", "BasicMotions",
                "CharacterTrajectories", "Cricket", "DuckDuckGeese", "EigenWorms"]
    
    coin_inception = [0.95, 0.98, 1.00, 0.99, 0.97, 0.96, 0.99]
    ts2vec = [0.92, 0.95, 0.98, 0.97, 0.94, 0.93, 0.97]
    inception_time = [0.91, 0.94, 0.97, 0.96, 0.93, 0.92, 0.96]
    resnet = [0.90, 0.93, 0.96, 0.95, 0.92, 0.91, 0.95]
    
    return {
        "title": "Table II: Classification Accuracy on Selected UEA Datasets",
        "columns": ["Dataset", "CoInception", "TS2Vec", "InceptionTime", "ResNet"],
        "data": list(zip(datasets, coin_inception, ts2vec, inception_time, resnet)),
        "highlight_column": 1  # Highlight best result (CoInception)
    }


def generate_table_iii():
    """Generate Table III from the CoInception paper."""
    # Sample data for Table III
    datasets = ["ETTh1", "ETTh2", "ETTm1", "Electricity"]
    horizons = [24, 48, 96]
    
    # Create multi-level columns
    columns = ["Dataset"] + [f"{h}-step" for h in horizons]
    
    # Sample accuracy data
    coin_inception = [0.92, 0.88, 0.83, 0.95, 0.90, 0.85, 0.93, 0.89, 0.84, 0.94, 0.90, 0.86]
    ts2vec = [0.88, 0.83, 0.77, 0.91, 0.86, 0.80, 0.89, 0.84, 0.79, 0.90, 0.85, 0.80]
    
    # Reshape data for table
    coin_data = [coin_inception[i:i+3] for i in range(0, len(coin_inception), 3)]
    ts_data = [ts2vec[i:i+3] for i in range(0, len(ts2vec), 3)]
    
    # Create table data
    table_data = []
    for i, dataset in enumerate(datasets):
        row = [dataset] + coin_data[i] + ts_data[i]
        table_data.append(row)
    
    return {
        "title": "Table III: Forecasting Performance",
        "columns": columns,
        "data": table_data,
        "highlight_column": 1  # Highlight best result (CoInception)
    }


def generate_table_iv():
    """Generate Table IV from the CoInception paper."""
    # Sample data for Table IV
    datasets = ["Yahoo", "KPI"]
    settings = ["Normal", "Coldstart"]
    
    # Sample AUC-ROC data
    coin_inception = [0.95, 0.92, 0.93, 0.89]
    ts2vec = [0.90, 0.86, 0.88, 0.84]
    
    # Create table data
    table_data = []
    idx = 0
    for dataset in datasets:
        for setting in settings:
            table_data.append([dataset, setting, coin_inception[idx], ts2vec[idx]])
            idx += 1
    
    return {
        "title": "Table IV: Anomaly Detection Performance",
        "columns": ["Dataset", "Setting", "CoInception", "TS2Vec"],
        "data": table_data,
        "highlight_column": 2  # Highlight best result (CoInception)
    }


def create_latex_table(table_data, output_file=None):
    """Create a LaTeX table from data.
    
    Args:
        table_data (dict): Table data dictionary.
        output_file (str): Output file path for LaTeX table.
        
    Returns:
        str: LaTeX table string.
    """
    title = table_data["title"]
    columns = table_data["columns"]
    data = table_data["data"]
    highlight_column = table_data.get("highlight_column", None)
    
    # Create LaTeX table header
    latex_str = f"\\begin{{table}}[h]\\centering\\caption{{{title}}}\\vspace{{0.5em}}\\n"
    latex_str += f"\\begin{{tabular}}{{{'l' + 'c' * (len(columns) - 1)}}}\\hline\\hline\\n"
    
    # Add column headers
    latex_str += " & ".join(columns) + " \\\\\\hline\\n"
    
    # Add data rows
    for row in data:
        row_str = []
        for i, val in enumerate(row):
            if i == highlight_column and isinstance(val, float):
                # Highlight best result with red bold
                row_str.append(f"\\textbf{{\\textcolor{{red}}{{{val:.2f}}}}}")
            elif isinstance(val, float):
                row_str.append(f"{val:.2f}")
            else:
                row_str.append(str(val))
        latex_str += " & ".join(row_str) + " \\\\\\hline\\n"
    
    # Close LaTeX table
    latex_str += f"\\end{{tabular}}\\end{{table}}"
    
    # Save to file if specified
    if output_file:
        # Ensure directory exists
        dir_path = os.path.dirname(output_file)
        if dir_path:
            os.makedirs(dir_path, exist_ok=True)
        
        with open(output_file, "w") as f:
            f.write(latex_str)
        print(f"Saved LaTeX table to: {output_file}")
    
    return latex_str


def create_png_table(table_data, output_file):
    """Create a PNG table from data.
    
    Args:
        table_data (dict): Table data dictionary.
        output_file (str): Output file path for PNG table.
    """
    # Get table settings
    table_settings = TABLE_SETTINGS
    
    # Set font sizes
    set_font_sizes(
        axis_label=table_settings["font_size"],
        tick=table_settings["font_size"],
        title=table_settings["font_size"] + 2
    )
    
    # Create pandas DataFrame
    df = pd.DataFrame(table_data["data"], columns=table_data["columns"])
    
    # Create figure
    fig, ax = plt.subplots(figsize=(8, len(df) * 0.4 + 1))
    ax.axis('tight')
    ax.axis('off')
    
    # Create table
    table = ax.table(
        cellText=df.values,
        colLabels=df.columns,
        cellLoc='center',
        loc='center'
    )
    
    # Set table properties
    table.auto_set_font_size(False)
    table.set_fontsize(table_settings["font_size"])
    table.scale(1.2, 1.5)
    
    # Add title
    ax.set_title(table_data["title"], fontsize=table_settings["font_size"] + 2)
    
    # Save table as PNG
    save_figure(fig, output_file)
    print(f"Saved PNG table to: {output_file}")


def generate_all_tables():
    """Generate all tables from the CoInception paper."""
    print("Generating all CoInception tables...")
    print("=" * 50)
    
    # Create table directory if it doesn't exist
    os.makedirs(PATH_CONFIG["table_dir"], exist_ok=True)
    
    # Generate each table
    tables = {
        "table_i": generate_table_i,
        "table_ii": generate_table_ii,
        "table_iii": generate_table_iii,
        "table_iv": generate_table_iv
    }
    
    for table_name, table_func in tables.items():
        print(f"\nGenerating {table_name}...")
        
        # Generate table data
        table_data = table_func()
        
        # Generate LaTeX table
        latex_file = os.path.join(PATH_CONFIG["table_dir"], f"{table_name}.tex")
        create_latex_table(table_data, latex_file)
        
        # Generate PNG table
        png_file = os.path.join(PATH_CONFIG["table_dir"], f"{table_name}.png")
        create_png_table(table_data, png_file)
    
    print("\n" + "=" * 50)
    print("All tables generated successfully!")
    print(f"Tables saved to: {PATH_CONFIG['table_dir']}")


def generate_specific_table(table_name):
    """Generate a specific table from the CoInception paper.
    
    Args:
        table_name (str): Name of the table to generate (table_i, table_ii, table_iii, table_iv).
    """
    # Create table directory if it doesn't exist
    os.makedirs(PATH_CONFIG["table_dir"], exist_ok=True)
    
    # Map table names to functions
    tables = {
        "table_i": generate_table_i,
        "table_ii": generate_table_ii,
        "table_iii": generate_table_iii,
        "table_iv": generate_table_iv
    }
    
    if table_name not in tables:
        print(f"Error: Unknown table name '{table_name}'")
        print("Available tables: table_i, table_ii, table_iii, table_iv")
        return False
    
    print(f"Generating {table_name}...")
    
    # Generate table data
    table_func = tables[table_name]
    table_data = table_func()
    
    # Generate LaTeX table
    latex_file = os.path.join(PATH_CONFIG["table_dir"], f"{table_name}.tex")
    create_latex_table(table_data, latex_file)
    
    # Generate PNG table
    png_file = os.path.join(PATH_CONFIG["table_dir"], f"{table_name}.png")
    create_png_table(table_data, png_file)
    
    print(f"\n{table_name} generated successfully!")
    print(f"Table saved to: {PATH_CONFIG['table_dir']}")
    
    return True


def main():
    """Main function to parse arguments and generate tables."""
    parser = argparse.ArgumentParser(description="Generate CoInception paper tables")
    parser.add_argument('--all', action='store_true', help='Generate all tables')
    parser.add_argument('--table', type=str, help='Generate a specific table (table_i, table_ii, table_iii, table_iv)')
    
    args = parser.parse_args()
    
    if args.all:
        generate_all_tables()
    elif args.table:
        generate_specific_table(args.table)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
