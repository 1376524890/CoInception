#!/usr/bin/env python3
"""
Test script for full_dataset_analysis.py

This script tests the full_dataset_analysis.py functionality by processing a single dataset.
"""

import os
import sys
import subprocess

# Test with a single dataset
TEST_DATASET = "FordA"  # A small UCR dataset
TEST_CATEGORY = "ucr"

# Run the full dataset analysis script with a single dataset
cmd = [
    sys.executable, "full_dataset_analysis.py",
    "--test", TEST_DATASET,
    "--category", TEST_CATEGORY
]

print(f"Running test command: {' '.join(cmd)}")

# Run the command
result = subprocess.run(
    cmd,
    check=False,
    capture_output=True,
    text=True,
    cwd=os.getcwd()
)

print("\n=== Command Output ===")
print(result.stdout)

if result.stderr:
    print("\n=== Command Error ===")
    print(result.stderr)

print(f"\nReturn code: {result.returncode}")
