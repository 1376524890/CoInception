#!/usr/bin/env python3
"""
Test script for the training completion check functions
"""
import os
import sys

# Add the project root to the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import our functions from run_all_datasets.py
try:
    from run_all_datasets import check_training_completion, check_anomaly_completion, check_all_training_completion
    
    # Create a simple mock object to simulate run_args
    class MockArgs:
        def __init__(self, force_retrain='false'):
            self.force_retrain = force_retrain
    
    print("Testing training completion check functions...")
    print("=" * 50)
    
    # Test 1: Check a dataset that should exist (FordA)
    print("\nTest 1: Check FordA dataset completion")
    run_args = MockArgs('false')
    result = check_training_completion('FordA', run_args)
    print(f"FordA completion result: {result}")
    
    # Test 2: Check a dataset that likely doesn't exist
    print("\nTest 2: Check non-existent dataset")
    result = check_training_completion('NonExistentDataset', run_args)
    print(f"NonExistentDataset completion result: {result}")
    
    # Test 3: Check anomaly detection completion
    print("\nTest 3: Check anomaly detection (SMD normal)")
    result = check_anomaly_completion('SMD', 'normal', run_args)
    print(f"SMD normal completion result: {result}")
    
    # Test 4: Test with force_retrain enabled
    print("\nTest 4: Check with force_retrain=true")
    run_args_force = MockArgs('true')
    result = check_training_completion('FordA', run_args_force)
    print(f"FordA completion result (force=true): {result}")
    
    print("\n" + "=" * 50)
    print("Training completion check tests completed!")
    
except ImportError as e:
    print(f"Import error: {e}")
    print("Make sure run_all_datasets.py is in the same directory")
except Exception as e:
    print(f"Error during testing: {e}")