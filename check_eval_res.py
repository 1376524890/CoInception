#!/usr/bin/env python3
"""
Check the content of eval_res.pkl files to understand the evaluation results structure.
"""

import pickle
import os

def check_eval_res(file_path):
    """Check the content of a specific eval_res.pkl file."""
    print(f'Loading {file_path}...')
    try:
        with open(file_path, 'rb') as f:
            data = pickle.load(f)
        
        print(f'Data type: {type(data)}')
        if isinstance(data, dict):
            print(f'Data keys: {data.keys()}')
            for key, value in data.items():
                print(f'  {key}: {type(value)}')
                if isinstance(value, dict):
                    print(f'    Subkeys: {value.keys()}')
                elif isinstance(value, (list, tuple)):
                    print(f'    Length: {len(value)}')
                else:
                    print(f'    Value: {value}')
        else:
            print(f'Data content: {data}')
    except Exception as e:
        print(f'Error loading {file_path}: {e}')

def main():
    """Main function to check all eval_res.pkl files."""
    # Check a sample eval_res.pkl file
    sample_file = '/home/codeserver/CoInception/training/ACSF1__run_ACSF1_20251204_043031/eval_res.pkl'
    if os.path.exists(sample_file):
        check_eval_res(sample_file)
    else:
        print(f'Sample file {sample_file} not found.')
    
    # Check if there are any forecasting-related eval_res.pkl files
    print('\nChecking for forecasting-related eval_res.pkl files...')
    import glob
    eval_files = glob.glob('/home/codeserver/CoInception/training/*/eval_res.pkl')
    print(f'Found {len(eval_files)} eval_res.pkl files.')
    
    # Check the first few files to understand the structure
    for i, file_path in enumerate(eval_files[:3]):
        print(f'\n=== File {i+1}: {file_path} ===')
        check_eval_res(file_path)

if __name__ == '__main__':
    main()
