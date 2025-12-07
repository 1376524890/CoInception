#!/usr/bin/env python3
"""
Check if forecasting dataset evaluation results have been generated.
"""

import os
import glob

def main():
    """Main function to check forecasting dataset evaluation results."""
    datasets = ['ETTh1', 'ETTh2', 'ETTm1', 'Electricity']
    found = False
    
    for ds in datasets:
        print(f'Checking for {ds} evaluation results...')
        ds_dirs = glob.glob(f'/home/codeserver/CoInception/training/{ds}*')
        if ds_dirs:
            found = True
            print(f'Found {len(ds_dirs)} directories for {ds}:')
            for d in ds_dirs:
                eval_file = os.path.join(d, 'eval_res.pkl')
                has_eval = os.path.exists(eval_file)
                print(f'  {d}')
                print(f'  Has eval_res.pkl: {has_eval}')
        else:
            print(f'No directories found for {ds}')
    
    if not found:
        print('\nNo forecasting dataset evaluation results found.')
    else:
        print('\nSome forecasting dataset evaluation results were found.')

if __name__ == '__main__':
    main()
