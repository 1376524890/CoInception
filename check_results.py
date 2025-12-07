#!/usr/bin/env python3
"""
检查噪声比率分析实验结果文件的内容
"""

import pickle
import os

# 结果文件路径
results_file = '/home/codeserver/CoInception/results/all_noise_results_ETTm1.pkl'

# 检查文件是否存在
if os.path.exists(results_file):
    print(f"结果文件存在: {results_file}")
    
    # 加载结果文件
    with open(results_file, 'rb') as f:
        all_results = pickle.load(f)
    
    # 打印结果的基本信息
    print(f"结果类型: {type(all_results)}")
    print(f"结果键: {list(all_results.keys())}")
    
    # 打印每个模型的结果
    for model_type, results in all_results.items():
        print(f"\n{model_type} 模型结果:")
        print(f"  结果类型: {type(results)}")
        print(f"  结果键: {list(results.keys())}")
        
        # 打印每个噪声比率的结果
        for noise_ratio, result in results.items():
            print(f"  噪声比率 {noise_ratio}%:")
            print(f"    结果类型: {type(result)}")
            print(f"    结果键: {list(result.keys())}")
            print(f"    MSE: {result['MSE']:.4f}")
            print(f"    MAE: {result['MAE']:.4f}")
            print(f"    mse_per_horizon: {result['mse_per_horizon']}")
            print(f"    mae_per_horizon: {result['mae_per_horizon']}")
else:
    print(f"结果文件不存在: {results_file}")
