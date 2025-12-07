#!/usr/bin/env python3
"""
加载已保存的 CoInception 模型并使用正确的预测长度对其进行评估
"""

import torch
import numpy as np
import os
import sys
import pickle

# 添加当前目录到 Python 路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def main():
    # 配置参数
    dataset = "Electricity"
    model_path = "training/Electricity__forecast_multivar/model.pkl"
    loader = "forecast_csv"
    
    print(f"加载 CoInception 模型: {model_path}")
    print(f"数据集: {dataset}")
    
    # 加载数据
    print('加载数据... ', end='')
    from datautils import load_forecast_csv
    data, train_slice, valid_slice, test_slice, scaler, pred_lens, n_covariate_cols = load_forecast_csv(dataset)
    # 使用正确的预测长度，移除720以适应算力限制
    correct_pred_lens = [24, 48, 168, 336]
    print(f'完成，使用正确的预测长度: {correct_pred_lens}')
    
    # 加载模型
    print('加载模型... ', end='')
    from modules.coinception import CoInception
    # 先获取数据的形状，用于创建 CoInception 实例
    input_len = data.shape[1]
    input_dims = data.shape[-1]
    # 创建 CoInception 实例，尝试使用GPU
    model = CoInception(
        input_len=input_len,
        input_dims=input_dims,
        output_dims=320,  # 默认值
        hidden_dims=64,   # 默认值
        depth=3,          # 默认值
        device='cuda'     # 尝试使用GPU
    )
    # 加载模型状态
    model.load(model_path)
    print('完成')
    
    # 评估模型
    print('评估模型... ')
    from tasks.forecasting import eval_forecasting
    # 使用极小的batch_size减少GPU内存占用
    batch_size = 4
    
    # 导入必要的库用于内存管理
    import torch
    import gc
    
    # 优化内存使用
    torch.set_num_threads(1)  # 限制CPU线程数
    torch.cuda.set_per_process_memory_fraction(0.7)  # 进一步限制GPU内存使用比例
    
    # 每个预测长度单独处理，最大限度减少内存占用
    all_out_log = {}
    all_eval_res_ours = {}
    
    for pred_len in correct_pred_lens:
        print(f'处理预测长度: {pred_len}')
        
        # 释放不需要的内存
        gc.collect()
        torch.cuda.empty_cache()
        
        # 加载最新的数据副本，确保每次评估都有干净的数据
        from datautils import load_forecast_csv
        data, train_slice, valid_slice, test_slice, scaler, _, _ = load_forecast_csv(dataset)
        
        # 运行评估，每次只处理一个预测长度
        out_log_batch, eval_res_batch = eval_forecasting(
            model, data, train_slice, valid_slice, test_slice, 
            scaler, [pred_len], n_covariate_cols, batch_size=batch_size
        )
        
        # 合并结果
        all_out_log.update(out_log_batch)
        all_eval_res_ours.update(eval_res_batch['ours'])
        
        # 释放不需要的内存
        del out_log_batch, eval_res_batch, data
        gc.collect()
        torch.cuda.empty_cache()
    
    # 构建最终的评估结果
    out_log = all_out_log
    eval_res = {'ours': all_eval_res_ours}
    print('完成')
    
    # 保存评估结果
    eval_res_path = "training/Electricity__forecast_multivar/eval_res_correct.pkl"
    out_path = "training/Electricity__forecast_multivar/out_correct.pkl"
    
    from utils import pkl_save
    pkl_save(eval_res_path, eval_res)
    pkl_save(out_path, out_log)
    
    print(f"正确的评估结果已保存到: {eval_res_path}")
    print(f"预测结果已保存到: {out_path}")
    
    # 打印评估结果摘要
    print("\n评估结果摘要:")
    for pred_len, result in eval_res['ours'].items():
        print(f"预测长度 {pred_len}:")
        print(f"  归一化 MSE: {result['norm']['MSE']:.6f}")
        print(f"  原始 MSE: {result['raw']['MSE']:.6f}")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
