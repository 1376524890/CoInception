#!/usr/bin/env python3
"""
评估所有CoInception分类模型
"""

import os
import sys
import numpy as np
import pickle
import glob

# 添加当前目录到Python路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def evaluate_coin_classification():
    """
    评估所有CoInception分类模型
    """
    # 获取所有训练目录
    training_dirs = glob.glob('training/*__run_*')
    print(f"找到 {len(training_dirs)} 个训练目录")
    
    # 过滤出分类模型目录（排除预测模型）
    classification_dirs = [d for d in training_dirs if 'forecast' not in d and 'anomaly' not in d]
    print(f"找到 {len(classification_dirs)} 个分类模型目录")
    
    # 遍历所有分类模型目录
    for training_dir in classification_dirs:
        # 提取数据集名称
        dataset_name = training_dir.split('__')[0].split('/')[-1]
        print(f"\n处理数据集: {dataset_name}")
        
        try:
            # 检查model.pkl是否存在
            model_path = os.path.join(training_dir, 'model.pkl')
            if not os.path.exists(model_path):
                print(f"警告: {training_dir} 中缺少 model.pkl 文件")
                continue
            
            # 加载数据集
            print(f"加载数据集 {dataset_name}...")
            from datautils import load_UCR, load_UEA
            
            # 根据数据集类型选择正确的数据加载函数
            from analysis_preset import DATASET_LISTS
            ucr_datasets = set(DATASET_LISTS['ucr']['datasets'])
            uea_datasets = set(DATASET_LISTS['uea']['datasets'])
            
            if dataset_name in ucr_datasets:
                # UCR数据集
                train_data, train_labels, test_data, test_labels = load_UCR(dataset_name)
            elif dataset_name in uea_datasets:
                # UEA数据集
                train_data, train_labels, test_data, test_labels = load_UEA(dataset_name)
            else:
                print(f"警告: 数据集 {dataset_name} 既不是UCR也不是UEA数据集，跳过")
                continue
            
            print(f"数据集加载完成: 训练数据 {train_data.shape}, 测试数据 {test_data.shape}")
            
            # 加载模型
            print(f"加载模型 {model_path}...")
            from modules.coinception import CoInception
            
            # 创建模型实例
            model = CoInception(
                input_len=train_data.shape[1],
                input_dims=train_data.shape[-1],
                output_dims=320,
                hidden_dims=64,
                depth=3,
                device='cuda' if os.environ.get('CUDA_VISIBLE_DEVICES') else 'cpu'
            )
            
            # 加载模型权重
            model.load(model_path)
            print("模型加载完成")
            
            # 运行评估
            print("运行评估...")
            from tasks.classification import eval_classification
            y_score, eval_res = eval_classification(
                model, train_data, train_labels, test_data, test_labels, 
                eval_protocol='linear'
            )
            
            print(f"评估结果: 准确率 = {eval_res['acc']:.4f}, AUPRC = {eval_res['auprc']:.4f}")
            
            # 保存评估结果
            eval_res_path = os.path.join(training_dir, 'eval_res.pkl')
            with open(eval_res_path, 'wb') as f:
                pickle.dump(eval_res, f)
            print(f"评估结果已保存到 {eval_res_path}")
            
        except Exception as e:
            print(f"处理数据集 {dataset_name} 时出错: {e}")
            import traceback
            traceback.print_exc()
            continue

if __name__ == "__main__":
    evaluate_coin_classification()
