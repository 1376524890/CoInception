#!/usr/bin/env python3
"""
噪声比率分析实验脚本

该脚本实现了CoInception和vs2rec两种方式的噪声训练和结果比较，
并生成符合要求的可视化表格和图表。
"""

import numpy as np
import os
import sys
import argparse
import time
import pickle
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
from datautils import load_forecast_csv
import torch

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# 噪声比率设置
NOISE_RATIOS = [0, 10, 20, 30, 40, 50]

# 预测长度
PREDICTION_HORIZONS = [24, 48, 96, 288, 672]

# Ridge回归的α参数搜索空间
ALPHA_SEARCH = [0.1, 0.2, 0.5, 1, 2, 5, 10, 20, 50, 100, 200, 500, 1000]


def add_gaussian_noise(x, noise_ratio):
    """
    向时间序列添加高斯噪声
    
    Args:
        x: 输入时间序列, shape (batch, seq_len, features)
        noise_ratio: 噪声比率 (0, 10, 20, 30, 40, 50)
    
    Returns:
        添加噪声后的时间序列
    """
    # 计算输入序列的均值幅度
    mean_amplitude = np.mean(np.abs(x))
    
    # 噪声的均值 = x% * 输入序列均值幅度
    noise_mean = (noise_ratio / 100.0) * mean_amplitude
    
    # 生成高斯噪声 (均值为noise_mean, 标准差可设为均值的一定比例)
    noise = np.random.normal(loc=noise_mean, scale=noise_mean, size=x.shape)
    
    # 添加噪声
    x_noisy = x + noise
    
    return x_noisy


def load_model(model_type, **kwargs):
    """
    加载模型
    
    Args:
        model_type: 模型类型, 'coinception' 或 'vs2rec'
        **kwargs: 模型初始化参数
    
    Returns:
        加载的模型
    """
    if model_type == 'coinception':
        from modules.coinception import CoInception
        model = CoInception(**kwargs)
    elif model_type == 'vs2rec':
        # 假设vs2rec模型在ts2vec目录下
        sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'ts2vec'))
        from ts2vec import TS2Vec
        # TS2Vec不需要input_len参数，所以我们需要从kwargs中移除它
        kwargs_copy = kwargs.copy()
        if 'input_len' in kwargs_copy:
            del kwargs_copy['input_len']
        model = TS2Vec(**kwargs_copy)
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    return model


def generate_pred_samples(features, data, pred_len, drop=0):
    """
    生成预测样本
    
    Args:
        features: 特征数据
        data: 原始数据
        pred_len: 预测长度
        drop: 丢弃的样本数量
    
    Returns:
        预测样本和标签
    """
    n = data.shape[1]
    features = features[:, :-pred_len]
    labels = np.stack([ data[:, i:1+n+i-pred_len] for i in range(pred_len)], axis=2)[:, 1:]
    features = features[:, drop:]
    labels = labels[:, drop:]
    return features.reshape(-1, features.shape[-1]), \
            labels.reshape(-1, labels.shape[2]*labels.shape[3])


def noise_ratio_experiment(model_type, dataset, noise_ratios=NOISE_RATIOS, device='cuda', n_epochs=100):
    """
    噪声比率分析实验
    
    Args:
        model_type: 模型类型, 'coinception' 或 'vs2rec'
        dataset: 数据集名称
        noise_ratios: 噪声比率列表
        device: 设备名称
        n_epochs: 训练轮数
    
    Returns:
        实验结果
    """
    results = {}
    
    # 使用所有噪声比率
    noise_ratios = [0, 10, 20, 30, 40, 50]
    
    for noise_ratio in noise_ratios:
        print(f"\n=== {model_type} - Noise Ratio: {noise_ratio}% ===")
        
        # 1. 加载数据
        data, train_slice, valid_slice, test_slice, scaler, pred_lens, n_covariate_cols = load_forecast_csv(
            dataset, univar=True
        )
        
        # 提取训练、验证和测试数据
        train_data = data[:, train_slice, :]
        valid_data = data[:, valid_slice, :]
        test_data = data[:, test_slice, :]
        
        # 2. 在预训练数据上添加噪声
        if noise_ratio > 0:
            train_data_noisy = add_gaussian_noise(train_data, noise_ratio)
        else:
            train_data_noisy = train_data
        
        # 3. 模型初始化和训练
        input_len = train_data_noisy.shape[1]  # 输入序列长度
        input_dims = train_data_noisy.shape[2]  # 输入特征维度
        
        # 设置更小的batch_size和max_train_length以减少内存使用
        batch_size = 8
        max_train_length = 1000
        
        model = load_model(
            model_type,
            input_len=input_len,
            input_dims=input_dims,
            device=device,
            batch_size=batch_size,
            lr=0.001,
            max_train_length=max_train_length
        )
        
        # 预训练模型
        t_start = time.time()
        model.fit(train_data_noisy, n_epochs=n_epochs, verbose=True)
        t_end = time.time()
        print(f"预训练时间: {t_end - t_start:.2f}秒")
        
        # 4. 提取表示 - 向完整数据添加噪声
        print("  向完整数据添加噪声...")
        if noise_ratio > 0:
            data_noisy = add_gaussian_noise(data, noise_ratio)
        else:
            data_noisy = data
        
        # 根据模型类型使用不同的参数名称
        if model_type == 'coinception':
            all_repr = model.encode(
                data_noisy,
                casual=True,
                sliding_length=1,
                sliding_padding=100,
                batch_size=16
            )
        else:
            all_repr = model.encode(
                data_noisy,
                causal=True,
                sliding_length=1,
                sliding_padding=100,
                batch_size=16
            )
        
        train_repr = all_repr[:, train_slice]
        valid_repr = all_repr[:, valid_slice]
        test_repr = all_repr[:, test_slice]
        
        train_data = data_noisy[:, train_slice, n_covariate_cols:]
        valid_data = data_noisy[:, valid_slice, n_covariate_cols:]
        test_data = data_noisy[:, test_slice, n_covariate_cols:]
        
        # 5. 训练下游预测模型
        mse_list = []
        mae_list = []
        
        # 使用generate_pred_samples函数生成预测样本
        for horizon in PREDICTION_HORIZONS:
            print(f"  预测长度: {horizon}")
            
            # 使用generate_pred_samples函数生成预测样本和标签
            train_features, train_labels = generate_pred_samples(train_repr, train_data, horizon, drop=100)
            valid_features, valid_labels = generate_pred_samples(valid_repr, valid_data, horizon)
            test_features, test_labels = generate_pred_samples(test_repr, test_data, horizon)
            
            # 网格搜索最优α，使用验证集进行调优
            best_mse = float('inf')
            best_mae = float('inf')
            best_alpha = 0
            
            for alpha in ALPHA_SEARCH:
                regressor = Ridge(alpha=alpha)
                regressor.fit(train_features, train_labels)
                
                # 在验证集上评估
                val_pred = regressor.predict(valid_features)
                val_mse = mean_squared_error(valid_labels, val_pred)
                val_mae = mean_absolute_error(valid_labels, val_pred)
                
                if val_mse < best_mse:
                    best_mse = val_mse
                    best_mae = val_mae
                    best_alpha = alpha
            
            # 使用最优α在测试集上评估
            regressor = Ridge(alpha=best_alpha)
            regressor.fit(train_features, train_labels)
            y_pred = regressor.predict(test_features)
            test_mse = mean_squared_error(test_labels, y_pred)
            test_mae = mean_absolute_error(test_labels, y_pred)
            
            print(f"    最优α: {best_alpha}, 测试MSE: {test_mse:.4f}, 测试MAE: {test_mae:.4f}")
            mse_list.append(test_mse)
            mae_list.append(test_mae)
        
        # 6. 记录平均结果
        results[noise_ratio] = {
            'MSE': np.mean(mse_list),
            'MAE': np.mean(mae_list),
            'mse_per_horizon': mse_list,
            'mae_per_horizon': mae_list
        }
        
        print(f"  平均MSE: {results[noise_ratio]['MSE']:.4f}")
        print(f"  平均MAE: {results[noise_ratio]['MAE']:.4f}")
    
    return results


def save_results(results, model_type, dataset):
    """
    保存实验结果
    
    Args:
        results: 实验结果
        model_type: 模型类型
        dataset: 数据集名称
    """
    results_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results')
    os.makedirs(results_dir, exist_ok=True)
    
    results_file = os.path.join(results_dir, f'{model_type}_noise_results_{dataset}.pkl')
    with open(results_file, 'wb') as f:
        pickle.dump(results, f)
    
    print(f"结果已保存到: {results_file}")
    return results_file


def load_results(model_type, dataset):
    """
    加载实验结果
    
    Args:
        model_type: 模型类型
        dataset: 数据集名称
    
    Returns:
        加载的实验结果
    """
    results_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results')
    results_file = os.path.join(results_dir, f'{model_type}_noise_results_{dataset}.pkl')
    
    if os.path.exists(results_file):
        with open(results_file, 'rb') as f:
            results = pickle.load(f)
        print(f"已加载结果: {results_file}")
        return results
    else:
        print(f"结果文件不存在: {results_file}")
        return None


def main():
    """
    主函数
    """
    parser = argparse.ArgumentParser(description='噪声比率分析实验')
    parser.add_argument('--dataset', type=str, default='ETTm1', help='数据集名称')
    parser.add_argument('--device', type=str, default='cuda', help='设备名称')
    parser.add_argument('--force-retrain', action='store_true', help='是否强制重新训练')
    parser.add_argument('--n-epochs', type=int, default=100, help='训练轮数')
    parser.add_argument('--model', type=str, default='coinception', choices=['coinception', 'vs2rec', 'all'], help='模型类型')
    
    args = parser.parse_args()
    
    # 确保结果目录存在
    results_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results')
    os.makedirs(results_dir, exist_ok=True)
    
    # 实验结果字典
    all_results = {}
    
    # 模型类型列表
    if args.model == 'all':
        model_types = ['coinception', 'vs2rec']
    else:
        model_types = [args.model]
    
    for model_type in model_types:
        # 尝试加载已有结果
        results = load_results(model_type, args.dataset)
        
        # 如果结果不存在或强制重新训练，则进行训练
        if results is None or args.force_retrain:
            results = noise_ratio_experiment(model_type, args.dataset, device=args.device, n_epochs=args.n_epochs)
            save_results(results, model_type, args.dataset)
        
        all_results[model_type] = results
    
    # 保存所有结果到一个文件
    all_results_file = os.path.join(results_dir, f'all_noise_results_{args.dataset}.pkl')
    
    # 如果文件已存在，加载之前的结果
    if os.path.exists(all_results_file):
        with open(all_results_file, 'rb') as f:
            existing_results = pickle.load(f)
        # 更新现有结果
        existing_results.update(all_results)
        all_results = existing_results
    
    # 保存更新后的结果
    with open(all_results_file, 'wb') as f:
        pickle.dump(all_results, f)
    
    print(f"\n所有结果已保存到: {all_results_file}")
    print("\n实验完成！")


if __name__ == "__main__":
    main()
