#!/usr/bin/env python3
"""
加载已保存的TS2Vec模型并进行优化的评估
优化内存使用，避免评估过程中内存耗尽
"""

import torch
import numpy as np
import argparse
import os
import sys
import time
import datetime
import sys

# 切换到ts2vec目录，以便相对导入正常工作
ts2vec_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'ts2vec')
os.chdir(ts2vec_dir)

# 导入TS2Vec和其他模块
from ts2vec import TS2Vec
from datautils import load_forecast_csv
from utils import init_dl_program, pkl_save
from tasks.forecasting import _eval_protocols

def main():
    # 解析命令行参数
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, required=True, help='Path to saved model file')
    parser.add_argument('--dataset', type=str, required=True, help='Dataset name')
    parser.add_argument('--loader', type=str, required=True, help='Data loader type')
    parser.add_argument('--gpu', type=int, default=1, help='GPU index')
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size for evaluation')
    parser.add_argument('--max_train_length', type=int, default=3000, help='Max train length')
    parser.add_argument('--max_threads', type=int, default=32, help='Max threads')
    args = parser.parse_args()
    
    print(f"加载模型: {args.model_path}")
    print(f"数据集: {args.dataset}")
    print(f"使用GPU: {args.gpu}")
    
    # 初始化深度学习环境
    device = init_dl_program(args.gpu, max_threads=args.max_threads)
    
    # 加载数据
    print('加载数据... ', end='')
    if args.loader == 'forecast_csv':
        task_type = 'forecasting'
        data, train_slice, valid_slice, test_slice, scaler, pred_lens, n_covariate_cols = datautils.load_forecast_csv(args.dataset)
        train_data = data[:, train_slice]
    else:
        print(f"不支持的加载器类型: {args.loader}")
        return 1
    print('完成')
    
    # 加载模型
    print('加载模型... ', end='')
    model = TS2Vec(input_dims=train_data.shape[-1], device=device)
    model.load(args.model_path)
    print('完成')
    
    # 设置模型为评估模式
    model.eval()
    
    # 优化评估过程的内存使用
    print('开始评估...')
    
    # 降低encode的batch_size以减少内存使用
    padding = 200
    batch_size = min(args.batch_size, 128)  # 进一步降低encode的batch_size
    
    t = time.time()
    all_repr = model.encode(
        data,
        causal=True,
        sliding_length=1,
        sliding_padding=padding,
        batch_size=batch_size  # 降低batch_size
    )
    infer_time = time.time() - t
    print(f"编码时间: {datetime.timedelta(seconds=infer_time)}")
    
    # 分割表示
    train_repr = all_repr[:, train_slice]
    valid_repr = all_repr[:, valid_slice]
    test_repr = all_repr[:, test_slice]
    
    train_data = data[:, train_slice, n_covariate_cols:]
    valid_data = data[:, valid_slice, n_covariate_cols:]
    test_data = data[:, test_slice, n_covariate_cols:]
    
    # 优化内存使用：删除不再需要的all_repr
    del all_repr
    
    ours_result = {}
    lr_train_time = {}
    lr_infer_time = {}
    out_log = {}
    
    for pred_len in pred_lens:
        print(f"评估预测长度: {pred_len}")
        
        # 生成预测样本
        def generate_pred_samples(features, data, pred_len, drop=0):
            n = data.shape[1]
            features = features[:, :-pred_len]
            labels = np.stack([ data[:, i:1+n+i-pred_len] for i in range(pred_len)], axis=2)[:, 1:]
            features = features[:, drop:]
            labels = labels[:, drop:]
            return features.reshape(-1, features.shape[-1]), \
                    labels.reshape(-1, labels.shape[2]*labels.shape[3])
        
        train_features, train_labels = generate_pred_samples(train_repr, train_data, pred_len, drop=padding)
        valid_features, valid_labels = generate_pred_samples(valid_repr, valid_data, pred_len)
        test_features, test_labels = generate_pred_samples(test_repr, test_data, pred_len)
        
        # 优化内存使用：在训练前转换为float32，减少内存占用
        train_features = train_features.astype(np.float32)
        train_labels = train_labels.astype(np.float32)
        valid_features = valid_features.astype(np.float32)
        valid_labels = valid_labels.astype(np.float32)
        test_features = test_features.astype(np.float32)
        test_labels = test_labels.astype(np.float32)
        
        # 训练线性回归模型
        t = time.time()
        lr = tasks._eval_protocols.fit_ridge(train_features, train_labels, valid_features, valid_labels)
        lr_train_time[pred_len] = time.time() - t
        print(f"线性回归训练时间: {lr_train_time[pred_len]:.2f}秒")
        
        # 进行预测
        t = time.time()
        test_pred = lr.predict(test_features)
        lr_infer_time[pred_len] = time.time() - t
        print(f"预测时间: {lr_infer_time[pred_len]:.2f}秒")
        
        # 优化内存使用：删除不再需要的训练和验证数据
        del train_features, train_labels, valid_features, valid_labels
        
        # 计算指标
        def cal_metrics(pred, target):
            return {
                'MSE': ((pred - target) ** 2).mean(),
                'MAE': np.abs(pred - target).mean()
            }
        
        # 重塑预测结果
        ori_shape = test_data.shape[0], -1, pred_len, test_data.shape[2]
        test_pred = test_pred.reshape(ori_shape)
        test_labels = test_labels.reshape(ori_shape)
        
        # 计算归一化的指标
        norm_metrics = cal_metrics(test_pred, test_labels)
        
        # 计算原始尺度的指标（如果需要）
        raw_metrics = None
        try:
            # 优化scaler.inverse_transform的内存使用
            if test_data.shape[0] > 1:
                # 对于多变量时间序列，需要将变量维度移到最后
                # 然后重塑为2维数组 (n_samples, n_features)
                test_pred_reshaped = test_pred.swapaxes(0, 1).reshape(-1, test_data.shape[0])
                test_labels_reshaped = test_labels.swapaxes(0, 1).reshape(-1, test_data.shape[0])
                
                # 应用逆变换
                test_pred_inv_reshaped = scaler.inverse_transform(test_pred_reshaped)
                test_labels_inv_reshaped = scaler.inverse_transform(test_labels_reshaped)
                
                # 恢复原始形状
                test_pred_inv = test_pred_inv_reshaped.reshape(ori_shape[1], ori_shape[0], ori_shape[2]).swapaxes(0, 1)
                test_labels_inv = test_labels_inv_reshaped.reshape(ori_shape[1], ori_shape[0], ori_shape[2]).swapaxes(0, 1)
            else:
                # 单序列情况，直接应用逆变换
                test_pred_inv = scaler.inverse_transform(test_pred)
                test_labels_inv = scaler.inverse_transform(test_labels)
            
            raw_metrics = cal_metrics(test_pred_inv, test_labels_inv)
            
            # 保存结果
            out_log[pred_len] = {
                'norm': test_pred,
                'raw': test_pred_inv,
                'norm_gt': test_labels,
                'raw_gt': test_labels_inv
            }
        except Exception as e:
            print(f"计算原始尺度指标时出错: {e}")
            # 只保存归一化的结果
            out_log[pred_len] = {
                'norm': test_pred,
                'norm_gt': test_labels
            }
        
        ours_result[pred_len] = {
            'norm': norm_metrics,
            'raw': raw_metrics
        }
        
        print(f"预测长度 {pred_len} 完成，MSE: {norm_metrics['MSE']:.6f}")
    
    # 构建评估结果
    eval_res = {
        'ours': ours_result,
        'infer_time': infer_time,
        'lr_train_time': lr_train_time,
        'lr_infer_time': lr_infer_time
    }
    
    # 保存评估结果
    model_dir = os.path.dirname(args.model_path)
    eval_res_path = os.path.join(model_dir, 'eval_res.pkl')
    out_path = os.path.join(model_dir, 'out.pkl')
    
    pkl_save(eval_res_path, eval_res)
    pkl_save(out_path, out_log)
    
    print(f"评估结果已保存到: {eval_res_path}")
    print(f"预测结果已保存到: {out_path}")
    
    # 打印评估结果
    print("\n评估结果摘要:")
    for pred_len, result in ours_result.items():
        print(f"预测长度 {pred_len}:")
        print(f"  归一化 MSE: {result['norm']['MSE']:.6f}")
        if result['raw'] is not None:
            print(f"  原始 MSE: {result['raw']['MSE']:.6f}")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
