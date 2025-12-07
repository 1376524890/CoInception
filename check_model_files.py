#!/usr/bin/env python3
"""
检查训练目录中的模型文件是否完整
"""

import os
import glob

def check_model_files(directory):
    """检查指定目录中的模型文件是否完整
    
    Args:
        directory (str): 训练目录路径
    
    Returns:
        tuple: (int, int) 总模型数和缺少model.pkl的模型数
    """
    # 查找所有训练子目录
    train_dirs = glob.glob(os.path.join(directory, '*__*'))
    print(f"在 {directory} 中找到 {len(train_dirs)} 个训练目录")
    
    missing_count = 0
    
    # 检查每个训练目录中的model.pkl文件
    for train_dir in train_dirs:
        model_path = os.path.join(train_dir, 'model.pkl')
        if not os.path.exists(model_path):
            print(f"警告：{train_dir} 中缺少 model.pkl 文件")
            missing_count += 1
    
    print(f"在 {directory} 中，{missing_count} 个模型缺少 model.pkl 文件")
    return len(train_dirs), missing_count

if __name__ == "__main__":
    # 检查根目录中的training目录
    print("=" * 50)
    print("检查根目录中的training目录")
    print("=" * 50)
    root_total, root_missing = check_model_files('training')
    
    # 检查ts2vec中的training目录
    print("\n" + "=" * 50)
    print("检查ts2vec中的training目录")
    print("=" * 50)
    ts2vec_total, ts2vec_missing = check_model_files('ts2vec/training')
    
    # 打印总结
    print("\n" + "=" * 50)
    print("总结")
    print("=" * 50)
    print(f"根目录training目录: 总模型数 {root_total}, 缺少model.pkl {root_missing}")
    print(f"ts2vec/training目录: 总模型数 {ts2vec_total}, 缺少model.pkl {ts2vec_missing}")
    print(f"总计: 总模型数 {root_total + ts2vec_total}, 缺少model.pkl {root_missing + ts2vec_missing}")
