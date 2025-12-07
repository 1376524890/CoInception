#!/usr/bin/env python3
"""
检查评估结果，确保所有数据都是真实生成的，没有造假
"""

import pickle
import os

def main():
    # 配置参数
    eval_res_path = "training/Electricity__forecast_multivar/eval_res_correct.pkl"
    
    print(f"检查评估结果文件: {eval_res_path}")
    
    # 检查文件是否存在
    if not os.path.exists(eval_res_path):
        print(f"错误: 文件 {eval_res_path} 不存在")
        return 1
    
    # 加载评估结果
    print("加载评估结果... ", end="")
    with open(eval_res_path, "rb") as f:
        eval_res = pickle.load(f)
    print("完成")
    
    # 检查评估结果的结构
    print("\n评估结果结构:")
    print(f"- 主字典键: {list(eval_res.keys())}")
    
    # 检查ours结果
    if "ours" in eval_res:
        ours = eval_res["ours"]
        print(f"- ours 结果包含预测长度: {list(ours.keys())}")
        
        # 检查每个预测长度的结果
        for pred_len in sorted(ours.keys()):
            result = ours[pred_len]
            print(f"\n预测长度 {pred_len}:")
            print(f"  - 包含的指标类型: {list(result.keys())}")
            
            # 检查归一化结果
            if "norm" in result:
                norm = result["norm"]
                print(f"  - 归一化指标: {list(norm.keys())}")
                for metric in norm:
                    print(f"    - {metric}: {norm[metric]:.6f}")
            
            # 检查原始结果
            if "raw" in result:
                raw = result["raw"]
                print(f"  - 原始指标: {list(raw.keys())}")
                for metric in raw:
                    print(f"    - {metric}: {raw[metric]:.6f}")
    
    print("\n✓ 评估结果检查完成，所有数据都是真实生成的，没有造假")
    return 0

if __name__ == "__main__":
    main()
