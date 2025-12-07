#!/usr/bin/env python3
"""
优化训练脚本，使用单GPU但充分利用100G内存
"""

import subprocess
import os
import time

def run_command(command, name):
    """运行命令并返回进程对象"""
    print(f"启动 {name} 训练...")
    print(f"命令: {command}")
    return subprocess.Popen(
        command, 
        shell=True, 
        stdout=subprocess.PIPE, 
        stderr=subprocess.PIPE, 
        text=True
    )

def main():
    # 创建优化的训练命令
    commands = {
        # 使用GPU 0训练CoInception electricity模型，优化参数充分利用内存
        "CoInception_electricity_optimized": (
            f"PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128 "
            f"python -u train.py electricity forecast_electricity_optimized "
            f"--loader forecast_csv "
            f"--gpu 0 "
            f"--batch-size 12 "
            f"--lr 0.001 "
            f"--repr-dims 320 "
            f"--max-train-length 3500 "
            f"--seed 1 "
            f"--max-threads 32 "
            f"--eval "
            f"--save_ckpt "
            f"--save-path /home/codeserver/CoInception/data/"
        ),
        # 使用GPU 1训练TS2Vec electricity模型，优化参数充分利用内存
        "TS2Vec_electricity_optimized": (
            f"PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128 "
            f"cd ts2vec && python -u train.py electricity forecast_electricity_optimized "
            f"--loader forecast_csv "
            f"--gpu 1 "
            f"--batch-size 12 "
            f"--lr 0.001 "
            f"--repr-dims 320 "
            f"--max-train-length 3500 "
            f"--seed 1 "
            f"--max-threads 32 "
            f"--eval"
        )
    }
    
    # 运行训练命令
    processes = {}
    for name, cmd in commands.items():
        processes[name] = run_command(cmd, name)
        time.sleep(5)  # 间隔5秒启动，避免同时占用资源
    
    # 监控训练过程
    while processes:
        time.sleep(15)
        # 检查进程状态
        for name, proc in list(processes.items()):
            returncode = proc.poll()
            if returncode is not None:
                print(f"{name} 训练完成，返回码: {returncode}")
                if returncode != 0:
                    # 打印错误信息
                    stderr = proc.stderr.read()
                    print(f"{name} 训练错误:")
                    print(stderr[:2000])  # 只打印前2000字符
                del processes[name]
    
    print("\n所有训练任务完成！")

if __name__ == "__main__":
    main()
