#!/usr/bin/env python3
"""
并行训练脚本，充分利用两个GPU和100G内存
使用GPU 0训练CoInception electricity模型
使用GPU 1训练TS2Vec electricity模型
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

def monitor_processes(processes):
    """监控多个进程的运行状态"""
    while processes:
        time.sleep(5)
        # 检查每个进程的状态
        for name, proc in list(processes.items()):
            returncode = proc.poll()
            if returncode is not None:
                print(f"{name} 训练完成，返回码: {returncode}")
                if returncode != 0:
                    # 打印错误信息
                    stderr = proc.stderr.read()
                    print(f"{name} 训练错误:")
                    print(stderr[:1000])  # 只打印前1000字符
                del processes[name]

def main():
    # 创建训练命令列表
    commands = {
        # GPU 0: CoInception electricity模型，增加batch size和max_train_length
        "CoInception_electricity": (
            f"python -u train.py electricity forecast_electricity_4090 "
            f"--loader forecast_csv "
            f"--gpu 0 "
            f"--batch-size 16 "
            f"--lr 0.001 "
            f"--repr-dims 320 "
            f"--max-train-length 5000 "
            f"--seed 1 "
            f"--max-threads 32 "
            f"--eval "
            f"--save_ckpt "
            f"--save-path /home/codeserver/CoInception/data/"
        ),
        # GPU 1: TS2Vec electricity模型
        "TS2Vec_electricity": (
            f"cd ts2vec && python -u train.py electricity forecast_electricity_4090 "
            f"--loader forecast_csv "
            f"--gpu 1 "
            f"--batch-size 16 "
            f"--lr 0.001 "
            f"--repr-dims 320 "
            f"--max-train-length 5000 "
            f"--seed 1 "
            f"--max-threads 32 "
            f"--eval"
        )
    }
    
    # 运行所有命令
    processes = {}
    for name, cmd in commands.items():
        processes[name] = run_command(cmd, name)
        time.sleep(2)  # 间隔2秒启动，避免同时占用资源
    
    # 监控所有进程
    monitor_processes(processes)
    
    print("\n所有训练任务完成！")

if __name__ == "__main__":
    main()
