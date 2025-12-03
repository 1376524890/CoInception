#!/usr/bin/env python3
"""
自动化运行CoInception模型的训练、分析和可视化流程

Usage:
    # 单数据集分析
    python run_complete_analysis.py <dataset_name> <run_name> --loader <loader> [options]
    
    # 全部数据集遍历分析
    python run_complete_analysis.py --all-datasets

Example:
    # 单数据集分析
    python run_complete_analysis.py Chinatown UCR --loader UCR --batch-size 8 --repr-dims 320 --gpu 0
    
    # 全部数据集遍历分析
    python run_complete_analysis.py --all-datasets
"""

import os
import sys
import argparse
import subprocess
import shutil
import time
import pickle
import numpy as np
from tqdm import tqdm

def check_dataset_files(dataset_name, loader):
    """检查数据集文件是否存在"""
    if loader == 'UCR':
        data_dir = os.path.join(os.getcwd(), 'data', 'UCR')
        train_file = os.path.join(data_dir, f'{dataset_name}', f'{dataset_name}_TRAIN.ts')
        test_file = os.path.join(data_dir, f'{dataset_name}', f'{dataset_name}_TEST.ts')
        return os.path.exists(train_file) and os.path.exists(test_file)
    elif loader == 'UEA':
        data_dir = os.path.join(os.getcwd(), 'data', 'UEA')
        train_file = os.path.join(data_dir, f'{dataset_name}', f'{dataset_name}_TRAIN.arff')
        test_file = os.path.join(data_dir, f'{dataset_name}', f'{dataset_name}_TEST.arff')
        return os.path.exists(train_file) and os.path.exists(test_file)
    else:
        # 其他类型的数据集检查
        return True  # 暂时跳过其他类型的数据集检查

def download_datasets_guide():
    """显示数据集下载指南"""
    guide = """
    === 数据集文件缺失错误 ===
    
    当前系统缺少必要的数据集文件。请按照以下步骤解决：
    
    1. UEA数据集下载:
       访问: http://www.timeseriesclassification.com/dataset.php
       下载: UEA & UCR Time Series Classification Repository
       解压后重命名为'UEA'文件夹
       移动到: /home/codeserver/CoInception/data/UEA/
    
    2. UCR数据集下载:
       访问: https://www.cs.ucr.edu/~eamonn/time_series_data_2018
       下载UCR时间序列数据集
       解压后重命名为'UCR'文件夹  
       移动到: /home/codeserver/CoInception/data/UCR/
    
    3. 其他数据集请参考项目README.md文件
    
    详细说明请查看: /home/codeserver/CoInception/data/download_uea_data.sh
    
    运行以下命令查看帮助:
    cat /home/codeserver/CoInception/data/download_uea_data.sh
    """
    print(guide)

class CompleteAnalysisRunner:
    def __init__(self, args):
        self.args = args
        self.base_dir = os.getcwd()
        self.training_dir = os.path.join(self.base_dir, 'training')
        self.results_dir = os.path.join(self.base_dir, 'results')
        
        # 确保结果目录存在
        os.makedirs(self.results_dir, exist_ok=True)
        
        # 预设参数配置
        self.preset_params = {
            'UCR': {
                'batch_size': 8,
                'repr_dims': 320,
                'max_threads': 8,
                'seed': 42
            },
            'UEA': {
                'batch_size': 8,
                'repr_dims': 320,
                'max_threads': 8,
                'seed': 42
            },
            'forecast_csv': {
                'batch_size': 8,
                'repr_dims': 320,
                'max_threads': 8,
                'seed': 42
            },
            'forecast_csv_univar': {
                'batch_size': 8,
                'repr_dims': 320,
                'max_threads': 8,
                'seed': 42
            },
            'anomaly': {
                'batch_size': 8,
                'repr_dims': 320,
                'max_threads': 8,
                'seed': 42
            },
            'anomaly_coldstart': {
                'batch_size': 8,
                'repr_dims': 320,
                'max_threads': 8,
                'seed': 42
            }
        }
        
        # 数据集配置
        self.dataset_configs = {
            'UCR': {
                'loader': 'UCR',
                'datasets': self._get_ucr_datasets()
            },
            'UEA': {
                'loader': 'UEA',
                'datasets': self._get_uea_datasets()
            },
            'ETT': {
                'loader': 'forecast_csv',
                'datasets': ['ETTh1', 'ETTh2', 'ETTm1']
            },
            'Electricity': {
                'loader': 'forecast_csv',
                'datasets': ['electricity']
            },
            'Yahoo': {
                'loader': 'anomaly',
                'datasets': ['yahoo']
            },
            'KPI': {
                'loader': 'anomaly',
                'datasets': ['kpi']
            }
        }
    
    def run_training(self):
        """运行训练脚本"""
        print("=" * 60)
        print("开始训练模型...")
        print("=" * 60)
        
        # 获取脚本所在目录，而不是当前工作目录
        script_dir = os.path.dirname(os.path.abspath(__file__))
        
        # 构建训练命令
        train_cmd = [
            sys.executable, 'train.py',
            self.args.dataset_name,
            self.args.run_name,
            '--loader', self.args.loader,
            '--batch-size', str(self.args.batch_size),
            '--repr-dims', str(self.args.repr_dims),
            '--gpu', str(self.args.gpu),
            '--eval'
        ]
        
        # 添加可选参数
        if self.args.max_threads:
            train_cmd.extend(['--max-threads', str(self.args.max_threads)])
        if self.args.seed:
            train_cmd.extend(['--seed', str(self.args.seed)])
        if self.args.save_ckpt:
            train_cmd.append('--save_ckpt')
        if self.args.irregular > 0:
            train_cmd.extend(['--irregular', str(self.args.irregular)])
        
        print(f"执行命令: {' '.join(train_cmd)}")
        
        # 运行训练命令，使用脚本所在目录作为工作目录
        result = subprocess.run(train_cmd, cwd=script_dir, capture_output=True, text=True)
        
        if result.returncode != 0:
            print(f"训练失败! 错误信息:")
            print(result.stderr)
            sys.exit(1)
        
        print(result.stdout)
        print("=" * 60)
        print("训练完成!")
        print("=" * 60)
        
        # 提取中间数据路径
        self.intermediate_data_path = self._find_intermediate_data_path()
        print(f"中间数据保存路径: {self.intermediate_data_path}")
    
    def _find_intermediate_data_path(self):
        """查找训练生成的中间数据路径"""
        # 训练目录格式: training/<dataset_name>__<run_name>/
        training_run_dir = f"{self.args.dataset_name}__{self.args.run_name}"
        
        # 获取脚本所在目录，而不是当前工作目录
        script_dir = os.path.dirname(os.path.abspath(__file__))
        intermediate_data_path = os.path.join(script_dir, 'training', training_run_dir, 'intermediate_data.pkl')
        
        if not os.path.exists(intermediate_data_path):
            print(f"找不到中间数据文件: {intermediate_data_path}")
            sys.exit(1)
        
        return intermediate_data_path
    
    @staticmethod
    def check_dataset_files(dataset_name, loader):
        """检查数据集文件是否存在"""
        script_dir = os.path.dirname(os.path.abspath(__file__))
        
        # UEA数据集文件路径
        if loader == 'UEA':
            uea_dir = os.path.join(script_dir, 'data', 'UEA', dataset_name)
            train_file = os.path.join(uea_dir, f"{dataset_name}_TRAIN.ts")
            test_file = os.path.join(uea_dir, f"{dataset_name}_TEST.ts")
            
            if not os.path.exists(train_file) or not os.path.exists(test_file):
                print(f"❌ UEA数据集文件缺失:")
                print(f"   缺少: {train_file}")
                print(f"   缺少: {test_file}")
                print(f"   目录内容: {os.listdir(uea_dir) if os.path.exists(uea_dir) else '目录不存在'}")
                return False
            print(f"✅ UEA数据集文件检查通过")
            return True
            
        # UCR数据集文件路径
        elif loader == 'UCR':
            ucr_dir = os.path.join(script_dir, 'data', 'UCR', dataset_name)
            train_file = os.path.join(ucr_dir, f"{dataset_name}_TRAIN.ts")
            test_file = os.path.join(ucr_dir, f"{dataset_name}_TEST.ts")
            
            if not os.path.exists(train_file) or not os.path.exists(test_file):
                print(f"❌ UCR数据集文件缺失:")
                print(f"   缺少: {train_file}")
                print(f"   缺少: {test_file}")
                return False
            print(f"✅ UCR数据集文件检查通过")
            return True
            
        # ETT预测数据集文件路径
        elif loader == 'forecast_csv':
            if dataset_name.startswith('ETT'):
                ett_file = os.path.join(script_dir, 'data', 'ETT', f"{dataset_name}.csv")
                if not os.path.exists(ett_file):
                    print(f"❌ ETT数据集文件缺失:")
                    print(f"   缺少: {ett_file}")
                    print(f"   ETT目录内容: {os.listdir(os.path.join(script_dir, 'data', 'ETT')) if os.path.exists(os.path.join(script_dir, 'data', 'ETT')) else 'ETT目录不存在'}")
                    return False
                print(f"✅ ETT数据集文件检查通过")
                return True
            else:
                # 其他CSV文件检查
                csv_file = os.path.join(script_dir, 'data', f"{dataset_name}.csv")
                if not os.path.exists(csv_file):
                    print(f"❌ CSV数据集文件缺失:")
                    print(f"   缺少: {csv_file}")
                    return False
                print(f"✅ CSV数据集文件检查通过")
                return True
                
        # 异常检测数据集文件路径
        elif loader == 'anomaly':
            pkl_file = os.path.join(script_dir, 'data', f"{dataset_name}.pkl")
            if not os.path.exists(pkl_file):
                print(f"❌ 异常检测数据集文件缺失:")
                print(f"   缺少: {pkl_file}")
                return False
            print(f"✅ 异常检测数据集文件检查通过")
            return True
            
        else:
            print(f"⚠️ 未知的数据集类型: {loader}")
            return True  # 暂时跳过检查
    
    @staticmethod
    def download_datasets_guide():
        """提供数据集下载指南"""
        print("\n" + "=" * 80)
        print("数据集下载和设置指南")
        print("=" * 80)
        
        print("\n📁 UEA 数据集:")
        print("-" * 30)
        print("1. 访问官网: http://www.timeseriesclassification.com/")
        print("2. 进入 'Datasets' 页面")
        print("3. 下载所需的多变量时间序列数据集")
        print("4. 解压文件并将其放在 data/UEA/ 目录下")
        print("5. 确保文件命名为: {数据集名}_TRAIN.arff 和 {数据集名}_TEST.arff")
        print("\n示例:")
        print("   - data/UEA/BasicMotions_TRAIN.arff")
        print("   - data/UEA/BasicMotions_TEST.arff")
        
        print("\n📁 UCR 数据集:")
        print("-" * 30)
        print("1. 访问官网: http://www.timeseriesclassification.com/")
        print("2. 进入 'UCR Archive' 页面")
        print("3. 下载所需的单变量时间序列数据集")
        print("4. 解压文件并将其放在 data/UCR/ 目录下")
        print("5. 确保文件命名为: {数据集名}_TRAIN.ts 和 {数据集名}_TEST.ts")
        print("\n示例:")
        print("   - data/UCR/Chinatown_TRAIN.ts")
        print("   - data/UCR/Chinatown_TEST.ts")
        
        print("\n📁 数据下载脚本:")
        print("-" * 30)
        print("运行以下命令执行自动下载脚本:")
        print("   bash data/download_uea_data.sh")
        
        print("\n" + "=" * 80)
        print("设置完成后，重新运行训练命令")
        print("=" * 80)
    
    def run_analysis(self):
        """运行分析脚本"""
        print("\n" + "=" * 60)
        print("开始分析中间数据...")
        print("=" * 60)
        
        # 获取脚本所在目录，而不是当前工作目录
        script_dir = os.path.dirname(os.path.abspath(__file__))
        
        # 构建分析报告目录
        analysis_report_dir = os.path.join(self.results_dir, f"{self.args.dataset_name}__{self.args.run_name}_analysis")
        
        # 构建分析命令
        analysis_cmd = [
            sys.executable, 'analyze_robustness.py',
            self.intermediate_data_path,
            '--report_dir', analysis_report_dir
        ]
        
        print(f"执行命令: {' '.join(analysis_cmd)}")
        
        # 运行分析命令，使用脚本所在目录作为工作目录
        result = subprocess.run(analysis_cmd, cwd=script_dir, capture_output=True, text=True)
        
        if result.returncode != 0:
            print(f"分析失败! 错误信息:")
            print(result.stderr)
            sys.exit(1)
        
        print(result.stdout)
        print("=" * 60)
        print("分析完成!")
        print("=" * 60)
        
        self.analysis_report_dir = analysis_report_dir
    
    def run_visualization(self):
        """运行可视化脚本"""
        print("\n" + "=" * 60)
        print("开始生成可视化报告...")
        print("=" * 60)
        
        # 获取脚本所在目录，而不是当前工作目录
        script_dir = os.path.dirname(os.path.abspath(__file__))
        
        # 构建可视化报告目录
        visualization_report_dir = os.path.join(self.results_dir, f"{self.args.dataset_name}__{self.args.run_name}_visualization")
        
        # 构建可视化命令
        visualization_cmd = [
            sys.executable, 'visualize_robustness.py',
            self.intermediate_data_path,
            '--report_dir', visualization_report_dir
        ]
        
        print(f"执行命令: {' '.join(visualization_cmd)}")
        
        # 运行可视化命令，使用脚本所在目录作为工作目录
        result = subprocess.run(visualization_cmd, cwd=script_dir, capture_output=True, text=True)
        
        if result.returncode != 0:
            print(f"可视化失败! 错误信息:")
            print(result.stderr)
            sys.exit(1)
        
        print(result.stdout)
        print("=" * 60)
        print("可视化完成!")
        print("=" * 60)
        
        self.visualization_report_dir = visualization_report_dir
    
    def generate_summary_report(self):
        """生成总结报告"""
        print("\n" + "=" * 60)
        print("生成总结报告...")
        print("=" * 60)
        
        summary_path = os.path.join(self.results_dir, f"{self.args.dataset_name}__{self.args.run_name}_summary.txt")
        
        with open(summary_path, 'w') as f:
            f.write("CoInception 完整分析总结\n")
            f.write("=" * 50 + "\n\n")
            
            f.write("1. 训练配置\n")
            f.write("-" * 30 + "\n")
            f.write(f"数据集名称: {self.args.dataset_name}\n")
            f.write(f"运行名称: {self.args.run_name}\n")
            f.write(f"加载器类型: {self.args.loader}\n")
            f.write(f"批大小: {self.args.batch_size}\n")
            f.write(f"表示维度: {self.args.repr_dims}\n")
            f.write(f"GPU编号: {self.args.gpu}\n")
            if self.args.max_threads:
                f.write(f"最大线程数: {self.args.max_threads}\n")
            if self.args.seed:
                f.write(f"随机种子: {self.args.seed}\n")
            f.write(f"是否保存模型: {self.args.save_ckpt}\n")
            if self.args.irregular > 0:
                f.write(f"缺失数据比例: {self.args.irregular}\n")
            f.write("\n")
            
            f.write("2. 结果文件\n")
            f.write("-" * 30 + "\n")
            f.write(f"中间数据路径: {self.intermediate_data_path}\n")
            f.write(f"分析报告目录: {self.analysis_report_dir}\n")
            f.write(f"可视化报告目录: {self.visualization_report_dir}\n")
            f.write("\n")
            
            f.write("3. 分析结果\n")
            f.write("-" * 30 + "\n")
            f.write("详细分析结果请查看分析报告目录中的文件\n")
            f.write("\n")
            
            f.write("4. 可视化结果\n")
            f.write("-" * 30 + "\n")
            f.write("详细可视化结果请查看可视化报告目录中的HTML文件\n")
        
        print(f"总结报告生成路径: {summary_path}")
        print("=" * 60)
        print("所有分析流程完成!")
        print("=" * 60)
    
    def _get_ucr_datasets(self):
        """获取UCR数据集列表"""
        # 默认UCR数据集列表
        return ['Chinatown', 'ItalyPowerDemand', 'TwoLeadECG', 'ECGFiveDays', 'GunPoint']
    
    def _get_ucr_datasets_from_script(self):
        """从ucr.sh脚本中提取UCR数据集列表（备用方法）"""
        ucr_script_path = os.path.join(self.base_dir, 'scripts', 'ucr.sh')
        if os.path.exists(ucr_script_path):
            with open(ucr_script_path, 'r') as f:
                content = f.read()
            # 提取数据集名称
            datasets = []
            # 使用正则表达式匹配数据集名称
            import re
            matches = re.findall(r'python -u train.py\s+([\w-]+)\s+UCR', content)
            return matches[:10]  # 返回前10个数据集作为示例
        else:
            return []
    
    def _get_uea_datasets(self):
        """获取UEA数据集列表"""
        # 默认UEA数据集列表
        return ['BasicMotions', 'FaceDetection', 'Heartbeat', 'UWaveGestureLibraryAll', 'Libras']
    
    def run_single_dataset(self, dataset_name, loader, run_name=None, **kwargs):
        """运行单个数据集的分析流程"""
        # 使用数据集名称作为默认运行名称
        if run_name is None:
            run_name = loader
        
        # 检查数据集文件是否存在
        if not check_dataset_files(dataset_name, loader):
            print(f"❌ 数据集文件缺失: {dataset_name}")
            print(f"请检查 {loader} 数据集是否已正确下载到 data/{loader} 目录")
            download_datasets_guide()
            raise FileNotFoundError(f"数据集文件不存在: {dataset_name} ({loader})")
        
        # 获取预设参数
        preset_params = self.preset_params.get(loader, {})
        
        # 合并参数：kwargs > 命令行参数 > 预设参数
        params = {
            **preset_params,
            **vars(self.args),
            **kwargs
        }
        
        # 更新args对象
        for key, value in params.items():
            if hasattr(self.args, key):
                setattr(self.args, key, value)
        
        self.args.dataset_name = dataset_name
        self.args.run_name = run_name
        self.args.loader = loader
        
        print(f"\n{'-'*80}")
        print(f"开始分析数据集: {dataset_name}")
        print(f"加载器类型: {loader}")
        print(f"运行名称: {run_name}")
        print(f"{'-'*80}")
        
        # 运行训练、分析和可视化
        self.run_training()
        self.run_analysis()
        self.run_visualization()
        
        # 保存当前结果路径，用于生成最终总结
        result_info = {
            'dataset_name': dataset_name,
            'loader': loader,
            'run_name': run_name,
            'intermediate_data_path': self.intermediate_data_path,
            'analysis_report_dir': self.analysis_report_dir,
            'visualization_report_dir': self.visualization_report_dir
        }
        
        return result_info
    
    def run_all_datasets(self):
        """运行所有数据集的分析流程"""
        print("=" * 80)
        print("开始遍历所有数据集")
        print("=" * 80)
        
        all_results = []
        total_datasets = sum(len(config['datasets']) for config in self.dataset_configs.values())
        current_dataset = 0
        
        start_time = time.time()
        
        for category, config in self.dataset_configs.items():
            loader = config['loader']
            datasets = config['datasets']
            
            print(f"\n{'-'*80}")
            print(f"处理数据集类别: {category}")
            print(f"加载器类型: {loader}")
            print(f"数据集数量: {len(datasets)}")
            print(f"{'-'*80}")
            
            for dataset in tqdm(datasets, desc=f"{category} 数据集", leave=True):
                current_dataset += 1
                
                try:
                    result_info = self.run_single_dataset(dataset, loader)
                    all_results.append(result_info)
                    
                    # 保存当前进度
                    progress_path = os.path.join(self.results_dir, 'analysis_progress.pkl')
                    with open(progress_path, 'wb') as f:
                        pickle.dump(all_results, f)
                    
                except KeyboardInterrupt:
                    print("\n分析流程被用户中断!")
                    # 保存当前进度
                    progress_path = os.path.join(self.results_dir, 'analysis_progress.pkl')
                    with open(progress_path, 'wb') as f:
                        pickle.dump(all_results, f)
                    # 生成最终总结报告
                    end_time = time.time()
                    total_time = end_time - start_time
                    self.generate_final_summary(all_results, total_time)
                    print(f"\n已处理数据集数量: {len(all_results)}/{total_datasets}")
                    print(f"总耗时: {time.strftime('%H:%M:%S', time.gmtime(total_time))}")
                    sys.exit(0)
                    
                except Exception as e:
                    print(f"❌ 处理数据集 {dataset} 时发生错误: {str(e)}")
                    import traceback
                    traceback.print_exc()
                    print(f"继续处理下一个数据集...")
                    continue
        
        end_time = time.time()
        total_time = end_time - start_time
        
        # 生成最终总结报告
        self.generate_final_summary(all_results, total_time)
        
        print("\n" + "=" * 80)
        print("所有数据集分析完成!")
        print(f"总耗时: {time.strftime('%H:%M:%S', time.gmtime(total_time))}")
        print(f"成功分析数据集数量: {len(all_results)}/{total_datasets}")
        print("=" * 80)
    
    def generate_final_summary(self, all_results, total_time):
        """生成所有数据集的最终总结报告"""
        summary_path = os.path.join(self.results_dir, 'final_summary.txt')
        
        with open(summary_path, 'w') as f:
            f.write("CoInception 所有数据集分析总结\n")
            f.write("=" * 60 + "\n\n")
            
            f.write("1. 分析概览\n")
            f.write("-" * 30 + "\n")
            f.write(f"总数据集数量: {len(all_results)}\n")
            f.write(f"总耗时: {time.strftime('%H:%M:%S', time.gmtime(total_time))}\n")
            f.write(f"开始时间: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time() - total_time))}\n")
            f.write(f"结束时间: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())}\n")
            f.write("\n")
            
            f.write("2. 数据集分析结果\n")
            f.write("-" * 30 + "\n")
            
            # 按数据集类别分组
            category_results = {}
            for result in all_results:
                category = result['loader']
                if category not in category_results:
                    category_results[category] = []
                category_results[category].append(result)
            
            for category, results in category_results.items():
                f.write(f"\n{category} 数据集 ({len(results)}):\n")
                f.write("-" * 20 + "\n")
                
                for result in results:
                    f.write(f"  - 数据集: {result['dataset_name']}\n")
                    f.write(f"     分析报告: {result['analysis_report_dir']}\n")
                    f.write(f"     可视化报告: {result['visualization_report_dir']}\n")
            
            f.write("\n3. 结果文件\n")
            f.write("-" * 30 + "\n")
            f.write(f"分析进度文件: {os.path.join(self.results_dir, 'analysis_progress.pkl')}\n")
            f.write(f"最终总结文件: {summary_path}\n")
            f.write("\n")
            
            f.write("4. 使用说明\n")
            f.write("-" * 30 + "\n")
            f.write("详细分析结果请查看各数据集对应的分析报告和可视化报告目录\n")
            f.write("可视化报告包含HTML文件，可直接在浏览器中打开查看\n")
        
        print(f"\n最终总结报告生成路径: {summary_path}")
    
    def run(self):
        """运行完整的分析流程"""
        try:
            if hasattr(self.args, 'all_datasets') and self.args.all_datasets:
                # 运行所有数据集
                self.run_all_datasets()
            else:
                # 运行单个数据集
                self.run_training()
                self.run_analysis()
                self.run_visualization()
                self.generate_summary_report()
        except KeyboardInterrupt:
            print("\n分析流程被用户中断!")
            sys.exit(1)
        except Exception as e:
            print(f"分析流程发生异常: {str(e)}")
            import traceback
            traceback.print_exc()
            sys.exit(1)

def main():
    parser = argparse.ArgumentParser(description='自动化运行CoInception模型的训练、分析和可视化流程')
    
    # 添加全部数据集分析参数
    parser.add_argument('--all-datasets', action='store_true', help='遍历所有数据集进行分析')
    
    # 单数据集分析参数（当--all-datasets未指定时必需）
    parser.add_argument('dataset_name', type=str, nargs='?', help='数据集名称')
    parser.add_argument('run_name', type=str, nargs='?', help='运行名称')
    parser.add_argument('--loader', type=str, help='数据加载器类型')
    
    # 可选参数
    parser.add_argument('--batch-size', type=int, default=8, help='批大小 (默认: 8)')
    parser.add_argument('--repr-dims', type=int, default=320, help='表示维度 (默认: 320)')
    parser.add_argument('--gpu', type=int, default=0, help='GPU编号 (默认: 0)')
    parser.add_argument('--max-threads', type=int, default=None, help='最大线程数')
    parser.add_argument('--seed', type=int, default=None, help='随机种子')
    parser.add_argument('--save_ckpt', action='store_true', help='是否保存模型检查点')
    parser.add_argument('--irregular', type=float, default=0, help='缺失数据比例 (默认: 0)')
    
    args = parser.parse_args()
    
    # 参数验证
    if not args.all_datasets:
        if not args.dataset_name or not args.run_name or not args.loader:
            parser.error('当未指定 --all-datasets 时，必须提供 dataset_name, run_name 和 --loader 参数')
    
    runner = CompleteAnalysisRunner(args)
    runner.run()

if __name__ == '__main__':
    main()