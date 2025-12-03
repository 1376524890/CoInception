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
        
        # 运行训练命令
        result = subprocess.run(train_cmd, cwd=self.base_dir, capture_output=True, text=True)
        
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
        intermediate_data_path = os.path.join(self.training_dir, training_run_dir, 'intermediate_data.pkl')
        
        if not os.path.exists(intermediate_data_path):
            print(f"找不到中间数据文件: {intermediate_data_path}")
            sys.exit(1)
        
        return intermediate_data_path
    
    def run_analysis(self):
        """运行分析脚本"""
        print("\n" + "=" * 60)
        print("开始分析中间数据...")
        print("=" * 60)
        
        # 构建分析报告目录
        analysis_report_dir = os.path.join(self.results_dir, f"{self.args.dataset_name}__{self.args.run_name}_analysis")
        
        # 构建分析命令
        analysis_cmd = [
            sys.executable, 'analyze_robustness.py',
            self.intermediate_data_path,
            '--report_dir', analysis_report_dir
        ]
        
        print(f"执行命令: {' '.join(analysis_cmd)}")
        
        # 运行分析命令
        result = subprocess.run(analysis_cmd, cwd=self.base_dir, capture_output=True, text=True)
        
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
        
        # 构建可视化报告目录
        visualization_report_dir = os.path.join(self.results_dir, f"{self.args.dataset_name}__{self.args.run_name}_visualization")
        
        # 构建可视化命令
        visualization_cmd = [
            sys.executable, 'visualize_robustness.py',
            self.intermediate_data_path,
            '--report_dir', visualization_report_dir
        ]
        
        print(f"执行命令: {' '.join(visualization_cmd)}")
        
        # 运行可视化命令
        result = subprocess.run(visualization_cmd, cwd=self.base_dir, capture_output=True, text=True)
        
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
            
            for dataset in datasets:
                current_dataset += 1
                print(f"\n[{current_dataset}/{total_datasets}] 处理数据集: {dataset}")
                
                try:
                    result_info = self.run_single_dataset(dataset, loader)
                    all_results.append(result_info)
                    
                    # 保存当前进度
                    progress_path = os.path.join(self.results_dir, 'analysis_progress.pkl')
                    with open(progress_path, 'wb') as f:
                        pickle.dump(all_results, f)
                    
                    print(f"✅ 数据集 {dataset} 分析完成")
                    
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
