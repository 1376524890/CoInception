import pickle
import numpy as np
import matplotlib.pyplot as plt
import os
import argparse
from glob import glob

class RobustnessAnalyzer:
    def __init__(self, data_path):
        self.data_path = data_path
        self.intermediate_data = self.load_data()
        
    def load_data(self):
        """加载中间数据"""
        with open(self.data_path, 'rb') as f:
            return pickle.load(f)
    
    def plot_loss_history(self, save_path=None):
        """绘制损失历史曲线"""
        loss_history = self.intermediate_data['loss_history']
        epochs = [item['epoch'] for item in loss_history]
        losses = [item['loss'] for item in loss_history]
        
        plt.figure(figsize=(10, 6))
        plt.plot(epochs, losses, '-o', markersize=4, label='Training Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training Loss History')
        plt.grid(True)
        plt.legend()
        
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        else:
            plt.show()
    
    def analyze_loss_components(self, save_path=None):
        """分析损失组件"""
        loss_components = self.intermediate_data['loss_components']
        
        if not loss_components:
            print("No loss components available.")
            return
        
        # 提取不同损失组件
        iters = [item['iter'] for item in loss_components]
        total_losses = [item['total_loss'] for item in loss_components]
        
        # 绘制损失组件对比图
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # 总损失
        axes[0, 0].plot(iters, total_losses, '-o', markersize=4, label='Total Loss')
        axes[0, 0].set_xlabel('Iteration')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].set_title('Total Loss')
        axes[0, 0].grid(True)
        axes[0, 0].legend()
        
        # 只绘制总损失，简化分析
        axes[0, 1].axis('off')
        axes[1, 0].axis('off')
        axes[1, 1].axis('off')
        
        plt.tight_layout()
        
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        else:
            plt.show()
    
    def analyze_representations(self, save_path=None):
        """分析编码器输出表示"""
        representations = self.intermediate_data['representations']
        
        if not representations:
            print("No representations saved.")
            return
        
        # 选择最后一次保存的表示
        last_repr = representations[-1]
        
        # 提取表示
        out1 = last_repr['out1']
        out1s = last_repr['out1s']
        out2 = last_repr['out2']
        out2s = last_repr['out2s']
        
        # 计算 L2 距离
        def l2_distance(a, b):
            a_flat = a.reshape(a.shape[0], -1)
            b_flat = b.reshape(b.shape[0], -1)
            return np.linalg.norm(a_flat - b_flat, axis=1)
        
        # 计算不同表示之间的 L2 距离
        dist_out1_out1s = l2_distance(out1, out1s)
        dist_out2_out2s = l2_distance(out2, out2s)
        dist_out1_out2 = l2_distance(out1, out2)
        dist_out1s_out2s = l2_distance(out1s, out2s)
        
        # 绘制 L2 距离分布
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        axes[0, 0].hist(dist_out1_out1s, bins=50, alpha=0.7)
        axes[0, 0].set_xlabel('L2 Distance')
        axes[0, 0].set_ylabel('Frequency')
        axes[0, 0].set_title('Distance: out1 vs out1s')
        axes[0, 0].grid(True)
        
        axes[0, 1].hist(dist_out2_out2s, bins=50, alpha=0.7)
        axes[0, 1].set_xlabel('L2 Distance')
        axes[0, 1].set_ylabel('Frequency')
        axes[0, 1].set_title('Distance: out2 vs out2s')
        axes[0, 1].grid(True)
        
        axes[1, 0].hist(dist_out1_out2, bins=50, alpha=0.7)
        axes[1, 0].set_xlabel('L2 Distance')
        axes[1, 0].set_ylabel('Frequency')
        axes[1, 0].set_title('Distance: out1 vs out2')
        axes[1, 0].grid(True)
        
        axes[1, 1].hist(dist_out1s_out2s, bins=50, alpha=0.7)
        axes[1, 1].set_xlabel('L2 Distance')
        axes[1, 1].set_ylabel('Frequency')
        axes[1, 1].set_title('Distance: out1s vs out2s')
        axes[1, 1].grid(True)
        
        plt.tight_layout()
        
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        else:
            plt.show()
    
    def analyze_alignment(self, save_path=None):
        """对齐性分析 - 论文 Figure 5 风格"""
        representations = self.intermediate_data['representations']
        
        if not representations:
            print("No representations saved.")
            return
        
        # 选择最后一次保存的表示
        last_repr = representations[-1]
        
        # 提取表示
        out1 = last_repr['out1']  # 原始数据的表示
        out1s = last_repr['out1s']  # 平滑数据的表示
        
        # 计算正样本对的 L2 距离
        def compute_alignment_distance(z1, z2):
            """计算对齐性距离 - 正样本对的 L2 距离"""
            z1_flat = z1.reshape(z1.shape[0], -1)
            z2_flat = z2.reshape(z2.shape[0], -1)
            l2_distances = np.linalg.norm(z1_flat - z2_flat, axis=1)
            return l2_distances
        
        # 计算对齐性距离
        alignment_distances = compute_alignment_distance(out1, out1s)
        
        # 绘制对齐性分析图
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # 绘制直方图
        ax.hist(alignment_distances, bins=40, alpha=0.7, 
                color='steelblue', edgecolor='white', density=True)
        
        # 计算并绘制均值线
        mean_dist = np.mean(alignment_distances)
        ax.axvline(mean_dist, color='red', linestyle='--', 
                   linewidth=2.5, label=f'mean')
        
        ax.set_xlabel('$l_2$ Distances', fontsize=12)
        ax.set_ylabel('Counts', fontsize=12)
        ax.set_title('Alignment Analysis - Positive Pair Feature Distance', fontsize=13, fontweight='bold')
        ax.text(0.75, 0.85, f'--- mean', transform=ax.transAxes, 
                fontsize=11, color='red')
        ax.set_xlim(left=0)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        else:
            plt.show()
    
    def analyze_uniformity(self, save_path=None):
        """均匀性分析 - 论文 Figure 6 风格"""
        representations = self.intermediate_data['representations']
        
        if not representations:
            print("No representations saved.")
            return
        
        # 选择最后一次保存的表示
        last_repr = representations[-1]
        
        # 提取表示
        out1 = last_repr['out1']
        
        # 检查是否有类别标签
        has_labels = 'labels' in last_repr
        labels = last_repr.get('labels', None)
        
        # 展平表示
        out1_flat = out1.reshape(out1.shape[0], -1)
        
        # 归一化表示
        norm_out1 = out1_flat / (np.linalg.norm(out1_flat, axis=1, keepdims=True) + 1e-8)
        
        # 创建可视化图表
        fig, axes = plt.subplots(2, 1, figsize=(12, 10))
        
        # 1. t-SNE 散点图
        from sklearn.manifold import TSNE
        tsne = TSNE(n_components=2, random_state=42, perplexity=30)
        tsne_result = tsne.fit_transform(out1_flat)
        
        ax1 = axes[0]
        
        if has_labels and labels is not None:
            # 使用真实类别标签着色
            unique_labels = np.unique(labels)
            colors = plt.cm.get_cmap('tab10', len(unique_labels))
            
            for i, label in enumerate(unique_labels):
                mask = labels == label
                ax1.scatter(tsne_result[mask, 0], tsne_result[mask, 1], 
                           label=f'Class {label}', alpha=0.7, s=50, color=colors(i))
            
            ax1.legend()
        else:
            # 使用单一颜色
            ax1.scatter(tsne_result[:, 0], tsne_result[:, 1], 
                       alpha=0.7, s=50, color='steelblue')
        
        ax1.set_title('t-SNE Visualization', fontsize=13, fontweight='bold')
        ax1.set_xlabel('t-SNE Component 1')
        ax1.set_ylabel('t-SNE Component 2')
        ax1.grid(True, alpha=0.3)
        
        # 2. 角度分布直方图（均匀性分析）
        ax2 = axes[1]
        
        # 计算角度（在 2D t-SNE 空间中）
        angles = np.arctan2(tsne_result[:, 1], tsne_result[:, 0])
        
        # 绘制角度分布直方图
        ax2.hist(angles, bins=50, alpha=0.7, 
                color='steelblue', edgecolor='white', density=True)
        
        ax2.set_title('Uniformity Analysis - Angle Distribution', fontsize=13, fontweight='bold')
        ax2.set_xlabel('Angle (radians)')
        ax2.set_ylabel('Density')
        ax2.set_xticks([-np.pi, -np.pi/2, 0, np.pi/2, np.pi])
        ax2.set_xticklabels(['$-\pi$', '$-\pi/2$', '0', '$\pi/2$', '$\pi$'])
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        else:
            plt.show()
    
    def generate_report(self, report_dir):
        """生成综合报告"""
        os.makedirs(report_dir, exist_ok=True)
        
        # 绘制并保存所有图表
        self.plot_loss_history(os.path.join(report_dir, 'loss_history.png'))
        self.analyze_loss_components(os.path.join(report_dir, 'loss_components.png'))
        self.analyze_representations(os.path.join(report_dir, 'representations.png'))
        self.analyze_alignment(os.path.join(report_dir, 'alignment_analysis.png'))
        self.analyze_uniformity(os.path.join(report_dir, 'uniformity_analysis.png'))
        
        # 生成文本报告
        report_path = os.path.join(report_dir, 'robustness_report.txt')
        with open(report_path, 'w') as f:
            f.write("CoInception Model Robustness Analysis Report\n")
            f.write("=" * 50 + "\n\n")
            
            # 训练配置
            f.write("1. Training Configuration\n")
            f.write("-" * 30 + "\n")
            config = self.intermediate_data['training_config']
            for key, value in config.items():
                f.write(f"{key}: {value}\n")
            f.write("\n")
            
            # 损失分析
            f.write("2. Loss Analysis\n")
            f.write("-" * 30 + "\n")
            loss_history = self.intermediate_data['loss_history']
            if loss_history:
                initial_loss = loss_history[0]['loss']
                final_loss = loss_history[-1]['loss']
                improvement = (initial_loss - final_loss) / initial_loss * 100
                f.write(f"Initial Loss: {initial_loss:.6f}\n")
                f.write(f"Final Loss: {final_loss:.6f}\n")
                f.write(f"Loss Improvement: {improvement:.2f}%\n")
            f.write("\n")
            
            # 表示分析
            f.write("3. Representation Analysis\n")
            f.write("-" * 30 + "\n")
            representations = self.intermediate_data['representations']
            if representations:
                f.write(f"Number of saved representations: {len(representations)}\n")
                f.write(f"Representation shape: {representations[-1]['out1'].shape}\n")
            f.write("\n")
            
            # 对齐性分析
            f.write("4. Alignment Analysis\n")
            f.write("-" * 30 + "\n")
            if representations:
                out1 = representations[-1]['out1']
                out1s = representations[-1]['out1s']
                # 计算对齐性指标
                def compute_alignment_score(z1, z2):
                    z1_flat = z1.reshape(z1.shape[0], -1)
                    z2_flat = z2.reshape(z2.shape[0], -1)
                    l2_distances = np.linalg.norm(z1_flat - z2_flat, axis=1)
                    return np.mean(l2_distances)
                
                alignment_score = compute_alignment_score(out1, out1s)
                f.write(f"Alignment Score (mean L2 distance): {alignment_score:.6f}\n")
            f.write("\n")
            
            # 均匀性分析
            f.write("5. Uniformity Analysis\n")
            f.write("-" * 30 + "\n")
            f.write("Uniformity analysis completed. See uniformity_analysis.png for details.\n")
            f.write("\n")
            
            # 损失组件分析
            f.write("6. Loss Components Analysis\n")
            f.write("-" * 30 + "\n")
            loss_components = self.intermediate_data['loss_components']
            if loss_components:
                f.write(f"Number of saved loss components: {len(loss_components)}\n")
        
        print(f"Report generated at: {report_dir}")

def main():
    parser = argparse.ArgumentParser(description='Analyze CoInception model robustness')
    parser.add_argument('data_path', type=str, help='Path to the intermediate data file')
    parser.add_argument('--report_dir', type=str, default='robustness_report', help='Directory to save the report')
    parser.add_argument('--plot', action='store_true', help='Show plots interactively')
    
    args = parser.parse_args()
    
    analyzer = RobustnessAnalyzer(args.data_path)
    
    # 生成报告
    analyzer.generate_report(args.report_dir)
    
    # 交互式绘图
    if args.plot:
        analyzer.plot_loss_history()
        analyzer.analyze_loss_components()
        analyzer.analyze_representations()

if __name__ == '__main__':
    main()
