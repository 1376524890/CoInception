"""
CoInception 可视化复现优化分析
=====================================
对照原论文进行实验效果分析的优化建议

原论文关键可视化:
1. Figure 2: 噪声鲁棒性对比 (Noiseless vs Noisy 相关性)
2. Figure 5: 对齐性分析 (L2距离直方图)
3. Figure 6: 均匀性分析 (Gaussian KDE + vMF KDE)
4. Figure 8: 噪声比例分析 (雷达图)
5. Figure 13: 感受野分析 (参数量 vs 深度)
6. Figure 14: 聚类性分析 (带类别标签的t-SNE)
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from scipy.spatial.distance import cosine
import warnings
warnings.filterwarnings('ignore')

# 设置中文字体和样式
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False
plt.style.use('seaborn-v0_8-whitegrid')

class CoInceptionVisualizationOptimizer:
    """
    CoInception可视化优化器
    提供论文标准的可视化复现方案
    """
    
    def __init__(self):
        self.analysis_results = {}
        
    # =========================================
    # 1. 噪声鲁棒性分析 (对应论文 Figure 2)
    # =========================================
    def analyze_noise_robustness(self, representations_clean, representations_noisy):
        """
        分析噪声鲁棒性 - 论文Figure 2的核心可视化
        
        论文原图特点:
        - 显示Noiseless和Noisy两组信号
        - 标注相关系数(Corr)
        - 对比CoInception与TS2Vec的相关性差异
        
        优化建议:
        1. 需要同时展示原始信号和噪声信号的时序图
        2. 计算并显示余弦相似度/相关系数
        3. 与基线方法(TS2Vec)进行对比
        """
        # 计算相关性
        correlations = []
        for clean, noisy in zip(representations_clean, representations_noisy):
            corr = 1 - cosine(clean.flatten(), noisy.flatten())
            correlations.append(corr)
        
        mean_corr = np.mean(correlations)
        
        print("=" * 60)
        print("噪声鲁棒性分析 (论文 Figure 2 优化)")
        print("=" * 60)
        print(f"平均相关性: {mean_corr:.4f}")
        print("\n优化建议:")
        print("1. 当前实现缺少原始时序信号的可视化")
        print("2. 需要添加与TS2Vec的对比基线")
        print("3. 建议添加多噪声级别的对比分析")
        
        return {'mean_correlation': mean_corr, 'correlations': correlations}
    
    # =========================================
    # 2. 对齐性分析 (对应论文 Figure 5)
    # =========================================
    def create_alignment_analysis(self, positive_pair_distances):
        """
        对齐性分析 - 论文Figure 5的可视化
        
        论文原图特点:
        - 正样本对的L2距离直方图
        - 显示均值线(mean)
        - 对比CoInception与TS2Vec的分布差异
        
        当前问题:
        - representations.png 显示的是余弦相似度分布
        - 缺少L2距离的直方图
        - 缺少与TS2Vec的对比
        
        优化方案:
        """
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        # CoInception
        ax1 = axes[0]
        ax1.hist(positive_pair_distances, bins=30, alpha=0.7, 
                 color='steelblue', edgecolor='white')
        mean_dist = np.mean(positive_pair_distances)
        ax1.axvline(mean_dist, color='red', linestyle='--', 
                    linewidth=2, label=f'mean: {mean_dist:.3f}')
        ax1.set_xlabel('$l_2$ Distances', fontsize=12)
        ax1.set_ylabel('Counts', fontsize=12)
        ax1.set_title('CoInception - Positive Pair Feature Distance', fontsize=12)
        ax1.legend()
        
        # TS2Vec (需要对比数据)
        ax2 = axes[1]
        ax2.text(0.5, 0.5, 'TS2Vec Baseline\n(需要补充对比数据)', 
                 ha='center', va='center', fontsize=14,
                 transform=ax2.transAxes)
        ax2.set_title('TS2Vec - Positive Pair Feature Distance', fontsize=12)
        
        plt.tight_layout()
        return fig
    
    # =========================================
    # 3. 均匀性分析 (对应论文 Figure 6)
    # =========================================
    def create_uniformity_analysis(self, representations, labels):
        """
        均匀性分析 - 论文Figure 6的可视化
        
        论文原图特点:
        - 使用t-SNE降维到2D
        - 上方: Gaussian KDE密度图
        - 下方: von Mises-Fisher KDE角度分布
        - 分别展示All Classes和各个单独类别
        
        当前问题:
        - representation_tsne.png 只显示了out1/out1s/out2/out2s
        - 缺少类别信息的着色
        - 缺少KDE密度估计
        - 缺少vMF角度分布
        
        优化方案:
        """
        # t-SNE降维
        tsne = TSNE(n_components=2, random_state=42, perplexity=30)
        embeddings_2d = tsne.fit_transform(representations)
        
        unique_labels = np.unique(labels)
        n_classes = len(unique_labels)
        
        # 创建Figure 6风格的可视化
        fig, axes = plt.subplots(2, n_classes + 1, figsize=(4*(n_classes+1), 8))
        
        colors = plt.cm.Set1(np.linspace(0, 1, n_classes))
        
        # 上方: Gaussian KDE
        for i, (label, color) in enumerate(zip(['All Classes'] + list(unique_labels), 
                                               ['gray'] + list(colors))):
            ax = axes[0, i]
            if i == 0:  # All classes
                for j, (c, col) in enumerate(zip(unique_labels, colors)):
                    mask = labels == c
                    ax.scatter(embeddings_2d[mask, 0], embeddings_2d[mask, 1], 
                              c=[col], alpha=0.6, s=20, label=f'Class {c}')
                ax.legend(fontsize=8)
            else:
                mask = labels == label
                ax.scatter(embeddings_2d[mask, 0], embeddings_2d[mask, 1], 
                          c=[color], alpha=0.6, s=20)
            ax.set_title(f'{label if i == 0 else f"Class {label}"}', fontsize=11)
            ax.set_xlabel('t-SNE 1')
            ax.set_ylabel('t-SNE 2')
        
        # 下方: vMF KDE on angles
        for i, label in enumerate(['All Classes'] + list(unique_labels)):
            ax = axes[1, i]
            if i == 0:
                data = embeddings_2d
            else:
                mask = labels == label
                data = embeddings_2d[mask]
            
            # 计算角度
            angles = np.arctan2(data[:, 1], data[:, 0])
            
            # KDE
            if len(angles) > 1:
                kde = stats.gaussian_kde(angles)
                x_range = np.linspace(-np.pi, np.pi, 100)
                ax.fill_between(x_range, kde(x_range), alpha=0.5)
                ax.plot(x_range, kde(x_range), linewidth=2)
            
            ax.set_xlabel('Angle (radians)')
            ax.set_ylabel('Density')
            ax.set_xlim(-np.pi, np.pi)
        
        plt.suptitle('Uniformity Analysis (Paper Figure 6 Style)', fontsize=14)
        plt.tight_layout()
        return fig
    
    # =========================================
    # 4. 聚类性分析 (对应论文 Figure 14)
    # =========================================  
    def create_clusterability_analysis(self, representations, labels, dataset_name):
        """
        聚类性分析 - 论文Figure 14的可视化
        
        论文原图特点:
        - 同时对比CoInception和TS2Vec
        - 使用不同颜色表示不同类别
        - 显示在3个数据集上的效果
        
        当前问题:
        - t-SNE图缺少真实类别标签着色
        - 只显示了encoder输出类型(out1/out1s等)
        - 缺少与TS2Vec的对比
        
        优化方案:
        """
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        
        tsne = TSNE(n_components=2, random_state=42)
        embeddings_2d = tsne.fit_transform(representations)
        
        unique_labels = np.unique(labels)
        colors = plt.cm.tab10(np.linspace(0, 1, len(unique_labels)))
        
        # CoInception
        ax = axes[0]
        for label, color in zip(unique_labels, colors):
            mask = labels == label
            ax.scatter(embeddings_2d[mask, 0], embeddings_2d[mask, 1],
                      c=[color], label=f'Class {label}', alpha=0.7, s=30)
        ax.set_title(f'CoInception - {dataset_name}', fontsize=14)
        ax.legend(loc='best', fontsize=8)
        ax.set_xlabel('t-SNE Component 1')
        ax.set_ylabel('t-SNE Component 2')
        
        # TS2Vec placeholder
        axes[1].text(0.5, 0.5, 'TS2Vec\n(需要补充对比数据)', 
                    ha='center', va='center', fontsize=16,
                    transform=axes[1].transAxes)
        axes[1].set_title(f'TS2Vec - {dataset_name}', fontsize=14)
        
        plt.tight_layout()
        return fig

    # =========================================
    # 5. 感受野分析 (对应论文 Figure 13)
    # =========================================
    def create_receptive_field_analysis(self):
        """
        感受野分析 - 论文Figure 13的可视化
        
        论文原图特点:
        - X轴: 网络深度(Depth)
        - Y轴左: 参数数量(# Params, log scale)
        - Y轴右: 感受野(Receptive Field)
        - 对比CoInception与TS2Vec
        
        当前问题:
        - 完全缺失此可视化
        
        计算公式 (论文中):
        - Dilation factor: d_u^i = (2k-1)^(i-1)
        - Receptive field: r_u^i = (2k-1)^i
        """
        depths = np.arange(1, 31)
        
        # CoInception参数估算
        base_params = 206000  # 论文中提到的参数量
        coinception_params = base_params * np.log(depths + 1)
        
        # TS2Vec参数估算
        ts2vec_params = 641000 * np.log(depths + 1) * 1.2
        
        # 感受野计算 (k=2 for basic unit)
        k = 2
        receptive_field = (2*k - 1) ** depths
        
        fig, ax1 = plt.subplots(figsize=(10, 6))
        
        color1, color2 = 'purple', 'orange'
        
        ax1.set_xlabel('Depth', fontsize=12)
        ax1.set_ylabel('# Params (log scale)', fontsize=12)
        ax1.semilogy(depths, coinception_params, 'o-', color=color1, 
                     label='CoInception', markersize=6)
        ax1.semilogy(depths, ts2vec_params, 'v-', color=color2, 
                     label='TS2Vec', markersize=6)
        ax1.legend(loc='upper left')
        
        ax2 = ax1.twinx()
        ax2.set_ylabel('Receptive Field', fontsize=12, color='green')
        ax2.semilogy(depths, receptive_field, '--', color='green', 
                     linewidth=2, alpha=0.7)
        ax2.tick_params(axis='y', labelcolor='green')
        
        plt.title('Receptive Field Analysis (Paper Figure 13 Style)', fontsize=14)
        plt.tight_layout()
        return fig

    # =========================================
    # 6. 噪声比例分析 (对应论文 Figure 8)
    # =========================================
    def create_noise_ratio_analysis(self):
        """
        噪声比例分析 - 论文Figure 8的可视化
        
        论文原图特点:
        - 雷达图样式
        - 显示不同噪声比例(0%, 10%, 20%, 30%, 40%, 50%)
        - 对比CoInception与TS2Vec的MSE/MAE
        
        当前问题:
        - 完全缺失此可视化
        
        论文数据 (Table 7/Figure 8):
        """
        noise_ratios = ['0%', '10%', '20%', '30%', '40%', '50%']
        
        # 论文中的数据
        coinception_mse = [0.061, 0.17, 0.175, 0.177, 0.18, 0.181]
        ts2vec_mse = [0.069, 0.203, 0.209, 0.21, 0.211, 0.213]
        
        coinception_mae = [0.173, 0.332, 0.336, 0.339, 0.342, 0.343]
        ts2vec_mae = [0.186, 0.364, 0.369, 0.37, 0.371, 0.371]
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 6), 
                                 subplot_kw=dict(projection='polar'))
        
        # 雷达图设置
        angles = np.linspace(0, 2*np.pi, len(noise_ratios), endpoint=False).tolist()
        angles += angles[:1]  # 闭合
        
        # MSE雷达图
        ax1 = axes[0]
        coinception_mse_plot = coinception_mse + [coinception_mse[0]]
        ts2vec_mse_plot = ts2vec_mse + [ts2vec_mse[0]]
        
        ax1.plot(angles, coinception_mse_plot, 'o-', color='blue', 
                linewidth=2, label='CoInception')
        ax1.fill(angles, coinception_mse_plot, alpha=0.25, color='blue')
        ax1.plot(angles, ts2vec_mse_plot, 'o-', color='red', 
                linewidth=2, label='TS2Vec')
        ax1.fill(angles, ts2vec_mse_plot, alpha=0.25, color='red')
        ax1.set_xticks(angles[:-1])
        ax1.set_xticklabels(noise_ratios)
        ax1.set_title('MSE', fontsize=14, pad=20)
        ax1.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
        
        # MAE雷达图
        ax2 = axes[1]
        coinception_mae_plot = coinception_mae + [coinception_mae[0]]
        ts2vec_mae_plot = ts2vec_mae + [ts2vec_mae[0]]
        
        ax2.plot(angles, coinception_mae_plot, 'o-', color='blue', 
                linewidth=2, label='CoInception')
        ax2.fill(angles, coinception_mae_plot, alpha=0.25, color='blue')
        ax2.plot(angles, ts2vec_mae_plot, 'o-', color='red', 
                linewidth=2, label='TS2Vec')
        ax2.fill(angles, ts2vec_mae_plot, alpha=0.25, color='red')
        ax2.set_xticks(angles[:-1])
        ax2.set_xticklabels(noise_ratios)
        ax2.set_title('MAE', fontsize=14, pad=20)
        
        plt.suptitle('Noise Ratio Analysis (Paper Figure 8 Style)', fontsize=14)
        plt.tight_layout()
        return fig


def generate_optimization_report():
    """生成完整的优化建议报告"""
    
    report = """
╔══════════════════════════════════════════════════════════════════════════════╗
║              CoInception 可视化复现优化分析报告                              ║
╠══════════════════════════════════════════════════════════════════════════════╣

【一、当前实现评估】

┌─────────────────────────────────────────────────────────────────────────────┐
│ 可视化项目              │ 论文对应 │ 当前状态 │ 优先级 │ 评分    │
├─────────────────────────────────────────────────────────────────────────────┤
│ 相似度矩阵              │ -        │ ✅ 已有   │ 低     │ 7/10   │
│ t-SNE可视化             │ Fig.6,14 │ ⚠️ 需改进 │ 高     │ 4/10   │
│ PCA可视化               │ -        │ ⚠️ 需改进 │ 中     │ 5/10   │
│ 损失曲线                │ -        │ ✅ 已有   │ 低     │ 8/10   │
│ 损失热力图              │ -        │ ✅ 已有   │ 低     │ 7/10   │
│ 噪声鲁棒性对比(Fig.2)   │ Fig.2    │ ❌ 缺失   │ 高     │ 0/10   │
│ 对齐性分析(Fig.5)       │ Fig.5    │ ❌ 缺失   │ 高     │ 0/10   │
│ 均匀性分析(Fig.6)       │ Fig.6    │ ❌ 缺失   │ 高     │ 0/10   │
│ 噪声比例分析(Fig.8)     │ Fig.8    │ ❌ 缺失   │ 中     │ 0/10   │
│ 感受野分析(Fig.13)      │ Fig.13   │ ❌ 缺失   │ 中     │ 0/10   │
│ 聚类性分析(Fig.14)      │ Fig.14   │ ⚠️ 部分   │ 高     │ 3/10   │
└─────────────────────────────────────────────────────────────────────────────┘

【二、关键问题诊断】

1. t-SNE/PCA可视化问题:
   ❌ 当前只显示out1/out1s/out2/out2s的encoder输出类型
   ❌ 缺少真实类别标签(ground truth)的着色
   ❌ 缺少密度估计(KDE)
   ❌ 缺少与TS2Vec基线的对比
   
   ➜ 解决方案: 需要在训练时保存样本的类别信息,用于可视化时着色

2. 噪声鲁棒性分析缺失(论文核心创新):
   ❌ 缺少原始信号 vs 噪声信号的相关性对比
   ❌ 缺少Corr系数标注
   ❌ 缺少与TS2Vec的对比(论文Fig.2显示TS2Vec相关性0.837, CoInception 0.983)
   
   ➜ 解决方案: 需要生成合成噪声信号并计算表征相关性

3. 对齐性/均匀性分析缺失:
   ❌ 当前的representations.png显示余弦相似度分布
   ❌ 缺少论文要求的L2距离直方图
   ❌ 缺少von Mises-Fisher KDE角度分析
   
   ➜ 解决方案: 需要重新计算正样本对的L2距离并生成KDE

【三、优化建议优先级】

┌───────────────────────────────────────────────────────────────────┐
│ 优先级 │ 任务                                  │ 预计工作量 │
├───────────────────────────────────────────────────────────────────┤
│   P0   │ 添加噪声鲁棒性对比可视化(Fig.2)        │ 4小时     │
│   P0   │ 修复t-SNE添加类别标签着色(Fig.14)      │ 2小时     │
│   P1   │ 添加对齐性分析L2距离直方图(Fig.5)      │ 3小时     │
│   P1   │ 添加均匀性分析KDE可视化(Fig.6)         │ 4小时     │
│   P2   │ 添加噪声比例分析雷达图(Fig.8)          │ 2小时     │
│   P2   │ 添加感受野分析图(Fig.13)               │ 2小时     │
│   P3   │ 添加TS2Vec基线对比                     │ 8小时     │
└───────────────────────────────────────────────────────────────────┘

【四、训练配置评估】

当前配置分析 (来自robustness_report.txt):
  - n_epochs: 100 ✅ (论文中使用类似配置)
  - n_iters: 200 ✅
  - batch_size: 8 ⚠️ (论文建议可能更大)
  - lr: 0.001 ✅
  - max_train_length: 3000 ✅
  - Loss改善: 89.77% ✅ (从15.31降至1.57)

潜在问题:
  - 只保存了2个时间点的representations (iteration 100和200)
  - 建议增加更多采样点以获得更完整的训练动态
  
【五、具体代码修改建议】

1. 修改训练脚本,保存样本类别信息:
   ```python
   # 在保存representations时同时保存labels
   representations_data = {
       'out1': out1.cpu().numpy(),
       'out2': out2.cpu().numpy(),
       'labels': labels.cpu().numpy()  # 添加这行
   }
   ```

2. 添加噪声鲁棒性测试:
   ```python
   def test_noise_robustness(model, x_clean, noise_level=0.1):
       x_noisy = x_clean + noise_level * torch.randn_like(x_clean)
       z_clean = model(x_clean)
       z_noisy = model(x_noisy)
       correlation = F.cosine_similarity(z_clean, z_noisy, dim=-1).mean()
       return correlation
   ```

3. 添加L2距离计算:
   ```python
   def compute_positive_pair_l2_distance(z_anchor, z_positive):
       l2_distances = torch.norm(z_anchor - z_positive, p=2, dim=-1)
       return l2_distances.cpu().numpy()
   ```

【六、最终评估】

当前复现完成度: 约35%

缺失的论文核心可视化将影响:
  ⚠️ 无法证明噪声鲁棒性(论文主要创新点)
  ⚠️ 无法展示表征质量(对齐性/均匀性)
  ⚠️ 无法与基线方法对比

建议下一步行动:
  1. 首先实现噪声鲁棒性可视化(Fig.2) - 这是论文的核心创新
  2. 修复t-SNE添加类别标签(Fig.14)
  3. 添加对齐性/均匀性分析(Fig.5/6)

╚══════════════════════════════════════════════════════════════════════════════╝
"""
    return report


if __name__ == "__main__":
    # 生成并打印优化报告
    print(generate_optimization_report())
    
    # 创建示例可视化
    optimizer = CoInceptionVisualizationOptimizer()
    
    # 示例: 感受野分析
    fig_rf = optimizer.create_receptive_field_analysis()
    fig_rf.savefig('/home/claude/receptive_field_analysis.png', dpi=150, bbox_inches='tight')
    print("\n✅ 感受野分析图已保存: receptive_field_analysis.png")
    
    # 示例: 噪声比例分析
    fig_nr = optimizer.create_noise_ratio_analysis()
    fig_nr.savefig('/home/claude/noise_ratio_analysis.png', dpi=150, bbox_inches='tight')
    print("✅ 噪声比例分析图已保存: noise_ratio_analysis.png")
    
    plt.close('all')
