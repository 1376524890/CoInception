#!/usr/bin/env python3
"""
Visualization Utility Functions for CoInception

This module provides utility functions for creating visualizations that match the exact format
of the CoInception paper figures.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap
from typing import Dict, List, Tuple, Optional, Union
import os

# Set default font to match paper requirements
# Configure matplotlib to use system fonts without warnings
import warnings
warnings.filterwarnings('ignore', category=UserWarning)

# Configure matplotlib font settings
import matplotlib.font_manager as fm

# Find and set DejaVu Serif font properly
font_list = fm.findSystemFonts()
dejavu_fonts = [f for f in font_list if 'DejaVuSerif' in f]

if dejavu_fonts:
    # Use the first available DejaVu Serif font
    font_path = dejavu_fonts[0]
    font_prop = fm.FontProperties(fname=font_path)
    plt.rcParams['font.family'] = font_prop.get_name()
    plt.rcParams['font.serif'] = font_prop.get_name()
else:
    # Fallback to default serif
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = 'serif'

plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.size'] = 10

print(f"Font configured: {plt.rcParams['font.family']}")


def create_directory(dir_path: str) -> None:
    """Create a directory if it doesn't exist.
    
    Args:
        dir_path (str): Directory path to create.
    """
    os.makedirs(dir_path, exist_ok=True)


def save_figure(fig: plt.Figure, filename: str, dpi: int = 300, 
                bbox_inches: str = 'tight') -> None:
    """Save a figure to a file.
    
    Args:
        fig (plt.Figure): Figure to save.
        filename (str): Filename to save the figure as.
        dpi (int): Dots per inch for the output file.
        bbox_inches (str): Bounding box inches for the figure.
    """
    # Ensure directory exists
    dir_path = os.path.dirname(filename)
    if dir_path:
        create_directory(dir_path)
    
    fig.savefig(filename, dpi=dpi, bbox_inches=bbox_inches)
    plt.close(fig)
    print(f"Saved figure: {filename}")


def set_font_sizes(axis_label: int = 15, tick: int = 10, 
                   title: int = 12, legend: int = 10) -> None:
    """Set font sizes for a plot.
    
    Args:
        axis_label (int): Font size for axis labels.
        tick (int): Font size for tick labels.
        title (int): Font size for titles.
        legend (int): Font size for legends.
    """
    plt.rcParams['axes.labelsize'] = axis_label
    plt.rcParams['xtick.labelsize'] = tick
    plt.rcParams['ytick.labelsize'] = tick
    plt.rcParams['axes.titlesize'] = title
    plt.rcParams['legend.fontsize'] = legend


def plot_waveform(ax: plt.Axes, data: np.ndarray, color: str = 'blue', 
                  label: Optional[str] = None, linewidth: float = 1.0) -> None:
    """Plot a waveform.
    
    Args:
        ax (plt.Axes): Axis to plot on.
        data (np.ndarray): Waveform data.
        color (str): Color of the waveform.
        label (Optional[str]): Label for the waveform.
        linewidth (float): Line width of the waveform.
    """
    ax.plot(data, color=color, linewidth=linewidth, label=label)
    ax.set_ylabel('Amp')
    ax.set_xlabel('Time Step')
    ax.grid(False)


def plot_heatmap(ax: plt.Axes, data: np.ndarray, cmap: str = 'RdPu', 
                 xlabel: str = 'Time Step', ylabel: str = 'Dim') -> None:
    """Plot a heatmap.
    
    Args:
        ax (plt.Axes): Axis to plot on.
        data (np.ndarray): Heatmap data.
        cmap (str): Colormap to use.
        xlabel (str): X-axis label.
        ylabel (str): Y-axis label.
    """
    sns.heatmap(data, cmap=cmap, ax=ax, cbar_kws={'shrink': 0.8})
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(False)


def plot_correlation_label(ax: plt.Axes, corr: float, position: Tuple[float, float] = (0.95, 0.95), 
                           fontsize: int = 12, color: str = 'black') -> None:
    """Plot a correlation label on an axis.
    
    Args:
        ax (plt.Axes): Axis to plot on.
        corr (float): Correlation coefficient.
        position (Tuple[float, float]): Position of the label (normalized coordinates).
        fontsize (int): Font size of the label.
        color (str): Color of the label.
    """
    ax.text(position[0], position[1], f"Corr: {corr:.3f}", 
            horizontalalignment='right', verticalalignment='top',
            transform=ax.transAxes, fontsize=fontsize, color=color)


def create_figure(size: Tuple[float, float]) -> plt.Figure:
    """Create a figure with the specified size.
    
    Args:
        size (Tuple[float, float]): Size of the figure (width, height) in inches.
        
    Returns:
        plt.Figure: Created figure.
    """
    return plt.figure(figsize=size)


def plot_histogram(ax: plt.Axes, data: np.ndarray, bins: int = 50, 
                   color: str = 'blue', edgecolor: str = 'black', 
                   alpha: float = 0.7) -> None:
    """Plot a histogram.
    
    Args:
        ax (plt.Axes): Axis to plot on.
        data (np.ndarray): Data to plot.
        bins (int): Number of bins.
        color (str): Color of the histogram.
        edgecolor (str): Edge color of the histogram.
        alpha (float): Alpha value of the histogram.
    """
    ax.hist(data, bins=bins, color=color, edgecolor=edgecolor, alpha=alpha)


def plot_mean_line(ax: plt.Axes, data: np.ndarray, color: str = 'black', 
                   linestyle: str = '--', linewidth: float = 1.5, 
                   label: Optional[str] = None) -> None:
    """Plot a mean line on a histogram.
    
    Args:
        ax (plt.Axes): Axis to plot on.
        data (np.ndarray): Data to calculate the mean from.
        color (str): Color of the mean line.
        linestyle (str): Line style of the mean line.
        linewidth (float): Line width of the mean line.
        label (Optional[str]): Label for the mean line.
    """
    mean = np.mean(data)
    ax.axvline(mean, color=color, linestyle=linestyle, linewidth=linewidth, label=label)
    return mean


def plot_gaussian_kde(ax: plt.Axes, data: np.ndarray, color: str = 'blue', 
                      label: Optional[str] = None, linewidth: float = 1.5) -> None:
    """Plot a Gaussian KDE.
    
    Args:
        ax (plt.Axes): Axis to plot on.
        data (np.ndarray): Data to plot KDE for.
        color (str): Color of the KDE line.
        label (Optional[str]): Label for the KDE.
        linewidth (float): Line width of the KDE line.
    """
    sns.kdeplot(data=data, ax=ax, color=color, linewidth=linewidth, label=label)


def plot_2d_kde(ax: plt.Axes, x: np.ndarray, y: np.ndarray, cmap: str = 'viridis', 
                 alpha: float = 0.7) -> None:
    """Plot a 2D KDE.
    
    Args:
        ax (plt.Axes): Axis to plot on.
        x (np.ndarray): X data.
        y (np.ndarray): Y data.
        cmap (str): Colormap to use.
        alpha (float): Alpha value for the KDE.
    """
    sns.kdeplot(x=x, y=y, cmap=cmap, fill=True, alpha=alpha, ax=ax)


def plot_vmf_kde_ring(ax: plt.Axes, data: np.ndarray, color: str = 'purple', 
                       alpha: float = 0.7, bins: int = 100) -> None:
    """Plot a von Mises-Fisher KDE ring.
    
    Args:
        ax (plt.Axes): Axis to plot on.
        data (np.ndarray): Data to plot.
        color (str): Color of the ring.
        alpha (float): Alpha value of the ring.
        bins (int): Number of bins for the histogram.
    """
    # Convert data to angles
    angles = np.arctan2(data[:, 1], data[:, 0])
    radii = np.sqrt(data[:, 0]**2 + data[:, 1]**2)
    
    # Create a polar histogram
    ax.hist(angles, bins=bins, weights=radii, color=color, alpha=alpha, 
            orientation='horizontal')
    
    # Set axis properties
    ax.set_yticklabels([])
    ax.set_xticklabels([])
    ax.set_ylim(0, np.max(radii) * 1.1)


def plot_radial_histogram(ax: plt.Axes, data: np.ndarray, color: str = 'purple', 
                           bins: int = 50, alpha: float = 0.7) -> None:
    """Plot a radial histogram.
    
    Args:
        ax (plt.Axes): Axis to plot on.
        data (np.ndarray): Data to plot.
        color (str): Color of the histogram.
        bins (int): Number of bins.
        alpha (float): Alpha value of the histogram.
    """
    ax.hist(data, bins=bins, color=color, alpha=alpha, edgecolor='black')
    ax.set_xlim(-1.5, 1.5)
    ax.grid(False)


def plot_critical_difference(ax: plt.Axes, ranks: Dict[str, float], cd: float, 
                             title: str = "Critical Difference Diagram") -> None:
    """Plot a critical difference diagram.
    
    Args:
        ax (plt.Axes): Axis to plot on.
        ranks (Dict[str, float]): Dictionary of method names to their ranks.
        cd (float): Critical difference value.
        title (str): Title of the plot.
    """
    # Sort methods by rank
    sorted_methods = sorted(ranks.items(), key=lambda x: x[1])
    method_names, method_ranks = zip(*sorted_methods)
    
    # Plot horizontal lines for each method
    y_pos = np.arange(len(method_names))
    ax.hlines(y_pos, 0, method_ranks, color='black', linewidth=2)
    ax.plot(method_ranks, y_pos, 'o', color='black', markersize=8)
    
    # Plot critical difference line
    ax.axvline(cd, color='black', linestyle='--', linewidth=1.5, label=f'CD = {cd:.2f}')
    
    # Set labels and ticks
    ax.set_yticks(y_pos)
    ax.set_yticklabels(method_names)
    ax.set_xlabel('Rank')
    ax.set_title(title)
    ax.grid(False)
    ax.legend()


def plot_hexagon_3d(ax: plt.Axes, x: np.ndarray, y: np.ndarray, z: np.ndarray, 
                     color: str = 'red', alpha: float = 0.7, edgecolor: str = 'black') -> None:
    """Plot a 3D hexagon plot.
    
    Args:
        ax (plt.Axes): Axis to plot on.
        x (np.ndarray): X data.
        y (np.ndarray): Y data.
        z (np.ndarray): Z data.
        color (str): Color of the hexagons.
        alpha (float): Alpha value of the hexagons.
        edgecolor (str): Edge color of the hexagons.
    """
    # This is a placeholder for 3D hexagon plotting
    # In practice, this would use a 3D plotting library or a different approach
    ax.scatter(x, y, z, c=color, alpha=alpha, edgecolor=edgecolor)
    ax.set_box_aspect([1, 1, 1])


def create_2x2_grid(size: Tuple[float, float]) -> Tuple[plt.Figure, np.ndarray]:
    """Create a 2x2 grid of subplots.
    
    Args:
        size (Tuple[float, float]): Size of the figure (width, height) in inches.
        
    Returns:
        Tuple[plt.Figure, np.ndarray]: Figure and array of axes.
    """
    fig, axs = plt.subplots(2, 2, figsize=size, squeeze=False)
    fig.subplots_adjust(hspace=0.4, wspace=0.3)
    return fig, axs


def create_1x2_grid(size: Tuple[float, float]) -> Tuple[plt.Figure, np.ndarray]:
    """Create a 1x2 grid of subplots.
    
    Args:
        size (Tuple[float, float]): Size of the figure (width, height) in inches.
        
    Returns:
        Tuple[plt.Figure, np.ndarray]: Figure and array of axes.
    """
    fig, axs = plt.subplots(1, 2, figsize=size, squeeze=False)
    fig.subplots_adjust(wspace=0.3)
    return fig, axs


def create_2x4_grid(size: Tuple[float, float]) -> Tuple[plt.Figure, np.ndarray]:
    """Create a 2x4 grid of subplots.
    
    Args:
        size (Tuple[float, float]): Size of the figure (width, height) in inches.
        
    Returns:
        Tuple[plt.Figure, np.ndarray]: Figure and array of axes.
    """
    fig, axs = plt.subplots(2, 4, figsize=size, squeeze=False)
    fig.subplots_adjust(hspace=0.5, wspace=0.3)
    return fig, axs


def add_subplot_title(ax: plt.Axes, title: str, fontsize: int = 12, 
                      fontweight: str = 'bold') -> None:
    """Add a title to a subplot.
    
    Args:
        ax (plt.Axes): Axis to add the title to.
        title (str): Title text.
        fontsize (int): Font size of the title.
        fontweight (str): Font weight of the title.
    """
    ax.set_title(title, fontsize=fontsize, fontweight=fontweight)


def add_suptitle(fig: plt.Figure, title: str, fontsize: int = 14, 
                 fontweight: str = 'bold') -> None:
    """Add a super title to a figure.
    
    Args:
        fig (plt.Figure): Figure to add the title to.
        title (str): Title text.
        fontsize (int): Font size of the title.
        fontweight (str): Font weight of the title.
    """
    fig.suptitle(title, fontsize=fontsize, fontweight=fontweight)


def calculate_correlation(x: np.ndarray, y: np.ndarray) -> float:
    """Calculate the correlation between two arrays.
    
    Args:
        x (np.ndarray): First array.
        y (np.ndarray): Second array.
        
    Returns:
        float: Correlation coefficient.
    """
    return np.corrcoef(x.flatten(), y.flatten())[0, 1]


def normalize_embeddings(embeddings: np.ndarray) -> np.ndarray:
    """Normalize embeddings to unit length.
    
    Args:
        embeddings (np.ndarray): Embeddings to normalize.
        
    Returns:
        np.ndarray: Normalized embeddings.
    """
    norms = np.linalg.norm(embeddings, axis=-1, keepdims=True)
    return embeddings / (norms + 1e-8)


def calculate_pairwise_distances(embeddings: np.ndarray) -> np.ndarray:
    """Calculate pairwise L2 distances between embeddings.
    
    Args:
        embeddings (np.ndarray): Embeddings to calculate distances between.
        
    Returns:
        np.ndarray: Pairwise distances.
    """
    n = embeddings.shape[0]
    distances = np.zeros((n, n))
    
    for i in range(n):
        for j in range(i+1, n):
            distances[i, j] = np.linalg.norm(embeddings[i] - embeddings[j])
            distances[j, i] = distances[i, j]
    
    return distances


def get_positive_pairs(distances: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """Get positive pairs from distances matrix based on labels.
    
    Args:
        distances (np.ndarray): Pairwise distances matrix.
        labels (np.ndarray): Labels for each instance.
        
    Returns:
        np.ndarray: Positive pairs distances.
    """
    positive_pairs = []
    n = len(labels)
    
    for i in range(n):
        for j in range(i+1, n):
            if labels[i] == labels[j]:
                positive_pairs.append(distances[i, j])
    
    return np.array(positive_pairs)


def create_color_palette(n_colors: int, palette: str = 'RdPu') -> List[str]:
    """Create a color palette.
    
    Args:
        n_colors (int): Number of colors to create.
        palette (str): Name of the palette to use.
        
    Returns:
        List[str]: List of colors.
    """
    return sns.color_palette(palette, n_colors)


def plot_class_distribution(ax: plt.Axes, labels: np.ndarray, 
                            color: str = 'blue', alpha: float = 0.7) -> None:
    """Plot class distribution.
    
    Args:
        ax (plt.Axes): Axis to plot on.
        labels (np.ndarray): Class labels.
        color (str): Color of the bars.
        alpha (float): Alpha value of the bars.
    """
    classes, counts = np.unique(labels, return_counts=True)
    ax.bar(classes, counts, color=color, alpha=alpha, edgecolor='black')
    ax.set_xlabel('Class')
    ax.set_ylabel('Count')
    ax.grid(False)


def plot_confusion_matrix(ax: plt.Axes, cm: np.ndarray, classes: List[str], 
                          normalize: bool = False, title: str = 'Confusion Matrix', 
                          cmap: str = 'RdPu') -> None:
    """Plot a confusion matrix.
    
    Args:
        ax (plt.Axes): Axis to plot on.
        cm (np.ndarray): Confusion matrix.
        classes (List[str]): List of class names.
        normalize (bool): Whether to normalize the confusion matrix.
        title (str): Title of the plot.
        cmap (str): Colormap to use.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    sns.heatmap(cm, annot=True, fmt='.2f' if normalize else 'd', 
                xticklabels=classes, yticklabels=classes, 
                cmap=cmap, ax=ax, cbar=False)
    
    ax.set_title(title)
    ax.set_xlabel('Predicted Label')
    ax.set_ylabel('True Label')
    ax.grid(False)


def plot_learning_curve(ax: plt.Axes, train_loss: List[float], 
                        val_loss: Optional[List[float]] = None, 
                        title: str = 'Learning Curve') -> None:
    """Plot a learning curve.
    
    Args:
        ax (plt.Axes): Axis to plot on.
        train_loss (List[float]): Training loss values.
        val_loss (Optional[List[float]]): Validation loss values.
        title (str): Title of the plot.
    """
    epochs = range(1, len(train_loss) + 1)
    ax.plot(epochs, train_loss, 'b-', label='Training Loss')
    
    if val_loss is not None:
        ax.plot(epochs, val_loss, 'r-', label='Validation Loss')
    
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title(title)
    ax.legend()
    ax.grid(False)


def plot_noise_robustness(ax: plt.Axes, noise_levels: List[float], 
                          metrics: Dict[str, List[float]], 
                          title: str = 'Noise Robustness') -> None:
    """Plot noise robustness.
    
    Args:
        ax (plt.Axes): Axis to plot on.
        noise_levels (List[float]): List of noise levels.
        metrics (Dict[str, List[float]]): Dictionary of metric names to their values.
        title (str): Title of the plot.
    """
    for name, values in metrics.items():
        ax.plot(noise_levels, values, marker='o', label=name)
    
    ax.set_xlabel('Noise Level')
    ax.set_ylabel('Performance Metric')
    ax.set_title(title)
    ax.legend()
    ax.grid(False)


def plot_forecast(ax: plt.Axes, actual: np.ndarray, forecast: np.ndarray, 
                  title: str = 'Forecast') -> None:
    """Plot a forecast.
    
    Args:
        ax (plt.Axes): Axis to plot on.
        actual (np.ndarray): Actual values.
        forecast (np.ndarray): Forecasted values.
        title (str): Title of the plot.
    """
    t = np.arange(len(actual))
    ax.plot(t, actual, 'b-', label='Actual')
    ax.plot(t, forecast, 'r-', label='Forecast')
    
    ax.set_xlabel('Time')
    ax.set_ylabel('Value')
    ax.set_title(title)
    ax.legend()
    ax.grid(False)


def plot_anomaly_detection(ax: plt.Axes, data: np.ndarray, 
                           anomalies: np.ndarray, scores: np.ndarray, 
                           title: str = 'Anomaly Detection') -> None:
    """Plot anomaly detection results.
    
    Args:
        ax (plt.Axes): Axis to plot on.
        data (np.ndarray): Time series data.
        anomalies (np.ndarray): Boolean array indicating anomalies.
        scores (np.ndarray): Anomaly scores.
        title (str): Title of the plot.
    """
    # Create a twin axis for scores
    ax2 = ax.twinx()
    
    # Plot data
    ax.plot(data, 'b-', label='Data')
    
    # Plot anomalies
    ax.scatter(np.where(anomalies)[0], data[anomalies], color='red', 
               marker='o', s=100, label='Anomalies')
    
    # Plot scores
    ax2.plot(scores, 'g-', label='Anomaly Score')
    
    # Set labels
    ax.set_xlabel('Time')
    ax.set_ylabel('Data Value')
    ax2.set_ylabel('Anomaly Score')
    
    # Set title
    ax.set_title(title)
    
    # Add legend
    lines, labels = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax2.legend(lines + lines2, labels + labels2, loc='upper right')
    
    # Turn off grid
    ax.grid(False)
    ax2.grid(False)
