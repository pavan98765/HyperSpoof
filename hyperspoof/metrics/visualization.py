"""
Visualization utilities for face anti-spoofing evaluation.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import torch
from sklearn.metrics import confusion_matrix, roc_curve, precision_recall_curve
from typing import Dict, List, Optional, Tuple, Union
import os


def plot_confusion_matrix(
    y_true: Union[np.ndarray, torch.Tensor, List],
    y_pred: Union[np.ndarray, torch.Tensor, List],
    class_names: Optional[List[str]] = None,
    title: str = "Confusion Matrix",
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (8, 6),
    normalize: bool = True,
) -> plt.Figure:
    """
    Plot confusion matrix.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        class_names: Names of classes
        title: Plot title
        save_path: Path to save the plot
        figsize: Figure size
        normalize: Whether to normalize the matrix
        
    Returns:
        Matplotlib figure
    """
    # Convert to numpy arrays
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.cpu().numpy()
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.cpu().numpy()
    
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    # Calculate confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        fmt = '.2f'
    else:
        fmt = 'd'
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot confusion matrix
    sns.heatmap(
        cm,
        annot=True,
        fmt=fmt,
        cmap='Blues',
        xticklabels=class_names or ['Real', 'Spoof'],
        yticklabels=class_names or ['Real', 'Spoof'],
        ax=ax
    )
    
    ax.set_title(title)
    ax.set_xlabel('Predicted Label')
    ax.set_ylabel('True Label')
    
    # Save plot if path provided
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def plot_roc_curve(
    y_true: Union[np.ndarray, torch.Tensor, List],
    y_prob: Union[np.ndarray, torch.Tensor, List],
    title: str = "ROC Curve",
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (8, 6),
) -> plt.Figure:
    """
    Plot ROC curve.
    
    Args:
        y_true: True labels
        y_prob: Predicted probabilities
        title: Plot title
        save_path: Path to save the plot
        figsize: Figure size
        
    Returns:
        Matplotlib figure
    """
    # Convert to numpy arrays
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.cpu().numpy()
    if isinstance(y_prob, torch.Tensor):
        y_prob = y_prob.cpu().numpy()
    
    y_true = np.array(y_true)
    y_prob = np.array(y_prob)
    
    if y_prob.ndim > 1:
        y_prob = y_prob[:, 1]  # Use probability of positive class
    
    # Calculate ROC curve
    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    
    # Calculate AUC
    from sklearn.metrics import roc_auc_score
    auc = roc_auc_score(y_true, y_prob)
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot ROC curve
    ax.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {auc:.3f})')
    ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random')
    
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title(title)
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)
    
    # Save plot if path provided
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def plot_precision_recall_curve(
    y_true: Union[np.ndarray, torch.Tensor, List],
    y_prob: Union[np.ndarray, torch.Tensor, List],
    title: str = "Precision-Recall Curve",
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (8, 6),
) -> plt.Figure:
    """
    Plot precision-recall curve.
    
    Args:
        y_true: True labels
        y_prob: Predicted probabilities
        title: Plot title
        save_path: Path to save the plot
        figsize: Figure size
        
    Returns:
        Matplotlib figure
    """
    # Convert to numpy arrays
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.cpu().numpy()
    if isinstance(y_prob, torch.Tensor):
        y_prob = y_prob.cpu().numpy()
    
    y_true = np.array(y_true)
    y_prob = np.array(y_prob)
    
    if y_prob.ndim > 1:
        y_prob = y_prob[:, 1]  # Use probability of positive class
    
    # Calculate precision-recall curve
    precision, recall, thresholds = precision_recall_curve(y_true, y_prob)
    
    # Calculate average precision
    from sklearn.metrics import average_precision_score
    avg_precision = average_precision_score(y_true, y_prob)
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot precision-recall curve
    ax.plot(recall, precision, color='darkorange', lw=2, 
            label=f'PR curve (AP = {avg_precision:.3f})')
    
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    ax.set_title(title)
    ax.legend(loc="lower left")
    ax.grid(True, alpha=0.3)
    
    # Save plot if path provided
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def plot_training_curves(
    train_losses: List[float],
    val_losses: Optional[List[float]] = None,
    train_metrics: Optional[Dict[str, List[float]]] = None,
    val_metrics: Optional[Dict[str, List[float]]] = None,
    title: str = "Training Curves",
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (15, 10),
) -> plt.Figure:
    """
    Plot training curves.
    
    Args:
        train_losses: Training losses
        val_losses: Validation losses (optional)
        train_metrics: Training metrics (optional)
        val_metrics: Validation metrics (optional)
        title: Plot title
        save_path: Path to save the plot
        figsize: Figure size
        
    Returns:
        Matplotlib figure
    """
    # Determine number of subplots
    num_plots = 1  # Loss plot
    if train_metrics or val_metrics:
        num_plots += 1  # Metrics plot
    
    fig, axes = plt.subplots(1, num_plots, figsize=figsize)
    if num_plots == 1:
        axes = [axes]
    
    # Plot losses
    ax = axes[0]
    epochs = range(1, len(train_losses) + 1)
    ax.plot(epochs, train_losses, 'b-', label='Training Loss', linewidth=2)
    
    if val_losses:
        ax.plot(epochs, val_losses, 'r-', label='Validation Loss', linewidth=2)
    
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title('Training and Validation Loss')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot metrics if provided
    if num_plots > 1 and (train_metrics or val_metrics):
        ax = axes[1]
        
        if train_metrics:
            for metric_name, values in train_metrics.items():
                ax.plot(epochs, values, 'b-', label=f'Training {metric_name}', linewidth=2)
        
        if val_metrics:
            for metric_name, values in val_metrics.items():
                ax.plot(epochs, values, 'r-', label=f'Validation {metric_name}', linewidth=2)
        
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Metric Value')
        ax.set_title('Training and Validation Metrics')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.suptitle(title)
    plt.tight_layout()
    
    # Save plot if path provided
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def plot_attention_maps(
    images: torch.Tensor,
    attention_maps: torch.Tensor,
    titles: Optional[List[str]] = None,
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (15, 5),
) -> plt.Figure:
    """
    Plot attention maps for visualization.
    
    Args:
        images: Input images (B, C, H, W)
        attention_maps: Attention maps (B, C, H, W)
        titles: Optional titles for each image
        save_path: Path to save the plot
        figsize: Figure size
        
    Returns:
        Matplotlib figure
    """
    # Convert to numpy
    if isinstance(images, torch.Tensor):
        images = images.cpu().numpy()
    if isinstance(attention_maps, torch.Tensor):
        attention_maps = attention_maps.cpu().numpy()
    
    batch_size = images.shape[0]
    num_cols = min(batch_size, 4)  # Show up to 4 images per row
    num_rows = (batch_size + num_cols - 1) // num_cols
    
    fig, axes = plt.subplots(num_rows, num_cols * 2, figsize=figsize)
    if num_rows == 1:
        axes = axes.reshape(1, -1)
    
    for i in range(batch_size):
        row = i // num_cols
        col = i % num_cols
        
        # Original image
        ax = axes[row, col * 2]
        img = images[i].transpose(1, 2, 0)
        if img.max() <= 1.0:
            img = img * 255.0
        img = np.clip(img, 0, 255).astype(np.uint8)
        ax.imshow(img)
        ax.set_title(f'Original {i+1}' if not titles else titles[i])
        ax.axis('off')
        
        # Attention map
        ax = axes[row, col * 2 + 1]
        att_map = attention_maps[i].mean(axis=0)  # Average across channels
        im = ax.imshow(att_map, cmap='hot', interpolation='nearest')
        ax.set_title(f'Attention {i+1}')
        ax.axis('off')
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    
    # Hide empty subplots
    for i in range(batch_size, num_rows * num_cols):
        row = i // num_cols
        col = i % num_cols
        axes[row, col * 2].axis('off')
        axes[row, col * 2 + 1].axis('off')
    
    plt.tight_layout()
    
    # Save plot if path provided
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def plot_spectral_bands(
    hsi_data: torch.Tensor,
    band_indices: Optional[List[int]] = None,
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (15, 10),
) -> plt.Figure:
    """
    Plot individual spectral bands from HSI data.
    
    Args:
        hsi_data: HSI data (B, C, H, W) where C is number of spectral bands
        band_indices: Indices of bands to plot (if None, plot all)
        save_path: Path to save the plot
        figsize: Figure size
        
    Returns:
        Matplotlib figure
    """
    # Convert to numpy
    if isinstance(hsi_data, torch.Tensor):
        hsi_data = hsi_data.cpu().numpy()
    
    batch_size, num_bands, height, width = hsi_data.shape
    
    if band_indices is None:
        band_indices = list(range(num_bands))
    
    num_bands_to_plot = len(band_indices)
    num_cols = min(num_bands_to_plot, 5)  # Show up to 5 bands per row
    num_rows = (num_bands_to_plot + num_cols - 1) // num_cols
    
    fig, axes = plt.subplots(num_rows, num_cols, figsize=figsize)
    if num_rows == 1:
        axes = axes.reshape(1, -1)
    
    for i, band_idx in enumerate(band_indices):
        row = i // num_cols
        col = i % num_cols
        
        ax = axes[row, col]
        band_data = hsi_data[0, band_idx]  # Use first sample
        im = ax.imshow(band_data, cmap='viridis')
        ax.set_title(f'Band {band_idx}')
        ax.axis('off')
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    
    # Hide empty subplots
    for i in range(num_bands_to_plot, num_rows * num_cols):
        row = i // num_cols
        col = i % num_cols
        axes[row, col].axis('off')
    
    plt.tight_layout()
    
    # Save plot if path provided
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig
