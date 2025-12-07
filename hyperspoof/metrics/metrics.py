"""
Evaluation metrics for face anti-spoofing.
"""

import torch
import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, roc_curve, precision_recall_curve
)
from typing import Dict, List, Tuple, Optional, Union
import warnings


def calculate_metrics(
    y_true: Union[np.ndarray, torch.Tensor, List],
    y_pred: Union[np.ndarray, torch.Tensor, List],
    y_prob: Optional[Union[np.ndarray, torch.Tensor, List]] = None,
    metrics: Optional[List[str]] = None,
) -> Dict[str, float]:
    """
    Calculate evaluation metrics for face anti-spoofing.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        y_prob: Predicted probabilities (optional)
        metrics: List of metrics to calculate (optional)
        
    Returns:
        Dictionary containing calculated metrics
    """
    # Convert to numpy arrays
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.cpu().numpy()
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.cpu().numpy()
    if y_prob is not None and isinstance(y_prob, torch.Tensor):
        y_prob = y_prob.cpu().numpy()
    
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    if y_prob is not None:
        y_prob = np.array(y_prob)
        if y_prob.ndim > 1:
            y_prob = y_prob[:, 1]  # Use probability of positive class
    
    # Default metrics
    if metrics is None:
        metrics = ['accuracy', 'precision', 'recall', 'f1', 'auc']
    
    results = {}
    
    # Basic classification metrics
    if 'accuracy' in metrics:
        results['accuracy'] = accuracy_score(y_true, y_pred)
    
    if 'precision' in metrics:
        results['precision'] = precision_score(y_true, y_pred, average='binary', zero_division=0)
    
    if 'recall' in metrics:
        results['recall'] = recall_score(y_true, y_pred, average='binary', zero_division=0)
    
    if 'f1' in metrics:
        results['f1'] = f1_score(y_true, y_pred, average='binary', zero_division=0)
    
    # AUC metrics
    if 'auc' in metrics and y_prob is not None:
        try:
            results['auc'] = roc_auc_score(y_true, y_prob)
        except ValueError:
            results['auc'] = 0.0
    
    # Face anti-spoofing specific metrics
    if 'acer' in metrics:
        results['acer'] = calculate_acer(y_true, y_pred)
    
    if 'hter' in metrics:
        results['hter'] = calculate_hter(y_true, y_pred)
    
    if 'eer' in metrics and y_prob is not None:
        results['eer'] = calculate_eer(y_true, y_prob)
    
    # Additional metrics
    if 'specificity' in metrics:
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        results['specificity'] = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    
    if 'sensitivity' in metrics:
        results['sensitivity'] = recall_score(y_true, y_pred, average='binary', zero_division=0)
    
    return results


def calculate_acer(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculate Attack Classification Error Rate (ACER).
    
    ACER = (APCER + BPCER) / 2
    where:
    - APCER = Attack Presentation Classification Error Rate
    - BPCER = Bona Fide Presentation Classification Error Rate
    
    Args:
        y_true: True labels (0: real, 1: spoof)
        y_pred: Predicted labels
        
    Returns:
        ACER value
    """
    # Calculate APCER (Attack Presentation Classification Error Rate)
    attack_mask = y_true == 1
    if np.sum(attack_mask) > 0:
        apcer = np.sum((y_pred[attack_mask] == 0)) / np.sum(attack_mask)
    else:
        apcer = 0.0
    
    # Calculate BPCER (Bona Fide Presentation Classification Error Rate)
    real_mask = y_true == 0
    if np.sum(real_mask) > 0:
        bpcer = np.sum((y_pred[real_mask] == 1)) / np.sum(real_mask)
    else:
        bpcer = 0.0
    
    # Calculate ACER
    acer = (apcer + bpcer) / 2.0
    
    return acer


def calculate_hter(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculate Half Total Error Rate (HTER).
    
    HTER = (FAR + FRR) / 2
    where:
    - FAR = False Acceptance Rate
    - FRR = False Rejection Rate
    
    Args:
        y_true: True labels (0: real, 1: spoof)
        y_pred: Predicted labels
        
    Returns:
        HTER value
    """
    # Calculate FAR (False Acceptance Rate) - spoof accepted as real
    attack_mask = y_true == 1
    if np.sum(attack_mask) > 0:
        far = np.sum((y_pred[attack_mask] == 0)) / np.sum(attack_mask)
    else:
        far = 0.0
    
    # Calculate FRR (False Rejection Rate) - real rejected as spoof
    real_mask = y_true == 0
    if np.sum(real_mask) > 0:
        frr = np.sum((y_pred[real_mask] == 1)) / np.sum(real_mask)
    else:
        frr = 0.0
    
    # Calculate HTER
    hter = (far + frr) / 2.0
    
    return hter


def calculate_eer(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """
    Calculate Equal Error Rate (EER).
    
    Args:
        y_true: True labels (0: real, 1: spoof)
        y_prob: Predicted probabilities for spoof class
        
    Returns:
        EER value
    """
    try:
        fpr, tpr, thresholds = roc_curve(y_true, y_prob)
        
        # Find the threshold where FPR = 1 - TPR (EER)
        fnr = 1 - tpr
        eer_threshold = thresholds[np.nanargmin(np.absolute((fnr - fpr)))]
        eer = fpr[np.nanargmin(np.absolute((fnr - fpr)))]
        
        return eer
    except:
        return 0.0


def calculate_confusion_matrix_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """
    Calculate metrics from confusion matrix.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        
    Returns:
        Dictionary containing confusion matrix metrics
    """
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    
    metrics = {
        'true_negatives': tn,
        'false_positives': fp,
        'false_negatives': fn,
        'true_positives': tp,
        'total_samples': tn + fp + fn + tp,
    }
    
    # Calculate rates
    if (tp + fn) > 0:
        metrics['sensitivity'] = tp / (tp + fn)  # Recall
    else:
        metrics['sensitivity'] = 0.0
    
    if (tn + fp) > 0:
        metrics['specificity'] = tn / (tn + fp)
    else:
        metrics['specificity'] = 0.0
    
    if (tp + fp) > 0:
        metrics['precision'] = tp / (tp + fp)
    else:
        metrics['precision'] = 0.0
    
    if (tp + fn) > 0:
        metrics['recall'] = tp / (tp + fn)
    else:
        metrics['recall'] = 0.0
    
    return metrics


def calculate_roc_metrics(y_true: np.ndarray, y_prob: np.ndarray) -> Dict[str, float]:
    """
    Calculate ROC curve metrics.
    
    Args:
        y_true: True labels
        y_prob: Predicted probabilities
        
    Returns:
        Dictionary containing ROC metrics
    """
    try:
        fpr, tpr, thresholds = roc_curve(y_true, y_prob)
        auc = roc_auc_score(y_true, y_prob)
        
        # Find optimal threshold (Youden's J statistic)
        j_scores = tpr - fpr
        optimal_idx = np.argmax(j_scores)
        optimal_threshold = thresholds[optimal_idx]
        
        return {
            'auc': auc,
            'optimal_threshold': optimal_threshold,
            'optimal_tpr': tpr[optimal_idx],
            'optimal_fpr': fpr[optimal_idx],
        }
    except:
        return {
            'auc': 0.0,
            'optimal_threshold': 0.5,
            'optimal_tpr': 0.0,
            'optimal_fpr': 0.0,
        }


def calculate_precision_recall_metrics(y_true: np.ndarray, y_prob: np.ndarray) -> Dict[str, float]:
    """
    Calculate precision-recall curve metrics.
    
    Args:
        y_true: True labels
        y_prob: Predicted probabilities
        
    Returns:
        Dictionary containing precision-recall metrics
    """
    try:
        precision, recall, thresholds = precision_recall_curve(y_true, y_prob)
        
        # Calculate average precision
        avg_precision = np.trapz(precision, recall)
        
        # Find optimal threshold (F1 score)
        f1_scores = 2 * (precision * recall) / (precision + recall + 1e-8)
        optimal_idx = np.argmax(f1_scores)
        optimal_threshold = thresholds[optimal_idx] if optimal_idx < len(thresholds) else 0.5
        
        return {
            'average_precision': avg_precision,
            'optimal_threshold': optimal_threshold,
            'optimal_precision': precision[optimal_idx],
            'optimal_recall': recall[optimal_idx],
            'optimal_f1': f1_scores[optimal_idx],
        }
    except:
        return {
            'average_precision': 0.0,
            'optimal_threshold': 0.5,
            'optimal_precision': 0.0,
            'optimal_recall': 0.0,
            'optimal_f1': 0.0,
        }
