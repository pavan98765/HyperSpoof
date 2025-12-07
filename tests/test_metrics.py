"""
Tests for evaluation metrics.
"""

import pytest
import torch
import numpy as np
from hyperspoof.metrics import (
    calculate_metrics, calculate_acer, calculate_hter, calculate_eer,
    calculate_confusion_matrix_metrics, calculate_roc_metrics
)


class TestMetrics:
    """Test evaluation metrics."""
    
    def test_calculate_metrics_basic(self):
        """Test basic metrics calculation."""
        y_true = np.array([0, 1, 0, 1, 0, 1])
        y_pred = np.array([0, 1, 0, 0, 1, 1])
        y_prob = np.array([0.8, 0.9, 0.7, 0.3, 0.6, 0.8])
        
        metrics = calculate_metrics(y_true, y_pred, y_prob)
        
        assert 'accuracy' in metrics
        assert 'precision' in metrics
        assert 'recall' in metrics
        assert 'f1' in metrics
        assert 'auc' in metrics
        
        # Check metric values are reasonable
        assert 0 <= metrics['accuracy'] <= 1
        assert 0 <= metrics['precision'] <= 1
        assert 0 <= metrics['recall'] <= 1
        assert 0 <= metrics['f1'] <= 1
        assert 0 <= metrics['auc'] <= 1
    
    def test_calculate_metrics_face_anti_spoofing(self):
        """Test face anti-spoofing specific metrics."""
        y_true = np.array([0, 1, 0, 1, 0, 1])
        y_pred = np.array([0, 1, 0, 0, 1, 1])
        y_prob = np.array([0.8, 0.9, 0.7, 0.3, 0.6, 0.8])
        
        metrics = calculate_metrics(
            y_true, y_pred, y_prob,
            metrics=['accuracy', 'acer', 'hter', 'eer']
        )
        
        assert 'acer' in metrics
        assert 'hter' in metrics
        assert 'eer' in metrics
        
        # Check metric values are reasonable
        assert 0 <= metrics['acer'] <= 1
        assert 0 <= metrics['hter'] <= 1
        assert 0 <= metrics['eer'] <= 1
    
    def test_calculate_acer(self):
        """Test ACER calculation."""
        # Perfect predictions
        y_true = np.array([0, 1, 0, 1])
        y_pred = np.array([0, 1, 0, 1])
        acer = calculate_acer(y_true, y_pred)
        assert acer == 0.0
        
        # All wrong predictions
        y_true = np.array([0, 1, 0, 1])
        y_pred = np.array([1, 0, 1, 0])
        acer = calculate_acer(y_true, y_pred)
        assert acer == 1.0
        
        # Mixed predictions
        y_true = np.array([0, 1, 0, 1, 0, 1])
        y_pred = np.array([0, 1, 0, 0, 1, 1])
        acer = calculate_acer(y_true, y_pred)
        assert 0 <= acer <= 1
    
    def test_calculate_hter(self):
        """Test HTER calculation."""
        # Perfect predictions
        y_true = np.array([0, 1, 0, 1])
        y_pred = np.array([0, 1, 0, 1])
        hter = calculate_hter(y_true, y_pred)
        assert hter == 0.0
        
        # All wrong predictions
        y_true = np.array([0, 1, 0, 1])
        y_pred = np.array([1, 0, 1, 0])
        hter = calculate_hter(y_true, y_pred)
        assert hter == 1.0
        
        # Mixed predictions
        y_true = np.array([0, 1, 0, 1, 0, 1])
        y_pred = np.array([0, 1, 0, 0, 1, 1])
        hter = calculate_hter(y_true, y_pred)
        assert 0 <= hter <= 1
    
    def test_calculate_eer(self):
        """Test EER calculation."""
        # Perfect separation
        y_true = np.array([0, 0, 1, 1])
        y_prob = np.array([0.1, 0.2, 0.8, 0.9])
        eer = calculate_eer(y_true, y_prob)
        assert eer == 0.0
        
        # Random predictions
        y_true = np.array([0, 1, 0, 1])
        y_prob = np.array([0.5, 0.5, 0.5, 0.5])
        eer = calculate_eer(y_true, y_prob)
        assert 0 <= eer <= 1
    
    def test_calculate_confusion_matrix_metrics(self):
        """Test confusion matrix metrics."""
        y_true = np.array([0, 1, 0, 1, 0, 1])
        y_pred = np.array([0, 1, 0, 0, 1, 1])
        
        metrics = calculate_confusion_matrix_metrics(y_true, y_pred)
        
        assert 'true_negatives' in metrics
        assert 'false_positives' in metrics
        assert 'false_negatives' in metrics
        assert 'true_positives' in metrics
        assert 'sensitivity' in metrics
        assert 'specificity' in metrics
        assert 'precision' in metrics
        assert 'recall' in metrics
        
        # Check that metrics are reasonable
        assert 0 <= metrics['sensitivity'] <= 1
        assert 0 <= metrics['specificity'] <= 1
        assert 0 <= metrics['precision'] <= 1
        assert 0 <= metrics['recall'] <= 1
    
    def test_calculate_roc_metrics(self):
        """Test ROC metrics calculation."""
        y_true = np.array([0, 1, 0, 1, 0, 1])
        y_prob = np.array([0.1, 0.9, 0.2, 0.8, 0.3, 0.7])
        
        metrics = calculate_roc_metrics(y_true, y_prob)
        
        assert 'auc' in metrics
        assert 'optimal_threshold' in metrics
        assert 'optimal_tpr' in metrics
        assert 'optimal_fpr' in metrics
        
        # Check that metrics are reasonable
        assert 0 <= metrics['auc'] <= 1
        assert 0 <= metrics['optimal_threshold'] <= 1
        assert 0 <= metrics['optimal_tpr'] <= 1
        assert 0 <= metrics['optimal_fpr'] <= 1


class TestMetricsEdgeCases:
    """Test metrics with edge cases."""
    
    def test_all_same_predictions(self):
        """Test metrics when all predictions are the same."""
        y_true = np.array([0, 1, 0, 1])
        y_pred = np.array([0, 0, 0, 0])  # All predicted as class 0
        y_prob = np.array([0.8, 0.8, 0.8, 0.8])
        
        metrics = calculate_metrics(y_true, y_pred, y_prob)
        
        assert metrics['accuracy'] == 0.5  # 2 out of 4 correct
        assert metrics['precision'] == 0.5  # 1 out of 2 predicted positives
        assert metrics['recall'] == 0.0    # 0 out of 2 actual positives
    
    def test_perfect_predictions(self):
        """Test metrics with perfect predictions."""
        y_true = np.array([0, 1, 0, 1])
        y_pred = np.array([0, 1, 0, 1])
        y_prob = np.array([0.1, 0.9, 0.2, 0.8])
        
        metrics = calculate_metrics(y_true, y_pred, y_prob)
        
        assert metrics['accuracy'] == 1.0
        assert metrics['precision'] == 1.0
        assert metrics['recall'] == 1.0
        assert metrics['f1'] == 1.0
        assert metrics['acer'] == 0.0
        assert metrics['hter'] == 0.0
    
    def test_empty_predictions(self):
        """Test metrics with empty arrays."""
        y_true = np.array([])
        y_pred = np.array([])
        y_prob = np.array([])
        
        # Should handle empty arrays gracefully
        metrics = calculate_metrics(y_true, y_pred, y_prob)
        
        # All metrics should be 0 or NaN for empty arrays
        for key, value in metrics.items():
            assert np.isnan(value) or value == 0.0


@pytest.mark.parametrize("num_samples", [10, 100, 1000])
def test_metrics_with_different_sizes(num_samples):
    """Test metrics with different sample sizes."""
    np.random.seed(42)
    
    y_true = np.random.randint(0, 2, num_samples)
    y_pred = np.random.randint(0, 2, num_samples)
    y_prob = np.random.rand(num_samples)
    
    metrics = calculate_metrics(y_true, y_pred, y_prob)
    
    # All metrics should be calculated successfully
    for key, value in metrics.items():
        assert not np.isnan(value)
        assert 0 <= value <= 1 or key in ['total_parameters', 'trainable_parameters']


def test_metrics_with_torch_tensors():
    """Test metrics with PyTorch tensors."""
    y_true = torch.tensor([0, 1, 0, 1])
    y_pred = torch.tensor([0, 1, 0, 0])
    y_prob = torch.tensor([0.8, 0.9, 0.7, 0.3])
    
    metrics = calculate_metrics(y_true, y_pred, y_prob)
    
    # Should work with torch tensors
    assert 'accuracy' in metrics
    assert 0 <= metrics['accuracy'] <= 1
