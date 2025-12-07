"""
Early stopping implementation for training.
"""

import torch
import numpy as np
from typing import Optional


class EarlyStopping:
    """
    Early stopping utility to stop training when validation loss stops improving.
    """
    
    def __init__(
        self,
        patience: int = 7,
        min_delta: float = 0.0,
        restore_best_weights: bool = True,
        mode: str = 'min'
    ):
        """
        Initialize early stopping.
        
        Args:
            patience: Number of epochs to wait before stopping
            min_delta: Minimum change to qualify as an improvement
            restore_best_weights: Whether to restore best weights when stopping
            mode: 'min' for loss, 'max' for accuracy
        """
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        self.mode = mode
        
        self.wait = 0
        self.stopped_epoch = 0
        self.best_score = None
        self.best_weights = None
        self.early_stop = False
        
        if mode == 'min':
            self.monitor_op = np.less
            self.min_delta *= -1
        else:
            self.monitor_op = np.greater
    
    def __call__(self, current_score: float, model: torch.nn.Module) -> None:
        """
        Check if training should stop.
        
        Args:
            current_score: Current validation score
            model: PyTorch model
        """
        if self.best_score is None:
            self.best_score = current_score
            if self.restore_best_weights:
                self.best_weights = model.state_dict().copy()
        elif self.monitor_op(current_score, self.best_score + self.min_delta):
            self.best_score = current_score
            self.wait = 0
            if self.restore_best_weights:
                self.best_weights = model.state_dict().copy()
        else:
            self.wait += 1
            if self.wait >= self.patience:
                self.stopped_epoch = self.wait
                self.early_stop = True
                if self.restore_best_weights and self.best_weights is not None:
                    model.load_state_dict(self.best_weights)
    
    def reset(self) -> None:
        """Reset early stopping state."""
        self.wait = 0
        self.stopped_epoch = 0
        self.best_score = None
        self.best_weights = None
        self.early_stop = False
