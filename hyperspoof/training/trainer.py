"""
Training class for HyperSpoof model.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from typing import Dict, Any, Optional, List, Tuple
import time
import os
from tqdm import tqdm

from ..utils import save_checkpoint, load_checkpoint
from ..metrics import calculate_metrics
from .early_stopping import EarlyStopping


class Trainer:
    """
    Trainer class for HyperSpoof model.
    """
    
    def __init__(
        self,
        model: nn.Module,
        optimizer: optim.Optimizer,
        scheduler: Optional[optim.lr_scheduler._LRScheduler],
        loss_fn: nn.Module,
        device: torch.device,
        config: Dict[str, Any],
        logger=None,
    ):
        """
        Initialize trainer.
        
        Args:
            model: PyTorch model
            optimizer: Optimizer
            scheduler: Learning rate scheduler
            loss_fn: Loss function
            device: Device to use
            config: Configuration dictionary
            logger: Logger instance
        """
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.loss_fn = loss_fn
        self.device = device
        self.config = config
        self.logger = logger
        
        # Training state
        self.current_epoch = 0
        self.best_metric = float('inf')
        self.train_losses = []
        self.val_losses = []
        self.train_metrics = {}
        self.val_metrics = {}
        
        # Early stopping
        self.early_stopping = EarlyStopping(
            patience=config['training'].get('early_stopping_patience', 10),
            min_delta=config['training'].get('early_stopping_min_delta', 0.001),
            restore_best_weights=True
        )
        
        # Setup logging
        if self.logger is None:
            import logging
            self.logger = logging.getLogger("hyperspoof")
    
    def train_epoch(self, train_loader: DataLoader) -> Tuple[float, Dict[str, float]]:
        """
        Train for one epoch.
        
        Args:
            train_loader: Training data loader
            
        Returns:
            Tuple of (average_loss, metrics)
        """
        self.model.train()
        total_loss = 0.0
        all_predictions = []
        all_labels = []
        all_probabilities = []
        
        progress_bar = tqdm(train_loader, desc=f"Epoch {self.current_epoch + 1}")
        
        for batch_idx, (images, labels) in enumerate(progress_bar):
            images = images.to(self.device)
            labels = labels.to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(images)
            loss = self.loss_fn(outputs['logits'], labels)
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            if self.config['training'].get('gradient_clip', 0) > 0:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config['training']['gradient_clip']
                )
            
            self.optimizer.step()
            
            # Statistics
            total_loss += loss.item()
            
            # Collect predictions for metrics
            predictions = torch.argmax(outputs['logits'], dim=1)
            probabilities = outputs['probabilities']
            
            all_predictions.extend(predictions.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probabilities.extend(probabilities.cpu().numpy())
            
            # Update progress bar
            progress_bar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'avg_loss': f'{total_loss / (batch_idx + 1):.4f}'
            })
        
        # Calculate metrics
        metrics = calculate_metrics(
            y_true=all_labels,
            y_pred=all_predictions,
            y_prob=all_probabilities,
            metrics=['accuracy', 'precision', 'recall', 'f1']
        )
        
        avg_loss = total_loss / len(train_loader)
        
        return avg_loss, metrics
    
    def validate_epoch(self, val_loader: DataLoader) -> Tuple[float, Dict[str, float]]:
        """
        Validate for one epoch.
        
        Args:
            val_loader: Validation data loader
            
        Returns:
            Tuple of (average_loss, metrics)
        """
        self.model.eval()
        total_loss = 0.0
        all_predictions = []
        all_labels = []
        all_probabilities = []
        
        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(self.device)
                labels = labels.to(self.device)
                
                # Forward pass
                outputs = self.model(images)
                loss = self.loss_fn(outputs['logits'], labels)
                
                # Statistics
                total_loss += loss.item()
                
                # Collect predictions for metrics
                predictions = torch.argmax(outputs['logits'], dim=1)
                probabilities = outputs['probabilities']
                
                all_predictions.extend(predictions.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                all_probabilities.extend(probabilities.cpu().numpy())
        
        # Calculate metrics
        metrics = calculate_metrics(
            y_true=all_labels,
            y_pred=all_predictions,
            y_prob=all_probabilities,
            metrics=['accuracy', 'precision', 'recall', 'f1', 'auc', 'acer', 'hter', 'eer']
        )
        
        avg_loss = total_loss / len(val_loader)
        
        return avg_loss, metrics
    
    def train(
        self,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        num_epochs: Optional[int] = None,
        save_dir: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Train the model.
        
        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            num_epochs: Number of epochs to train
            save_dir: Directory to save checkpoints
            
        Returns:
            Training history
        """
        if num_epochs is None:
            num_epochs = self.config['training']['epochs']
        
        if save_dir is None:
            save_dir = self.config['logging']['save_dir']
        
        os.makedirs(save_dir, exist_ok=True)
        
        self.logger.info(f"Starting training for {num_epochs} epochs")
        self.logger.info(f"Model has {sum(p.numel() for p in self.model.parameters())} parameters")
        
        start_time = time.time()
        
        for epoch in range(num_epochs):
            self.current_epoch = epoch
            
            # Training
            train_loss, train_metrics = self.train_epoch(train_loader)
            self.train_losses.append(train_loss)
            
            # Update training metrics
            for key, value in train_metrics.items():
                if key not in self.train_metrics:
                    self.train_metrics[key] = []
                self.train_metrics[key].append(value)
            
            # Validation
            val_loss = 0.0
            val_metrics = {}
            if val_loader is not None:
                val_loss, val_metrics = self.validate_epoch(val_loader)
                self.val_losses.append(val_loss)
                
                # Update validation metrics
                for key, value in val_metrics.items():
                    if key not in self.val_metrics:
                        self.val_metrics[key] = []
                    self.val_metrics[key].append(value)
            
            # Learning rate scheduling
            if self.scheduler is not None:
                if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_loss if val_loader is not None else train_loss)
                else:
                    self.scheduler.step()
            
            # Logging
            self.logger.info(f"Epoch {epoch + 1}/{num_epochs}")
            self.logger.info(f"Train Loss: {train_loss:.4f}")
            for key, value in train_metrics.items():
                self.logger.info(f"Train {key}: {value:.4f}")
            
            if val_loader is not None:
                self.logger.info(f"Val Loss: {val_loss:.4f}")
                for key, value in val_metrics.items():
                    self.logger.info(f"Val {key}: {value:.4f}")
            
            # Save checkpoint
            is_best = False
            if val_loader is not None:
                # Use validation loss for best model selection
                if val_loss < self.best_metric:
                    self.best_metric = val_loss
                    is_best = True
            else:
                # Use training loss if no validation
                if train_loss < self.best_metric:
                    self.best_metric = train_loss
                    is_best = True
            
            # Save checkpoint
            checkpoint_path = os.path.join(save_dir, f"checkpoint_epoch_{epoch + 1}.pth")
            save_checkpoint(
                model=self.model,
                optimizer=self.optimizer,
                epoch=epoch + 1,
                loss=val_loss if val_loader is not None else train_loss,
                metrics=val_metrics if val_loader is not None else train_metrics,
                checkpoint_path=checkpoint_path,
                scheduler=self.scheduler,
                config=self.config,
                is_best=is_best
            )
            
            # Early stopping
            if val_loader is not None:
                self.early_stopping(val_loss, self.model)
                if self.early_stopping.early_stop:
                    self.logger.info(f"Early stopping at epoch {epoch + 1}")
                    break
        
        training_time = time.time() - start_time
        self.logger.info(f"Training completed in {training_time:.2f} seconds")
        
        # Return training history
        history = {
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'train_metrics': self.train_metrics,
            'val_metrics': self.val_metrics,
            'training_time': training_time,
            'best_metric': self.best_metric,
        }
        
        return history
    
    def load_checkpoint(self, checkpoint_path: str) -> Dict[str, Any]:
        """
        Load checkpoint and resume training.
        
        Args:
            checkpoint_path: Path to checkpoint file
            
        Returns:
            Checkpoint data
        """
        checkpoint = load_checkpoint(
            checkpoint_path=checkpoint_path,
            model=self.model,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            device=self.device
        )
        
        self.current_epoch = checkpoint.get('epoch', 0)
        self.best_metric = checkpoint.get('loss', float('inf'))
        
        self.logger.info(f"Loaded checkpoint from epoch {self.current_epoch}")
        
        return checkpoint
    
    def evaluate(
        self,
        test_loader: DataLoader,
        metrics: Optional[List[str]] = None,
    ) -> Dict[str, float]:
        """
        Evaluate the model on test data.
        
        Args:
            test_loader: Test data loader
            metrics: List of metrics to calculate
            
        Returns:
            Evaluation metrics
        """
        if metrics is None:
            metrics = ['accuracy', 'precision', 'recall', 'f1', 'auc', 'acer', 'hter', 'eer']
        
        self.logger.info("Starting evaluation...")
        
        test_loss, test_metrics = self.validate_epoch(test_loader)
        
        self.logger.info("Evaluation Results:")
        for key, value in test_metrics.items():
            self.logger.info(f"{key}: {value:.4f}")
        
        return test_metrics
