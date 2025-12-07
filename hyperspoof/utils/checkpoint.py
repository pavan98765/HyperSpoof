"""
Checkpoint management utilities.
"""

import os
import torch
from typing import Dict, Any, Optional, Union
from pathlib import Path
import json


def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    loss: float,
    metrics: Dict[str, float],
    checkpoint_path: str,
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
    config: Optional[Dict[str, Any]] = None,
    is_best: bool = False,
) -> None:
    """
    Save model checkpoint.
    
    Args:
        model: PyTorch model
        optimizer: Optimizer
        epoch: Current epoch
        loss: Current loss
        metrics: Dictionary of metrics
        checkpoint_path: Path to save checkpoint
        scheduler: Optional learning rate scheduler
        config: Optional configuration dictionary
        is_best: Whether this is the best checkpoint
    """
    checkpoint_path = Path(checkpoint_path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Prepare checkpoint data
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
        'metrics': metrics,
        'is_best': is_best,
    }
    
    if scheduler is not None:
        checkpoint['scheduler_state_dict'] = scheduler.state_dict()
    
    if config is not None:
        checkpoint['config'] = config
    
    # Save checkpoint
    torch.save(checkpoint, checkpoint_path)
    
    # Save best checkpoint separately
    if is_best:
        best_path = checkpoint_path.parent / "best_model.pth"
        torch.save(checkpoint, best_path)


def load_checkpoint(
    checkpoint_path: str,
    model: Optional[torch.nn.Module] = None,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
    device: Optional[torch.device] = None,
) -> Dict[str, Any]:
    """
    Load model checkpoint.
    
    Args:
        checkpoint_path: Path to checkpoint file
        model: Optional model to load state dict into
        optimizer: Optional optimizer to load state dict into
        scheduler: Optional scheduler to load state dict into
        device: Device to load checkpoint on
        
    Returns:
        Dictionary containing checkpoint data
    """
    checkpoint_path = Path(checkpoint_path)
    
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    # Load checkpoint
    if device is not None:
        checkpoint = torch.load(checkpoint_path, map_location=device)
    else:
        checkpoint = torch.load(checkpoint_path)
    
    # Load model state dict
    if model is not None and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    
    # Load optimizer state dict
    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    # Load scheduler state dict
    if scheduler is not None and 'scheduler_state_dict' in checkpoint:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    
    return checkpoint


def get_latest_checkpoint(checkpoint_dir: str) -> Optional[str]:
    """
    Get the latest checkpoint file in a directory.
    
    Args:
        checkpoint_dir: Directory containing checkpoints
        
    Returns:
        Path to latest checkpoint or None if not found
    """
    checkpoint_dir = Path(checkpoint_dir)
    
    if not checkpoint_dir.exists():
        return None
    
    # Find all checkpoint files
    checkpoint_files = list(checkpoint_dir.glob("*.pth"))
    
    if not checkpoint_files:
        return None
    
    # Sort by modification time
    latest_checkpoint = max(checkpoint_files, key=os.path.getmtime)
    
    return str(latest_checkpoint)


def get_best_checkpoint(checkpoint_dir: str) -> Optional[str]:
    """
    Get the best checkpoint file in a directory.
    
    Args:
        checkpoint_dir: Directory containing checkpoints
        
    Returns:
        Path to best checkpoint or None if not found
    """
    checkpoint_dir = Path(checkpoint_dir)
    best_checkpoint = checkpoint_dir / "best_model.pth"
    
    if best_checkpoint.exists():
        return str(best_checkpoint)
    
    return None


def resume_training(
    checkpoint_path: str,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
    device: Optional[torch.device] = None,
) -> Dict[str, Any]:
    """
    Resume training from checkpoint.
    
    Args:
        checkpoint_path: Path to checkpoint file
        model: Model to resume
        optimizer: Optimizer to resume
        scheduler: Optional scheduler to resume
        device: Device to load checkpoint on
        
    Returns:
        Dictionary containing training state
    """
    checkpoint = load_checkpoint(
        checkpoint_path=checkpoint_path,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device
    )
    
    return {
        'epoch': checkpoint.get('epoch', 0),
        'loss': checkpoint.get('loss', float('inf')),
        'metrics': checkpoint.get('metrics', {}),
        'is_best': checkpoint.get('is_best', False),
    }


def save_model_only(
    model: torch.nn.Module,
    model_path: str,
    config: Optional[Dict[str, Any]] = None,
) -> None:
    """
    Save only the model (without optimizer, scheduler, etc.).
    
    Args:
        model: PyTorch model
        model_path: Path to save model
        config: Optional configuration dictionary
    """
    model_path = Path(model_path)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Prepare model data
    model_data = {
        'model_state_dict': model.state_dict(),
    }
    
    if config is not None:
        model_data['config'] = config
    
    # Save model
    torch.save(model_data, model_path)


def load_model_only(
    model_path: str,
    model: torch.nn.Module,
    device: Optional[torch.device] = None,
) -> Dict[str, Any]:
    """
    Load only the model (without optimizer, scheduler, etc.).
    
    Args:
        model_path: Path to model file
        model: Model to load state dict into
        device: Device to load model on
        
    Returns:
        Dictionary containing model data
    """
    model_path = Path(model_path)
    
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")
    
    # Load model
    if device is not None:
        model_data = torch.load(model_path, map_location=device)
    else:
        model_data = torch.load(model_path)
    
    # Load model state dict
    if 'model_state_dict' in model_data:
        model.load_state_dict(model_data['model_state_dict'])
    
    return model_data
