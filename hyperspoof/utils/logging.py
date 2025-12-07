"""
Logging utilities for HyperSpoof.
"""

import os
import logging
import wandb
from datetime import datetime
from typing import Optional, Dict, Any
from pathlib import Path


def setup_logging(
    log_dir: str = "./logs",
    log_level: str = "INFO",
    experiment_name: Optional[str] = None,
) -> logging.Logger:
    """
    Setup logging configuration.
    
    Args:
        log_dir: Directory to save logs
        log_level: Logging level
        experiment_name: Name of the experiment
        
    Returns:
        Configured logger
    """
    # Create log directory
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    
    # Create experiment-specific log file
    if experiment_name:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = log_dir / f"{experiment_name}_{timestamp}.log"
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = log_dir / f"hyperspoof_{timestamp}.log"
    
    # Configure logging
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    
    logger = logging.getLogger("hyperspoof")
    logger.info(f"Logging initialized. Log file: {log_file}")
    
    return logger


def get_logger(name: str = "hyperspoof") -> logging.Logger:
    """
    Get logger instance.
    
    Args:
        name: Logger name
        
    Returns:
        Logger instance
    """
    return logging.getLogger(name)


def setup_wandb(
    project_name: str = "hyperspoof",
    experiment_name: Optional[str] = None,
    config: Optional[Dict[str, Any]] = None,
    tags: Optional[list] = None,
) -> None:
    """
    Setup Weights & Biases logging.
    
    Args:
        project_name: W&B project name
        experiment_name: Experiment name
        config: Configuration dictionary
        tags: List of tags
    """
    if experiment_name is None:
        experiment_name = f"hyperspoof_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    wandb.init(
        project=project_name,
        name=experiment_name,
        config=config,
        tags=tags or [],
    )


def log_metrics(
    metrics: Dict[str, float],
    step: Optional[int] = None,
    prefix: str = "",
) -> None:
    """
    Log metrics to W&B.
    
    Args:
        metrics: Dictionary of metrics to log
        step: Step number
        prefix: Prefix for metric names
    """
    if wandb.run is not None:
        if prefix:
            metrics = {f"{prefix}/{k}": v for k, v in metrics.items()}
        
        if step is not None:
            wandb.log(metrics, step=step)
        else:
            wandb.log(metrics)


def log_model_info(model: torch.nn.Module, input_shape: tuple) -> None:
    """
    Log model information to W&B.
    
    Args:
        model: PyTorch model
        input_shape: Input tensor shape
    """
    if wandb.run is not None:
        # Log model architecture
        wandb.watch(model, log="all", log_freq=100)
        
        # Log model parameters
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        wandb.log({
            "model/total_parameters": total_params,
            "model/trainable_parameters": trainable_params,
            "model/input_shape": input_shape,
        })


def log_images(
    images: torch.Tensor,
    captions: Optional[list] = None,
    step: Optional[int] = None,
    prefix: str = "",
) -> None:
    """
    Log images to W&B.
    
    Args:
        images: Tensor of images (B, C, H, W)
        captions: List of captions for images
        step: Step number
        prefix: Prefix for image names
    """
    if wandb.run is not None:
        # Convert to numpy and denormalize if needed
        if images.dim() == 4:  # Batch of images
            images_np = images.detach().cpu().numpy()
        else:  # Single image
            images_np = images.detach().cpu().numpy().unsqueeze(0)
        
        # Convert to wandb format
        wandb_images = []
        for i, img in enumerate(images_np):
            # Transpose from (C, H, W) to (H, W, C)
            if img.shape[0] == 3:  # RGB
                img = img.transpose(1, 2, 0)
            
            # Normalize to [0, 1] if needed
            if img.max() > 1.0:
                img = img / 255.0
            
            wandb_img = wandb.Image(img, caption=captions[i] if captions else None)
            wandb_images.append(wandb_img)
        
        log_dict = {f"{prefix}/images": wandb_images}
        if step is not None:
            wandb.log(log_dict, step=step)
        else:
            wandb.log(log_dict)


def finish_wandb() -> None:
    """
    Finish W&B run.
    """
    if wandb.run is not None:
        wandb.finish()
