"""
Example script for evaluating HyperSpoof model.
"""

import os
import sys
import torch
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from hyperspoof.models import HyperSpoofModel
from hyperspoof.utils import load_config, get_device, setup_logging, load_checkpoint
from hyperspoof.metrics import calculate_metrics, plot_confusion_matrix, plot_roc_curve


def main():
    """Main evaluation example."""
    
    # Setup logging
    logger = setup_logging(experiment_name="hyperspoof_evaluation")
    
    # Load configuration
    config_path = "configs/default.yaml"
    config = load_config(config_path)
    
    # Get device
    device = get_device(config['device'])
    logger.info(f"Using device: {device}")
    
    # Create model
    model = HyperSpoofModel(
        input_channels=config['model']['input_channels'],
        hsi_channels=config['model']['hsi_channels'],
        attention_channels=config['model']['attention_channels'],
        num_classes=config['model']['num_classes'],
        model_name=config['model']['classifier'],
        pretrained=config['model']['pretrained'],
        dropout_rate=config['model']['dropout_rate'],
    )
    
    model = model.to(device)
    
    # Load checkpoint
    checkpoint_path = "checkpoints/best_model.pth"
    logger.info(f"Loading checkpoint: {checkpoint_path}")
    checkpoint = load_checkpoint(checkpoint_path, model=model, device=device)
    logger.info(f"Loaded checkpoint from epoch {checkpoint.get('epoch', 'unknown')}")
    
    # Example evaluation (you would need to create actual test data loader)
    logger.info("Starting evaluation...")
    
    # Create dummy data for demonstration
    batch_size = 32
    dummy_images = torch.randn(batch_size, 3, 256, 256).to(device)
    dummy_labels = torch.randint(0, 2, (batch_size,)).to(device)
    
    # Make predictions
    model.eval()
    with torch.no_grad():
        outputs = model(dummy_images)
        predictions = torch.argmax(outputs['logits'], dim=1)
        probabilities = outputs['probabilities']
    
    # Calculate metrics
    metrics = calculate_metrics(
        y_true=dummy_labels.cpu().numpy(),
        y_pred=predictions.cpu().numpy(),
        y_prob=probabilities.cpu().numpy(),
        metrics=['accuracy', 'precision', 'recall', 'f1', 'auc', 'acer', 'hter', 'eer']
    )
    
    # Print results
    logger.info("Evaluation Results:")
    for metric, value in metrics.items():
        logger.info(f"{metric}: {value:.4f}")
    
    # Create visualizations
    logger.info("Creating visualizations...")
    
    # Plot confusion matrix
    plot_confusion_matrix(
        y_true=dummy_labels.cpu().numpy(),
        y_pred=predictions.cpu().numpy(),
        save_path="results/confusion_matrix.png"
    )
    
    # Plot ROC curve
    plot_roc_curve(
        y_true=dummy_labels.cpu().numpy(),
        y_prob=probabilities.cpu().numpy(),
        save_path="results/roc_curve.png"
    )
    
    logger.info("Evaluation completed!")


if __name__ == "__main__":
    main()
