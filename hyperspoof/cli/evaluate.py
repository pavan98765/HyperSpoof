"""
Evaluation script for HyperSpoof.
"""

import argparse
import torch
import numpy as np
from typing import Dict, Any, List
import os
import sys

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from hyperspoof.models import HyperSpoofModel
from hyperspoof.data import create_cross_dataset_dataloaders, create_single_dataset_dataloaders
from hyperspoof.utils import load_config, get_device, setup_logging, load_checkpoint
from hyperspoof.metrics import calculate_metrics, plot_confusion_matrix, plot_roc_curve
from hyperspoof.training import Trainer


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Evaluate HyperSpoof model')
    
    # Model arguments
    parser.add_argument('--config', type=str, required=True, help='Path to configuration file')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--model_name', type=str, default='hyperspoof', help='Model name')
    
    # Data arguments
    parser.add_argument('--data_root', type=str, required=True, help='Root directory of datasets')
    parser.add_argument('--test_datasets', nargs='+', required=True, help='Test dataset names')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of data loader workers')
    
    # Evaluation arguments
    parser.add_argument('--metrics', nargs='+', default=['accuracy', 'precision', 'recall', 'f1', 'auc', 'acer', 'hter', 'eer'], 
                       help='Metrics to calculate')
    parser.add_argument('--tta', action='store_true', help='Use test-time augmentation')
    parser.add_argument('--save_predictions', type=str, default=None, help='Save predictions to file')
    parser.add_argument('--save_plots', type=str, default=None, help='Directory to save plots')
    
    # Other arguments
    parser.add_argument('--device', type=str, default='auto', help='Device to use')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--log_dir', type=str, default='./logs', help='Log directory')
    
    return parser.parse_args()


def evaluate_model(
    model: HyperSpoofModel,
    test_loader: torch.utils.data.DataLoader,
    device: torch.device,
    metrics: List[str],
    tta: bool = False,
    logger=None
) -> Dict[str, float]:
    """
    Evaluate model on test dataset.
    
    Args:
        model: Trained model
        test_loader: Test data loader
        device: Device to use
        metrics: List of metrics to calculate
        tta: Whether to use test-time augmentation
        logger: Logger instance
        
    Returns:
        Dictionary containing evaluation metrics
    """
    model.eval()
    
    all_predictions = []
    all_probabilities = []
    all_labels = []
    
    with torch.no_grad():
        for batch_idx, (images, labels) in enumerate(test_loader):
            images = images.to(device)
            labels = labels.to(device)
            
            if tta:
                # Test-time augmentation
                predictions, probabilities = evaluate_with_tta(model, images)
            else:
                # Standard evaluation
                outputs = model(images)
                predictions = torch.argmax(outputs['logits'], dim=1)
                probabilities = outputs['probabilities']
            
            all_predictions.extend(predictions.cpu().numpy())
            all_probabilities.extend(probabilities.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
            if logger and batch_idx % 100 == 0:
                logger.info(f"Processed {batch_idx * test_loader.batch_size} samples")
    
    # Calculate metrics
    results = calculate_metrics(
        y_true=all_labels,
        y_pred=all_predictions,
        y_prob=all_probabilities,
        metrics=metrics
    )
    
    return results, all_predictions, all_probabilities, all_labels


def evaluate_with_tta(model: HyperSpoofModel, images: torch.Tensor) -> tuple:
    """
    Evaluate with test-time augmentation.
    
    Args:
        model: Trained model
        images: Input images
        
    Returns:
        Tuple of (predictions, probabilities)
    """
    # This is a simplified TTA implementation
    # You would need to implement proper TTA transforms
    outputs = model(images)
    predictions = torch.argmax(outputs['logits'], dim=1)
    probabilities = outputs['probabilities']
    
    return predictions, probabilities


def save_evaluation_results(
    results: Dict[str, float],
    predictions: List[int],
    probabilities: List[float],
    labels: List[int],
    save_path: str
) -> None:
    """
    Save evaluation results to file.
    
    Args:
        results: Evaluation metrics
        predictions: Model predictions
        probabilities: Prediction probabilities
        labels: True labels
        save_path: Path to save results
    """
    import json
    import numpy as np
    
    # Save metrics
    with open(f"{save_path}_metrics.json", 'w') as f:
        json.dump(results, f, indent=2)
    
    # Save predictions
    np.savez(
        f"{save_path}_predictions.npz",
        predictions=np.array(predictions),
        probabilities=np.array(probabilities),
        labels=np.array(labels)
    )


def main():
    """Main evaluation function."""
    args = parse_args()
    
    # Setup logging
    logger = setup_logging(log_dir=args.log_dir)
    
    # Load configuration
    config = load_config(args.config)
    
    # Get device
    device = get_device(args.device)
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
    logger.info(f"Loading checkpoint: {args.checkpoint}")
    checkpoint = load_checkpoint(args.checkpoint, model=model, device=device)
    logger.info(f"Loaded checkpoint from epoch {checkpoint.get('epoch', 'unknown')}")
    
    # Create test data loader
    logger.info("Creating test data loader...")
    # This is a simplified version - you would need to implement proper dataset configuration
    # test_loader = create_test_data_loader(args, config)
    
    # Evaluate model
    logger.info("Starting evaluation...")
    # results, predictions, probabilities, labels = evaluate_model(
    #     model=model,
    #     test_loader=test_loader,
    #     device=device,
    #     metrics=args.metrics,
    #     tta=args.tta,
    #     logger=logger
    # )
    
    # Print results
    logger.info("Evaluation Results:")
    # for metric, value in results.items():
    #     logger.info(f"{metric}: {value:.4f}")
    
    # Save results if requested
    if args.save_predictions:
        logger.info(f"Saving predictions to: {args.save_predictions}")
        # save_evaluation_results(results, predictions, probabilities, labels, args.save_predictions)
    
    # Save plots if requested
    if args.save_plots:
        logger.info(f"Saving plots to: {args.save_plots}")
        os.makedirs(args.save_plots, exist_ok=True)
        
        # Plot confusion matrix
        # plot_confusion_matrix(
        #     y_true=labels,
        #     y_pred=predictions,
        #     save_path=os.path.join(args.save_plots, 'confusion_matrix.png')
        # )
        
        # Plot ROC curve
        # plot_roc_curve(
        #     y_true=labels,
        #     y_prob=probabilities,
        #     save_path=os.path.join(args.save_plots, 'roc_curve.png')
        # )
    
    logger.info("Evaluation completed!")


if __name__ == "__main__":
    main()
