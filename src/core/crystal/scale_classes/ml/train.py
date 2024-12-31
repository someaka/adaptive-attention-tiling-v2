"""Training script for the holographic neural network.

This script handles:
1. Model initialization with configurable parameters
2. Training setup and execution
3. Checkpoint and metrics management
4. Logging and progress tracking
"""

import torch
import os
from pathlib import Path
import logging
import json
from typing import Dict, Any, Optional, Tuple

from .models import HolographicNet
from .trainer import HolographicTrainer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def load_config(config_path: str = "config/holographic_net.json") -> Dict[str, Any]:
    """Load training configuration from JSON file or return defaults.
    
    The default configuration is optimized for:
    - Fast training cycles (~10s)
    - Efficient memory usage
    - Quick convergence
    """
    defaults = {
        "model": {
            "dim": 4,  # Small dimension for quick training
            "hidden_dim": 64,  # Increased for better capacity
            "n_layers": 3,  # Added one more layer
            "dtype": "complex64"  # Single precision complex
        },
        "training": {
            "n_epochs": 50,  # More epochs for better convergence
            "batch_size": 32,  # Smaller batches for stability
            "val_split": 0.2,
            "lr": 1e-4,  # Reduced learning rate for stability
            "patience": 10,  # More patience for convergence
            "min_delta": 1e-3,  # Relaxed convergence criterion
            "grad_accum_steps": 4  # More gradient accumulation
        }
    }
    
    try:
        with open(config_path) as f:
            config = json.load(f)
            logger.info(f"Loaded configuration from {config_path}")
            return config
    except FileNotFoundError:
        logger.warning(f"Config file {config_path} not found, using defaults")
        return defaults

def setup_directories() -> Path:
    """Create necessary directories for checkpoints and logs."""
    checkpoint_dir = Path("checkpoints/holographic_net")
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Created checkpoint directory at {checkpoint_dir}")
    return checkpoint_dir

def find_latest_checkpoint(checkpoint_dir: Path) -> Optional[Tuple[Path, int]]:
    """Find the latest checkpoint in the directory.
    
    Returns:
        Tuple of (checkpoint path, epoch number) or None if no checkpoints exist
    """
    checkpoints = list(checkpoint_dir.glob("checkpoint_epoch_*.pt"))
    if not checkpoints:
        return None
        
    # Extract epoch numbers and find latest
    epoch_nums = []
    for cp in checkpoints:
        try:
            epoch = int(cp.stem.split('_')[-1])
            epoch_nums.append((epoch, cp))
        except ValueError:
            continue
            
    if not epoch_nums:
        return None
        
    latest_epoch, latest_checkpoint = max(epoch_nums, key=lambda x: x[0])
    return latest_checkpoint, latest_epoch

def main() -> None:
    """Main training function."""
    # Load configuration and setup directories
    config = load_config()
    checkpoint_dir = setup_directories()
    
    # Save configuration
    with open(checkpoint_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)
    
    # Initialize model with configuration
    logger.info("Initializing model...")
    model = HolographicNet(
        dim=config["model"]["dim"],
        hidden_dim=config["model"]["hidden_dim"],
        n_layers=config["model"]["n_layers"],
        dtype=getattr(torch, config["model"]["dtype"])
    )
    
    # Initialize trainer
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    trainer = HolographicTrainer(
        model=model,
        save_dir=str(checkpoint_dir),
        device=device
    )
    
    # Look for existing checkpoints
    latest_checkpoint = find_latest_checkpoint(checkpoint_dir)
    start_epoch = 0
    
    if latest_checkpoint:
        checkpoint_path, epoch = latest_checkpoint
        logger.info(f"Found checkpoint at epoch {epoch}, resuming training...")
        try:
            start_epoch = trainer.load_checkpoint(checkpoint_path)
            logger.info(f"Successfully loaded checkpoint from epoch {start_epoch}")
        except Exception as e:
            logger.warning(f"Failed to load checkpoint: {e}")
            logger.info("Starting training from scratch...")
    else:
        logger.info("No checkpoints found, starting training from scratch...")
    
    # Adjust remaining epochs
    remaining_epochs = max(0, config["training"]["n_epochs"] - start_epoch)
    if remaining_epochs == 0:
        logger.info("Training already completed, no more epochs to train.")
        return
    
    # Train model
    logger.info(f"Starting training for {remaining_epochs} epochs...")
    metrics = trainer.train(
        n_epochs=remaining_epochs,
        batch_size=config["training"]["batch_size"],
        val_split=config["training"]["val_split"],
        lr=config["training"]["lr"],
        patience=config["training"]["patience"],
        min_delta=config["training"]["min_delta"]
    )
    
    # Save final metrics
    metrics_file = checkpoint_dir / "training_metrics.pt"
    torch.save(metrics, metrics_file)
    logger.info(f"Training complete! Metrics saved to {metrics_file}")
    
    # Save final model state
    final_model_path = checkpoint_dir / "final_model.pt"
    torch.save({
        "model_state": model.state_dict(),
        "config": config,
        "final_metrics": metrics,
        "total_epochs": start_epoch + remaining_epochs
    }, final_model_path)
    logger.info(f"Final model saved to {final_model_path}")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.exception("Training failed with error:") 