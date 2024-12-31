"""Trainer module for holographic neural networks.

This module provides the HolographicTrainer class which handles:
1. Training data generation with physics-informed examples
2. Loss computation using conformal structure
3. Training loop with validation and early stopping
4. Model checkpointing and metrics logging

The trainer implements a physics-informed training approach that preserves:
- Conformal structure
- Quantum corrections
- Phase coherence
- Scale invariance
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from torch.optim.lr_scheduler import ReduceLROnPlateau
from typing import Optional, Dict, List, Tuple, Union, Any, cast, TypeVar, Iterable, Callable
import numpy as np
from pathlib import Path
import json
import logging
from datetime import datetime
from collections import defaultdict

from .models import HolographicNet

# Configure logging
logger = logging.getLogger(__name__)

# Type variables for improved type hints
T = TypeVar('T', bound=torch.Tensor)
MetricsDict = Dict[str, float]
LossComponents = Dict[str, float]

class HolographicTrainer:
    """Trainer for holographic neural networks.
    
    This class implements a physics-informed training approach that preserves
    the essential properties of holographic systems while training neural networks
    to approximate the UV/IR mapping.
    
    Attributes:
        model: The holographic neural network model
        save_dir: Directory for saving checkpoints and metrics
        device: Device to use for training (CPU/GPU)
        inverse_model: Inverse model for cycle consistency
        metrics: Dictionary tracking training metrics
    """
    
    def __init__(
        self,
        model: HolographicNet,
        save_dir: str = "pretrained",
        device: Optional[torch.device] = None
    ):
        """Initialize the trainer.
        
        Args:
            model: The holographic neural network model
            save_dir: Directory for saving checkpoints and metrics
            device: Device to use for training (CPU/GPU)
        """
        self.model = model
        self.save_dir = Path(save_dir)
        self.device = device or torch.device("cpu")
        self.model.to(self.device)
        
        # Create save directory and initialize inverse model
        self.save_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Using device: {self.device}")
        logger.info(f"Saving checkpoints to: {self.save_dir}")
        
        # Initialize inverse model with doubled hidden dimension
        hidden_dim = model.output_proj.in_features
        self.inverse_model = HolographicNet(
            dim=model.dim,
            hidden_dim=hidden_dim * 2,
            n_layers=len(model.blocks),
            z_uv=model.z_ir,
            z_ir=model.z_uv
        ).to(self.device)
        
        # Initialize metrics tracking
        self.metrics: Dict[str, List[float]] = {
            "train_loss": [], "val_loss": [],
            "quantum_error": [], "scaling_error": [],
            "phase_error": []
        }
    
    def generate_training_data(
        self,
        batch_size: int,
        include_special_cases: bool = True
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Generate physics-informed training data efficiently.
        
        Optimized for:
        - Fast data generation using vectorized operations
        - Memory efficiency with smaller dataset size
        - Reproducible results with fixed seed
        - Balanced mix of random and special cases
        
        Args:
            batch_size: Number of examples to generate
            include_special_cases: Whether to include special physical states
            
        Returns:
            Tuple of (UV data, IR data) tensors
        """
        # Use smaller total dataset size for quick training
        total_samples = batch_size * 10  # Reduced from 100
        
        # Use deterministic generation for reproducibility
        generator = torch.Generator(device=self.device).manual_seed(42)
        
        # Vectorized random UV state generation
        uv_data = torch.randn(total_samples, self.model.dim, 
                            dtype=self.model.dtype, device=self.device,
                            generator=generator)
        uv_data = uv_data / torch.norm(uv_data, dim=1, keepdim=True)
        
        if include_special_cases:
            n_special = total_samples // 8  # Reduced proportion of special cases
            
            # Generate special cases in parallel
            # Symmetric states
            symmetric = torch.randn(n_special, self.model.dim // 2, 
                                dtype=self.model.dtype, device=self.device,
                                generator=generator)
            symmetric = torch.cat([symmetric, symmetric.conj()], dim=1) / np.sqrt(2)
            
            # Localized states using efficient scatter
            localized = torch.zeros(n_special, self.model.dim, 
                                dtype=self.model.dtype, device=self.device)
            indices = torch.randint(0, self.model.dim, (n_special, 1), 
                                device=self.device, generator=generator)
            localized.scatter_(1, indices, torch.ones_like(indices, dtype=self.model.dtype))
            
            # Combine special cases efficiently
            special_cases = torch.cat([symmetric, localized], dim=0)
            uv_data[:len(special_cases)] = special_cases
        
        # Vectorized IR computation with quantum corrections
        z_ratio = self.model.z_ratio
        powers = torch.tensor([-self.model.dim] + 
                            [-self.model.dim + 2*n for n in range(1, 4)],
                            device=self.device)
        coeffs = torch.tensor([1.0] + 
                            [(-0.1 / (1 + z_ratio**2)) * (-1)**n / n for n in range(1, 4)],
                            device=self.device)
        
        # Compute all corrections at once using efficient broadcasting
        z_powers = z_ratio**powers[:, None, None]
        ir_data = (uv_data[None, :, :] * z_powers * coeffs[:, None, None]).sum(dim=0)
        
        return uv_data, ir_data
    
    def compute_loss(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        input: torch.Tensor,
        return_components: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, LossComponents]]:
        """Compute physics-informed loss function.
        
        The loss combines:
        1. Basic holographic error with conformal weight
        2. Quantum corrections with proper scaling
        3. Phase coherence with conformal coupling
        
        Args:
            pred: Predicted IR state
            target: Target IR state
            input: Input UV state
            return_components: Whether to return individual loss components
            
        Returns:
            Total loss tensor or tuple of (total loss, components dict)
        """
        # Get the natural scale from z_ratio
        z_ratio = self.model.z_ratio
        N = z_ratio**self.model.dim
        
        # Convert N to tensor for operations
        N_tensor = torch.tensor(N, device=self.device, dtype=pred.dtype)
        
        # Normalize inputs for stability
        pred_norm = pred / (torch.norm(pred, dim=1, keepdim=True) + 1e-8)
        target_norm = target / (torch.norm(target, dim=1, keepdim=True) + 1e-8)
        input_norm = input / (torch.norm(input, dim=1, keepdim=True) + 1e-8)
        
        # Holographic error with conformal weight and stability
        holographic_error = torch.norm(N_tensor*pred_norm - input_norm, dim=1)
        # Take real part before using maximum for complex tensors
        conformal_factor = torch.maximum(
            (N_tensor**2 + 1).real,
            torch.tensor(1.0, device=self.device, dtype=torch.float32)
        )
        basic_loss = torch.mean(holographic_error**2) / conformal_factor
        
        # Quantum corrections with proper scaling and stability
        quantum_pred = self.model.compute_quantum_corrections(input_norm)
        quantum_target = target_norm - input_norm/N_tensor  # Remove classical scaling
        quantum_diff = quantum_pred - quantum_target
        quantum_error = torch.norm(quantum_diff, dim=1)**2
        # Take real part and convert to float for scaling
        N_real = torch.maximum(
            N_tensor.real,
            torch.tensor(1.0, device=self.device, dtype=torch.float32)
        )
        quantum_loss = torch.mean(quantum_error) * N_real
        
        # Phase coherence with conformal coupling and stability
        phase_diff = torch.angle(pred_norm + 1e-8) - torch.angle(input_norm/N_tensor + 1e-8)
        phase_loss = torch.mean(torch.abs(input_norm)**2 * (1 - torch.cos(phase_diff))) / conformal_factor
        
        # Total loss combines conformal terms with balanced weights
        total_loss = (0.5 * basic_loss + 0.3 * quantum_loss + 0.2 * phase_loss).real
        
        if return_components:
            components: LossComponents = {
                "basic": float(basic_loss.real),
                "quantum": float(quantum_loss.real),
                "phase": float(phase_loss.real)
            }
            return total_loss, components
        return total_loss
    
    def train(
        self,
        n_epochs: int = 20,
        batch_size: int = 64,
        val_split: float = 0.2,
        lr: float = 1e-3,
        patience: int = 5,
        min_delta: float = 1e-4,
        grad_accum_steps: int = 2
    ) -> Dict[str, List[float]]:
        """Train the holographic network with optimized performance.
        
        Features:
        - Gradient accumulation for larger effective batch size
        - Memory-efficient data loading
        - Early convergence detection
        - Progress tracking
        """
        # Import tqdm with proper typing
        from typing import TypeVar, Iterable, Any, Callable, cast
        T = TypeVar('T')
        
        try:
            from tqdm.auto import tqdm
            range_iter = tqdm(range(n_epochs), desc="Training")
        except ImportError:
            range_iter = range(n_epochs)
        
        # Initialize metrics tracking
        self.metrics = {
            "train_loss": [], "val_loss": [],
            "quantum_error": [], "scaling_error": [],
            "phase_error": [], "grad_norm": []
        }
        
        # Setup data loaders with memory pinning
        train_loader, val_loader = self._get_data_loaders(
            batch_size, val_split, num_workers=2
        )
        
        # Setup optimizer and scheduler
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        scheduler = ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5,
            patience=patience//2, verbose='True'
        )
        
        # Training loop
        best_val_loss = float('inf')
        best_epoch = 0
        steps_without_improvement = 0
        
        for epoch in range_iter:
            # Training phase
            self.model.train()
            train_metrics = self._train_epoch(
                train_loader, optimizer,
                grad_accum_steps=grad_accum_steps
            )
            
            # Validation phase
            self.model.eval()
            with torch.no_grad():
                val_metrics = self._validate_epoch(val_loader)
            
            # Update metrics
            self.metrics["train_loss"].append(train_metrics["loss"])
            self.metrics["val_loss"].append(val_metrics["loss"])
            self.metrics["quantum_error"].append(val_metrics["quantum"])
            self.metrics["scaling_error"].append(val_metrics["scaling"])
            self.metrics["phase_error"].append(val_metrics["phase"])
            self.metrics["grad_norm"].append(train_metrics.get("grad_norm", 0.0))
            
            # Update learning rate
            scheduler.step(val_metrics["loss"])
            
            # Save best model and check for improvement
            if val_metrics["loss"] < best_val_loss - min_delta:
                best_val_loss = val_metrics["loss"]
                best_epoch = epoch
                steps_without_improvement = 0
                self._save_checkpoint(epoch, val_metrics)
            else:
                steps_without_improvement += 1
            
            # Early stopping with fast convergence detection
            if steps_without_improvement >= patience:
                logger.info(f"Early stopping at epoch {epoch}")
                break
            
            # Check for quick convergence
            if val_metrics["loss"] < min_delta and epoch >= 5:
                logger.info(f"Converged at epoch {epoch}")
                break
            
            # Log progress
            if epoch % max(1, n_epochs // 10) == 0:
                self._log_progress(epoch, train_metrics, val_metrics)
        
        return self.metrics
        
    def _train_epoch(
        self,
        train_loader: DataLoader,
        optimizer: torch.optim.Optimizer,
        grad_accum_steps: int = 1
    ) -> Dict[str, float]:
        """Train for one epoch with optimized performance."""
        total_loss = 0.0
        total_quantum = 0.0
        total_scaling = 0.0
        total_phase = 0.0
        total_grad_norm = 0.0
        n_batches = 0
        optimizer.zero_grad()
        
        for i, (uv_batch, ir_batch) in enumerate(train_loader):
            # Move data to device efficiently
            uv_batch = uv_batch.to(self.device)
            ir_batch = ir_batch.to(self.device)
            
            # Forward pass
            pred_ir = self.model(uv_batch)
            loss, components = self.compute_loss(
                pred_ir, ir_batch, uv_batch,
                return_components=True
            )
            loss = loss / grad_accum_steps  # Scale loss for accumulation
            
            # Backward pass
            loss.backward()
            
            # Update weights with gradient accumulation
            if (i + 1) % grad_accum_steps == 0:
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), max_norm=1.0
                )
                optimizer.step()
                optimizer.zero_grad()
                total_grad_norm += grad_norm.item()
            
            # Update metrics
            total_loss += loss.item() * grad_accum_steps
            components_dict = cast(Dict[str, Union[torch.Tensor, float]], components)
            total_quantum += float(components_dict.get("quantum", 0.0))
            total_scaling += float(components_dict.get("scaling", 0.0))
            total_phase += float(components_dict.get("phase", 0.0))
            n_batches += 1
        
        return {
            "loss": total_loss / n_batches,
            "quantum": total_quantum / n_batches,
            "scaling": total_scaling / n_batches,
            "phase": total_phase / n_batches,
            "grad_norm": total_grad_norm / (n_batches // grad_accum_steps)
        }
        
    def _log_progress(
        self,
        epoch: int,
        train_metrics: Dict[str, float],
        val_metrics: Dict[str, float]
    ) -> None:
        """Log training progress with formatted output."""
        logger.info(f"\nEpoch {epoch}:")
        logger.info(f"  Train loss: {train_metrics['loss']:.2e}")
        logger.info(f"  Val loss: {val_metrics['loss']:.2e}")
        logger.info(f"  Quantum error: {val_metrics['quantum']:.2e}")
        logger.info(f"  Scaling error: {val_metrics['scaling']:.2e}")
        logger.info(f"  Phase error: {val_metrics['phase']:.2e}")
        if "grad_norm" in train_metrics:
            logger.info(f"  Grad norm: {train_metrics['grad_norm']:.2e}")
    
    @torch.no_grad()
    def validate(self, val_loader: DataLoader) -> Dict[str, float]:
        """Validate the model."""
        self.model.eval()
        metrics = defaultdict(float)
        
        for uv_batch, ir_batch in val_loader:
            pred_ir = self.model(uv_batch)
            loss, components = self.compute_loss(pred_ir, ir_batch, uv_batch, return_components=True)
            
            # Ensure components is a dict, not a tensor
            if isinstance(components, dict):
                for name, value in components.items():
                    metrics[name] += value
            metrics["loss"] += loss.item()
        
        n_batches = len(val_loader)
        return {k: v/n_batches for k, v in metrics.items()}
    
    def save_checkpoint(self, epoch: int, metrics: Dict[str, float]) -> None:
        """Save model checkpoint with metrics."""
        checkpoint = {
            "epoch": epoch,
            "model_state": self.model.state_dict(),
            "inverse_model_state": self.inverse_model.state_dict(),
            "metrics": metrics,
            "history": self.metrics
        }
        
        checkpoint_path = self.save_dir / f"checkpoint_epoch_{epoch}.pt"
        torch.save(checkpoint, checkpoint_path)
        logger.info(f"Saved checkpoint to {checkpoint_path}")
        
        # Save best model separately
        if metrics["val_loss"] < min(self.metrics.get("val_loss", [float('inf')])):
            best_model_path = self.save_dir / "best_model.pt"
            torch.save(checkpoint, best_model_path)
            logger.info("Saved new best model")
    
    def save_history(self):
        """Save training history."""
        # Convert tensors to floats for JSON serialization
        history = {
            k: [float(v) for v in vals]
            for k, vals in self.metrics.items()
        }
        
        with open(self.save_dir / "training_history.json", "w") as f:
            json.dump(history, f, indent=2)
    
    def load_checkpoint(self, path: Union[str, Path]) -> int:
        """Load checkpoint and return epoch number."""
        checkpoint = torch.load(path, map_location=self.device, weights_only=True)
        self.model.load_state_dict(checkpoint["model_state"])
        
        # Load inverse model state if available
        if "inverse_model_state" in checkpoint:
            self.inverse_model.load_state_dict(checkpoint["inverse_model_state"])
            
        # Load metrics if available
        if "history" in checkpoint:
            self.metrics = checkpoint["history"]
            
        return checkpoint.get("epoch", 0)
    
    def _get_data_loaders(
        self,
        batch_size: int,
        val_split: float,
        num_workers: int = 0
    ) -> Tuple[DataLoader, DataLoader]:
        """Generate training data and create data loaders."""
        # Generate dataset
        uv_data, ir_data = self.generate_training_data(batch_size * 100)  # Large dataset
        
        # Split into train/val
        n_val = int(len(uv_data) * val_split)
        indices = torch.randperm(len(uv_data))
        
        train_idx, val_idx = indices[n_val:], indices[:n_val]
        train_data = TensorDataset(uv_data[train_idx], ir_data[train_idx])
        val_data = TensorDataset(uv_data[val_idx], ir_data[val_idx])
        
        train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=num_workers)
        val_loader = DataLoader(val_data, batch_size=batch_size, num_workers=num_workers)
        
        return train_loader, val_loader
        
    def _validate_epoch(self, val_loader: DataLoader) -> Dict[str, float]:
        """Validate the model."""
        total_loss = 0.0
        total_quantum = 0.0
        total_scaling = 0.0
        total_phase = 0.0
        n_batches = 0
        
        for uv_batch, ir_batch in val_loader:
            # Forward pass
            pred_ir = self.model(uv_batch.to(self.device))
            loss, components = self.compute_loss(pred_ir, ir_batch.to(self.device), uv_batch.to(self.device), return_components=True)
            
            # Update metrics (convert tensors to floats)
            total_loss += loss.item()
            components_dict = cast(Dict[str, Union[torch.Tensor, float]], components)
            
            quantum_val = components_dict.get("quantum", 0.0)
            scaling_val = components_dict.get("scaling", 0.0)
            phase_val = components_dict.get("phase", 0.0)
            
            total_quantum += float(quantum_val.item() if torch.is_tensor(quantum_val) else quantum_val)
            total_scaling += float(scaling_val.item() if torch.is_tensor(scaling_val) else scaling_val)
            total_phase += float(phase_val.item() if torch.is_tensor(phase_val) else phase_val)
            n_batches += 1
        
        return {
            "loss": total_loss / n_batches,
            "quantum": total_quantum / n_batches,
            "scaling": total_scaling / n_batches,
            "phase": total_phase / n_batches
        }
        
    def _save_checkpoint(self, epoch: int, metrics: Dict[str, float]) -> None:
        """Save model checkpoint."""
        checkpoint = {
            "epoch": epoch,
            "model_state": self.model.state_dict(),
            "metrics": metrics,
            "config": {
                "dim": self.model.dim,
                "hidden_dim": self.model.output_proj.in_features,
                "n_layers": len(self.model.blocks),
                "dtype": str(self.model.dtype),
                "z_uv": self.model.z_uv,
                "z_ir": self.model.z_ir
            }
        }
        
        checkpoint_path = self.save_dir / f"checkpoint_epoch_{epoch}.pt"
        torch.save(checkpoint, checkpoint_path)
        print(f"\nSaved checkpoint to {checkpoint_path}")
        
        # Save best model separately
        best_model_path = self.save_dir / "best_model.pt"
        torch.save(checkpoint, best_model_path) 
    
    def train_epoch(
        self,
        train_loader: DataLoader,
        optimizer: torch.optim.Optimizer,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None
    ) -> Dict[str, float]:
        """Public interface for training a single epoch.
        
        Returns metrics dictionary with all required keys:
        - loss: Total loss
        - basic: Basic holographic error
        - quantum: Quantum correction error
        - phase: Phase coherence error
        """
        metrics = self._train_epoch(train_loader, optimizer)
        
        # Ensure all required keys are present
        metrics.update({
            'basic': metrics.get('scaling', 0.0),  # Scaling error is our basic error
            'quantum': metrics.get('quantum', 0.0),
            'phase': metrics.get('phase', 0.0)
        })
        
        if scheduler is not None:
            scheduler.step()
            
        return metrics 