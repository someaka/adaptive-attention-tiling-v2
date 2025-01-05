import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from torch.optim.lr_scheduler import ReduceLROnPlateau
from typing import Optional, Dict, List, Tuple, Union
import numpy as np
from pathlib import Path
import json
from datetime import datetime
from collections import defaultdict

from .models import HolographicNet


class HolographicTrainer:
    """Trainer for holographic neural networks.
    
    This class handles:
    1. Training data generation with physics-informed examples
    2. Loss computation using conformal structure
    3. Training loop with validation and early stopping
    4. Model checkpointing and metrics logging
    """
    
    def __init__(
        self,
        model: HolographicNet,
        save_dir: str = "pretrained",
        device: Optional[torch.device] = None
    ):
        self.model = model
        self.save_dir = Path(save_dir)
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        
        # Create save directory and inverse model in one go
        self.save_dir.mkdir(parents=True, exist_ok=True)
        # Get hidden dimension from output projection layer
        hidden_dim = model.output_proj.in_features
        self.inverse_model = HolographicNet(
            dim=model.dim,
            hidden_dim=hidden_dim * 2,  # Double the hidden dimension for inverse model
            n_layers=len(model.blocks),  # Same number of layers
            z_uv=model.z_ir,  # Swap UV/IR for inverse
            z_ir=model.z_uv
        ).to(self.device)
        
        # Initialize metrics tracking
        self.metrics = {
            "train_loss": [], "val_loss": [],
            "scaling_error": [], "quantum_error": [],
            "lr": [], "epoch": []
        }
    
    def generate_training_data(
        self,
        batch_size: int,
        include_special_cases: bool = True
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Generate training data with physics-informed examples."""
        # Vectorized random UV state generation
        uv_data = torch.randn(batch_size, self.model.dim, 
                            dtype=self.model.dtype, device=self.device)
        uv_data = uv_data / torch.norm(uv_data, dim=1, keepdim=True)
        
        if include_special_cases:
            n_special = batch_size // 4
            # Generate special cases in parallel
            symmetric = torch.randn(n_special, self.model.dim // 2, 
                                  dtype=self.model.dtype, device=self.device)
            symmetric = torch.cat([symmetric, symmetric.conj()], dim=1) / np.sqrt(2)
            
            # Localized states using scatter
            localized = torch.zeros(n_special, self.model.dim, 
                                  dtype=self.model.dtype, device=self.device)
            indices = torch.randint(0, self.model.dim, (n_special, 1), device=self.device)
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
        
        # Compute all corrections at once
        z_powers = z_ratio**powers[:, None, None]
        ir_data = (uv_data[None, :, :] * z_powers * coeffs[:, None, None]).sum(dim=0)
        
        return uv_data, ir_data
    
    def compute_loss(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        input: torch.Tensor,
        return_components: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict[str, float]]]:
        """Compute physics-informed loss function with natural weighting."""
        # Get the natural scale from z_ratio
        z_ratio = self.model.z_ratio
        N = z_ratio**self.model.dim
        
        # Holographic error with conformal weight
        holographic_error = N*pred - input
        conformal_factor = N**2 + 1  # From our symbolic analysis
        mse_loss = torch.mean(torch.abs(holographic_error)**2) / conformal_factor
        
        # Scale term for quantum corrections
        scale_loss = torch.mean(torch.abs(pred - target)**2)
        
        # Phase term as conformal transformation
        phase_diff = torch.angle(pred) - torch.angle(input)
        quantum_loss = torch.mean(torch.abs(input)**2 * (1 - torch.cos(phase_diff))) / N
        
        # Total loss combines all terms
        total_loss = mse_loss.real + scale_loss.real + quantum_loss.real
        
        if return_components:
            return total_loss, {
                "mse": float(mse_loss.real),
                "scale": float(scale_loss.real),
                "quantum": float(quantum_loss.real)
            }
        return total_loss
    
    def train_epoch(
        self,
        train_loader: DataLoader,
        optimizer: torch.optim.Optimizer,
        scheduler: Optional[Union[torch.optim.lr_scheduler._LRScheduler, ReduceLROnPlateau]] = None
    ) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        components = {"mse": 0.0, "scale": 0.0, "quantum": 0.0}
        
        for uv_batch, ir_batch in train_loader:
            optimizer.zero_grad()
            pred_ir = self.model(uv_batch)
            loss, components_dict = self.compute_loss(pred_ir, ir_batch, uv_batch, return_components=True)
            total_loss += loss.item()
            batch_components: Dict[str, float] = components_dict  # type: ignore
            for k, v in batch_components.items():
                components[k] += v
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            optimizer.step()
        
        if scheduler is not None:
            if isinstance(scheduler, ReduceLROnPlateau):
                scheduler.step(total_loss / len(train_loader))
            else:
                scheduler.step()
        
        n_batches = len(train_loader)
        return {
            "loss": total_loss / n_batches,
            **{k: v / n_batches for k, v in components.items()}
        }
    
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
    
    def train(
        self,
        n_epochs: int = 1000,
        batch_size: int = 128,
        val_split: float = 0.2,
        lr: float = 1e-3,
        patience: int = 50,
        min_delta: float = 1e-4
    ) -> Dict[str, List[float]]:
        """Train the model with early stopping."""
        # Generate dataset
        uv_data, ir_data = self.generate_training_data(batch_size * 100)  # Large dataset
        
        # Split into train/val
        n_val = int(len(uv_data) * val_split)
        indices = torch.randperm(len(uv_data))
        
        train_idx, val_idx = indices[n_val:], indices[:n_val]
        train_data = TensorDataset(uv_data[train_idx], ir_data[train_idx])
        val_data = TensorDataset(uv_data[val_idx], ir_data[val_idx])
        
        train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_data, batch_size=batch_size)
        
        # Setup training
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        scheduler = ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=patience//2, verbose='True'
        )
        
        best_val_loss = float('inf')
        best_epoch = 0
        no_improve = 0
        
        # Training loop
        for epoch in range(n_epochs):
            # Train
            train_metrics = self.train_epoch(train_loader, optimizer, scheduler)
            
            # Validate
            val_metrics = self.validate(val_loader)
            
            # Update history
            self.metrics["train_loss"].append(train_metrics["loss"])
            self.metrics["val_loss"].append(val_metrics["loss"])
            self.metrics["scaling_error"].append(val_metrics["scale"])
            self.metrics["quantum_error"].append(val_metrics["quantum"])
            self.metrics["lr"].append(optimizer.param_groups[0]["lr"])
            self.metrics["epoch"].append(epoch)
            
            # Log progress
            if epoch % 10 == 0:
                print(f"Epoch {epoch}:")
                print(f"  Train loss: {train_metrics['loss']:.2e}")
                print(f"  Val loss: {val_metrics['loss']:.2e}")
                print(f"  Scale error: {val_metrics['scale']:.2e}")
                print(f"  Quantum error: {val_metrics['quantum']:.2e}")
            
            # Check for improvement
            if val_metrics["loss"] < best_val_loss - min_delta:
                best_val_loss = val_metrics["loss"]
                best_epoch = epoch
                no_improve = 0
                
                # Save best model
                self.save_checkpoint(epoch, val_metrics)
            else:
                no_improve += 1
            
            # Early stopping
            if no_improve >= patience:
                print(f"Early stopping at epoch {epoch}")
                break
        
        # Save training history
        self.save_history()
        
        print(f"\nTraining completed:")
        print(f"Best validation loss: {best_val_loss:.2e} at epoch {best_epoch}")
        
        return self.metrics
    
    def save_checkpoint(self, epoch: int, metrics: Dict[str, float]) -> None:
        """Save model checkpoint with metrics."""
        checkpoint = {
            "epoch": epoch,
            "model_state": self.model.state_dict(),
            "inverse_model_state": self.inverse_model.state_dict(),
            "metrics": metrics,
            "history": self.metrics
        }
        torch.save(checkpoint, self.save_dir / f"checkpoint_epoch_{epoch}.pt")
        
        # Save best model separately
        if metrics["val_loss"] < min(self.metrics["val_loss"], default=float('inf')):
            torch.save(checkpoint, self.save_dir / "best_model.pt")
    
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
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state"])
        self.inverse_model.load_state_dict(checkpoint["inverse_model_state"])
        self.metrics = checkpoint["history"]
        return checkpoint["epoch"] 