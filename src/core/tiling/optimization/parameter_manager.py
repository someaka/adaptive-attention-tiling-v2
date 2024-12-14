"""Parameter management for adaptive attention tiling."""

from typing import Dict, List, Optional
import torch
import torch.nn as nn


class ParameterMonitor:
    """Monitor and track parameter changes during training."""

    def __init__(self, window_size: int = 100):
        """Initialize parameter monitor.
        
        Args:
            window_size: Size of sliding window for statistics
        """
        self.window_size = window_size
        self.parameter_history: Dict[str, List[torch.Tensor]] = {}
        self.gradient_history: Dict[str, List[torch.Tensor]] = {}
        
    def update(self, name: str, param: nn.Parameter):
        """Update parameter tracking.
        
        Args:
            name: Parameter name
            param: Parameter tensor
        """
        if name not in self.parameter_history:
            self.parameter_history[name] = []
            self.gradient_history[name] = []
            
        # Track parameter value
        self.parameter_history[name].append(param.data.clone())
        if len(self.parameter_history[name]) > self.window_size:
            self.parameter_history[name].pop(0)
            
        # Track gradient if available
        if param.grad is not None:
            self.gradient_history[name].append(param.grad.clone())
            if len(self.gradient_history[name]) > self.window_size:
                self.gradient_history[name].pop(0)
                
    def get_statistics(self, name: str) -> Dict[str, torch.Tensor]:
        """Get tracking statistics for parameter.
        
        Args:
            name: Parameter name
            
        Returns:
            Dictionary of statistics
        """
        if name not in self.parameter_history:
            return {}
            
        param_history = torch.stack(self.parameter_history[name])
        stats = {
            "mean": param_history.mean(0),
            "std": param_history.std(0),
            "min": param_history.min(0)[0],
            "max": param_history.max(0)[0],
        }
        
        if name in self.gradient_history and self.gradient_history[name]:
            grad_history = torch.stack(self.gradient_history[name])
            stats.update({
                "grad_mean": grad_history.mean(0),
                "grad_std": grad_history.std(0),
                "grad_norm": grad_history.norm(dim=1).mean(),
            })
            
        return stats


class AdaptiveParameterManager:
    """Manage parameters with adaptive updates."""
    
    def __init__(
        self,
        base_lr: float = 0.001,
        momentum: float = 0.9,
        adapt_factor: float = 1.01,
        min_lr: float = 1e-6,
        max_lr: float = 0.1
    ):
        """Initialize parameter manager.
        
        Args:
            base_lr: Base learning rate
            momentum: Momentum factor
            adapt_factor: Learning rate adaptation factor
            min_lr: Minimum learning rate
            max_lr: Maximum learning rate
        """
        self.base_lr = base_lr
        self.momentum = momentum
        self.adapt_factor = adapt_factor
        self.min_lr = min_lr
        self.max_lr = max_lr
        
        self.learning_rates: Dict[str, float] = {}
        self.velocities: Dict[str, torch.Tensor] = {}
        self.monitor = ParameterMonitor()
        
    def step(
        self,
        name: str,
        param: nn.Parameter,
        grad: Optional[torch.Tensor] = None
    ):
        """Perform parameter update step.
        
        Args:
            name: Parameter name
            param: Parameter to update
            grad: Optional gradient override
        """
        if name not in self.learning_rates:
            self.learning_rates[name] = self.base_lr
            self.velocities[name] = torch.zeros_like(param.data)
            
        # Get gradient
        if grad is None:
            if param.grad is None:
                return
            grad = param.grad
            
        # Update velocity with momentum
        self.velocities[name] = (
            self.momentum * self.velocities[name] +
            (1 - self.momentum) * grad
        )
        
        # Compute update
        update = -self.learning_rates[name] * self.velocities[name]
        
        # Apply update
        param.data.add_(update)
        
        # Monitor parameter
        self.monitor.update(name, param)
        
        # Adapt learning rate based on gradient behavior
        stats = self.monitor.get_statistics(name)
        if "grad_norm" in stats:
            grad_norm = stats["grad_norm"]
            if grad_norm < 1e-4:  # Increase if gradients are small
                self.learning_rates[name] = min(
                    self.learning_rates[name] * self.adapt_factor,
                    self.max_lr
                )
            elif grad_norm > 1.0:  # Decrease if gradients are large
                self.learning_rates[name] = max(
                    self.learning_rates[name] / self.adapt_factor,
                    self.min_lr
                )
                
    def get_lr(self, name: str) -> float:
        """Get current learning rate for parameter.
        
        Args:
            name: Parameter name
            
        Returns:
            Current learning rate
        """
        return self.learning_rates.get(name, self.base_lr)
        
    def get_state_dict(self) -> Dict:
        """Get state dictionary for checkpointing."""
        return {
            "learning_rates": self.learning_rates,
            "velocities": self.velocities,
            "base_lr": self.base_lr,
            "momentum": self.momentum,
            "adapt_factor": self.adapt_factor,
            "min_lr": self.min_lr,
            "max_lr": self.max_lr,
        }
        
    def load_state_dict(self, state_dict: Dict):
        """Load state dictionary from checkpoint."""
        self.learning_rates = state_dict["learning_rates"]
        self.velocities = state_dict["velocities"]
        self.base_lr = state_dict["base_lr"]
        self.momentum = state_dict["momentum"]
        self.adapt_factor = state_dict["adapt_factor"]
        self.min_lr = state_dict["min_lr"]
        self.max_lr = state_dict["max_lr"] 