"""Arithmetic Dynamics Layer.

This module implements the arithmetic dynamics layer that captures
deep structural patterns in computational motives. It bridges:

- Arithmetic Height Theory
- Dynamical Systems
- Motivic Integration
- L-functions and Modular Forms

The core idea is that computational patterns have an inherent
arithmetic structure that can be understood through dynamical
systems over adelic spaces.
"""

from typing import Dict, List, Tuple, Any, Union, Optional

import numpy as np
import torch
from torch import nn
from torchtyping import TensorType, patch_typeguard
from typeguard import typechecked
from torchinfo import summary

patch_typeguard()  # Enable runtime shape checking


class ArithmeticDynamics(nn.Module):
    """Arithmetic dynamics computation with quantum corrections."""

    def __init__(
        self,
        hidden_dim: int,
        motive_rank: int,
        num_primes: int = 8,  # Number of prime bases for adelic structure
        height_dim: Optional[int] = None,  # Dimension of height space
        quantum_weight: float = 0.1,  # Weight for quantum corrections
        dtype: Optional[torch.dtype] = None
    ):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.motive_rank = motive_rank
        self.num_primes = num_primes
        self.height_dim = motive_rank if height_dim is None else height_dim  # Use motive_rank as height_dim if not specified
        self.quantum_weight = quantum_weight
        self.dtype = dtype if dtype is not None else torch.float32  # Default to float32 for better compatibility

        # Initialize prime bases
        self.register_buffer(
            "primes",
            torch.tensor(self._get_first_n_primes(num_primes), dtype=self.dtype)
        )

        # Initialize coupling matrix as a learnable parameter
        self.coupling = nn.Parameter(
            torch.randn(num_primes, self.height_dim, dtype=self.dtype) / np.sqrt(num_primes * self.height_dim)
        )

        # Initialize networks
        self.height_map = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2, dtype=self.dtype),
            nn.Tanh(),
            nn.Linear(hidden_dim // 2, self.height_dim, dtype=self.dtype)
        )
        
        # Flow network should match height_dim
        self.flow = nn.Linear(self.height_dim, self.height_dim, dtype=self.dtype)
        
        # Initialize L-function network to output motive_rank dimensions
        self.l_function = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2, dtype=self.dtype),
            nn.Tanh(),
            nn.Linear(hidden_dim // 2, motive_rank, dtype=self.dtype)
        )

        # Adelic projection
        self.adelic_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2, dtype=self.dtype),
            nn.Tanh(),
            nn.Linear(hidden_dim // 2, num_primes * motive_rank, dtype=self.dtype)
        )

        # Output projection layers
        self.min_dim = hidden_dim  # Use hidden_dim as min_dim
        self.measure_proj = nn.Sequential(
            nn.Linear(self.height_dim, hidden_dim // 2, dtype=self.dtype),
            nn.Tanh(),
            nn.Linear(hidden_dim // 2, hidden_dim, dtype=self.dtype)
        )
        self.output_proj = nn.Linear(hidden_dim, hidden_dim, dtype=self.dtype)  # Keep same dimension

        # Quantum correction networks - ensure same output size as height_map
        self.quantum_height = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2, dtype=self.dtype),
            nn.Tanh(),
            nn.Linear(hidden_dim // 2, self.height_dim, dtype=self.dtype)  # Match height_map output size
        )
        
        self.quantum_l_function = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2, dtype=self.dtype),
            nn.Tanh(),
            nn.Linear(hidden_dim // 2, motive_rank, dtype=self.dtype)  # Match l_function output size
        )

        # Quantum correction projection
        self.quantum_proj = nn.Linear(hidden_dim, self.height_dim, dtype=self.dtype)  # Project to height_dim

    @staticmethod
    def _get_first_n_primes(n: int) -> List[int]:
        """Get first n prime numbers.
        
        Args:
            n: Number of primes to return
            
        Returns:
            List of first n primes
        """
        primes = []
        num = 2
        while len(primes) < n:
            if all(num % p != 0 for p in primes):
                primes.append(num)
            num += 1
        return primes

    def compute_height(self, x: torch.Tensor) -> torch.Tensor:
        """Compute height with quantum corrections.
        
        Args:
            x: Input tensor of shape [..., hidden_dim] or [..., N] where N != hidden_dim
            
        Returns:
            Height tensor of shape [..., height_dim]
        """
        # Save original shape for later reshaping
        original_shape = x.shape
        
        # Reshape input to [-1, hidden_dim] if last dimension doesn't match
        if x.shape[-1] != self.hidden_dim:
            # Flatten all but last dimension
            flat_shape = (-1, x.shape[-1])
            x = x.reshape(flat_shape)
            # Project to hidden_dim using adaptive pooling
            x = torch.nn.functional.adaptive_avg_pool1d(
                x.unsqueeze(1),  # Add channel dimension
                output_size=self.hidden_dim
            ).squeeze(1)  # Remove channel dimension
        
        # Classical height
        classical_height = self.height_map(x)
        
        # Quantum correction
        quantum_correction = self.quantum_height(x)
        
        # Combine with quantum weight
        result = classical_height + self.quantum_weight * quantum_correction
        
        # Reshape result to match input shape except for last dimension
        if len(original_shape) > 2:
            new_shape = original_shape[:-1] + (self.height_dim,)
            result = result.reshape(new_shape)
            
        return result

    def compute_l_function(self, x: torch.Tensor) -> torch.Tensor:
        """Compute L-function with quantum corrections.
        
        Args:
            x: Input tensor of shape [..., hidden_dim] or [..., N] where N != hidden_dim
            
        Returns:
            L-function values of shape [..., motive_rank]
        """
        # Save original shape for later reshaping
        original_shape = x.shape
        
        # Reshape input to [-1, hidden_dim] if last dimension doesn't match
        if x.shape[-1] != self.hidden_dim:
            # Flatten all but last dimension
            flat_shape = (-1, x.shape[-1])
            x = x.reshape(flat_shape)
            # Project to hidden_dim using adaptive pooling
            x = torch.nn.functional.adaptive_avg_pool1d(
                x.unsqueeze(1),  # Add channel dimension
                output_size=self.hidden_dim
            ).squeeze(1)  # Remove channel dimension
        
        # Classical L-function
        classical_l = self.l_function(x)
        
        # Quantum correction
        quantum_correction = self.quantum_l_function(x)
        
        # Combine with quantum weight
        result = classical_l + self.quantum_weight * quantum_correction
        
        # Reshape result to match input shape except for last dimension
        if len(original_shape) > 2:
            new_shape = original_shape[:-1] + (self.motive_rank,)
            result = result.reshape(new_shape)
            
        return result

    def compute_dynamics(self, x: torch.Tensor) -> torch.Tensor:
        """Compute arithmetic dynamics with quantum corrections.

        Args:
            x: Input tensor of shape [..., hidden_dim] or [..., N] where N != hidden_dim

        Returns:
            Evolved tensor of shape [..., hidden_dim]
        """
        # Save original shape and size for later reshaping
        original_shape = x.shape
        original_size = x.shape[-1]

        # Reshape input to [-1, hidden_dim] if last dimension doesn't match
        if x.shape[-1] != self.hidden_dim:
            # Flatten all but last dimension
            flat_shape = (-1, x.shape[-1])
            x_flat = x.reshape(flat_shape)
            
            # Project to hidden_dim using nearest neighbor interpolation
            x_proj = torch.nn.functional.interpolate(
                x_flat.unsqueeze(1),  # Add channel dimension
                size=self.hidden_dim,
                mode='nearest'  # Use nearest neighbor which works with complex numbers
            ).squeeze(1)  # Remove channel dimension
            
            # Convert to complex if needed
            if self.dtype.is_complex:
                x_proj = torch.complex(x_proj, torch.zeros_like(x_proj))
        else:
            x_proj = x
            if self.dtype.is_complex:
                x_proj = torch.complex(x_proj, torch.zeros_like(x_proj))

        # Apply height map
        height = self.height_map(x_proj)

        # Apply quantum flow
        flow = self.flow(height)

        # Project back to original size if needed
        if original_size != self.hidden_dim:
            # Handle complex tensors by splitting real and imaginary parts
            if flow.is_complex():
                # Split into real and imaginary parts
                flow_real = flow.real.unsqueeze(1)  # Add channel dimension
                flow_imag = flow.imag.unsqueeze(1)  # Add channel dimension
                
                # Interpolate real and imaginary parts separately
                flow_real_proj = torch.nn.functional.interpolate(
                    flow_real,
                    size=original_size,
                    mode='nearest'
                ).squeeze(1)  # Remove channel dimension
                
                flow_imag_proj = torch.nn.functional.interpolate(
                    flow_imag,
                    size=original_size,
                    mode='nearest'
                ).squeeze(1)  # Remove channel dimension
                
                # Combine back into complex tensor
                flow_proj = torch.complex(flow_real_proj, flow_imag_proj)
            else:
                # For real tensors, proceed as before
                flow_proj = torch.nn.functional.interpolate(
                    flow.unsqueeze(1),  # Add channel dimension
                    size=original_size,
                    mode='nearest'
                ).squeeze(1)  # Remove channel dimension
            
            # Reshape back to original shape
            flow = flow_proj.reshape(original_shape)
        else:
            flow = flow.reshape(original_shape)

        return flow

    def forward(
        self, x: torch.Tensor, steps: int = 1, return_trajectory: bool = False
    ) -> Tuple[torch.Tensor, Dict]:
        """Apply arithmetic dynamics with quantum corrections.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, hidden_dim) or (batch_size, hidden_dim)
            steps: Number of dynamics steps
            return_trajectory: Whether to return full trajectory
            
        Returns:
            Processed tensor and metrics dictionary
        """
        # Handle both 2D and 3D inputs
        if x.dim() == 2:
            batch_size, feat_dim = x.shape
            x = x.unsqueeze(1)  # Add sequence dimension
            seq_len = 1
        else:
            batch_size, seq_len, feat_dim = x.shape
        
        # Project to height space
        height_coords = self.height_map(x.reshape(-1, feat_dim))  # [batch_size * seq_len, height_dim]
        
        # Apply flow
        base_flow = self.flow(height_coords)  # [batch_size * seq_len, height_dim]
        
        # Project back to hidden_dim for L-function
        base_flow_hidden = self.measure_proj(base_flow)  # [batch_size * seq_len, hidden_dim]
        
        # Compute L-function values
        l_values = self.l_function(base_flow_hidden)  # [batch_size * seq_len, motive_rank]
        
        # Apply quantum corrections
        quantum_correction = self.quantum_height(x.reshape(-1, feat_dim))  # [batch_size * seq_len, height_dim]
        quantum_l = self.quantum_l_function(x.reshape(-1, feat_dim))  # [batch_size * seq_len, motive_rank]
        
        # Combine classical and quantum terms
        output = base_flow_hidden + self.quantum_weight * self.measure_proj(quantum_correction)
        output = output.view(batch_size, seq_len, -1)  # Restore original shape
        
        # Update metrics
        metrics = {
            'height': torch.norm(height_coords).item(),
            'l_value': torch.mean(l_values).item(),
            'flow_magnitude': torch.norm(base_flow).item(),
            'quantum_correction': torch.norm(quantum_correction).item(),
            'quantum_l': torch.mean(quantum_l).item()
        }
        
        return output, metrics

    def compute_modular_form(self, x: torch.Tensor) -> torch.Tensor:
        """Compute approximate modular form.

        This gives us a way to understand the symmetries in
        our computational patterns through the lens of
        modular forms.
        """
        height_coords = self.height_map(x)

        # Compute modular parameters (simplified)
        q = torch.exp(2 * np.pi * 1j * height_coords[..., 0])

        # Approximate modular form using q-expansion
        powers = torch.arange(1, 5, device=x.device)[None, :]
        q_powers = q[..., None] ** powers

        return q_powers.sum(dim=-1)

    def compute_quantum_correction(self, metric: torch.Tensor) -> torch.Tensor:
        """Compute quantum corrections to the metric.
        
        Args:
            metric: Input metric tensor of shape [batch_size, hidden_dim] or [batch_size, seq_len, hidden_dim]
                or [batch_size, manifold_dim, manifold_dim] or higher dimensional tensors
            
        Returns:
            Quantum correction tensor projected to measure space with shape matching the input metric
        """
        # Get original shape and flatten to 2D
        orig_shape = metric.shape[:-1]  # Remove last dimension
        metric_flat = metric.reshape(-1, metric.shape[-1])  # [batch_size * ..., hidden_dim]
        
        # Project to height space
        height_coords = self.quantum_height(metric_flat)  # [batch_size * ..., height_dim]
        
        # Project back to hidden_dim
        quantum_measure = self.measure_proj(height_coords)  # [batch_size * ..., hidden_dim]
        
        # Reshape back to original dimensions
        quantum_measure = quantum_measure.reshape(*orig_shape, self.hidden_dim)
            
        return quantum_measure

    def compute_quantum_metric(self, x: torch.Tensor) -> torch.Tensor:
        """Compute quantum metric tensor.
        
        Args:
            x: Input tensor of shape [..., hidden_dim]
            
        Returns:
            Quantum metric tensor of shape [..., hidden_dim, hidden_dim]
        """
        # Save original shape for later reshaping
        original_shape = x.shape[:-1]  # All but last dimension
        
        # Flatten input to 2D
        x_flat = x.reshape(-1, x.shape[-1])  # [batch_size * ..., hidden_dim]
        
        # Project to quantum space
        quantum_state = self.project_quantum_state(x_flat)  # [batch_size * ..., height_dim]
        
        # Project back to hidden_dim
        quantum_measure = self.measure_proj(quantum_state)  # [batch_size * ..., hidden_dim]
        
        # Compute outer product to get metric tensor
        quantum_metric = torch.einsum('bi,bj->bij', quantum_measure, quantum_measure)
        
        # Reshape back to original dimensions plus metric dimensions
        if original_shape:
            quantum_metric = quantum_metric.reshape(*original_shape, self.hidden_dim, self.hidden_dim)
        
        return quantum_metric

    def project_quantum_state(self, x: torch.Tensor) -> torch.Tensor:
        """Project input to quantum state space."""
        # Ensure input is in complex domain
        x = x.to(dtype=torch.complex64)

        # Project to quantum state space
        x_flat = x.view(-1, x.size(-1))  # [batch_size * ..., height_dim]
        quantum_state = self.quantum_proj(x_flat)  # [batch_size * ..., height_dim]
        
        # Normalize the quantum state
        quantum_state = quantum_state / torch.norm(quantum_state, dim=1, keepdim=True)
        
        return quantum_state


class ArithmeticPattern(nn.Module):
    """Pattern detection through arithmetic dynamics."""

    def __init__(
        self, input_dim: int, hidden_dim: int, motive_rank: int = 4, num_layers: int = 3,
        dtype: Optional[torch.dtype] = None
    ):
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.motive_rank = motive_rank
        self.num_layers = num_layers
        self.dtype = dtype if dtype is not None else torch.float32

        # Arithmetic dynamics layers
        self.layers = nn.ModuleList(
            [
                ArithmeticDynamics(
                    hidden_dim=hidden_dim if i > 0 else input_dim,
                    motive_rank=motive_rank,
                    dtype=self.dtype
                )
                for i in range(num_layers)
            ]
        )

        # Pattern projection
        self.pattern_proj = nn.Linear(hidden_dim, hidden_dim, dtype=self.dtype)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, List[Dict]]:
        """
        Apply layered arithmetic dynamics.

        Args:
            x: Input tensor of shape (batch_size, seq_len, hidden_dim)

        Returns:
            - Output tensor of shape (batch_size, seq_len, hidden_dim)
            - List of metrics from each layer
        """
        batch_size, seq_len, hidden_dim = x.shape
        metrics = []

        # Process through layers
        current = x
        for layer in self.layers:
            current, layer_metrics = layer(current)
            metrics.append(
                {
                    "layer_norm": current.norm().item(),
                    "layer_mean": current.mean().item(),
                    "layer_std": current.std().item(),
                    **layer_metrics,  # Include metrics from ArithmeticDynamics
                }
            )

        # Project final output
        output = self.pattern_proj(current)

        return output, metrics


class ModularFormComputer(nn.Module):
    """Compute modular forms for arithmetic dynamics."""
    
    def __init__(
        self,
        hidden_dim: int,
        weight: int = 2,  # Weight of the modular form
        level: int = 1,   # Level of the modular form
        num_coeffs: int = 10,  # Number of q-expansion coefficients
    ):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.weight = weight
        self.level = level
        self.num_coeffs = num_coeffs
        
        # Network for computing q-expansion coefficients
        self.coeff_net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, num_coeffs * 2)  # Real and imaginary parts
        )
        
        # Network for computing symmetry parameters
        self.symmetry_net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.Tanh(),
            nn.Linear(hidden_dim // 2, 2)  # Translation and inversion parameters
        )
        
    def compute_q_expansion(self, x: torch.Tensor) -> torch.Tensor:
        """Compute q-expansion coefficients.
        
        Args:
            x: Input tensor of shape [batch_size, hidden_dim]
            
        Returns:
            Complex coefficients of shape [batch_size, num_coeffs]
        """
        # Get coefficients from network
        coeffs = self.coeff_net(x)
        
        # Split into real and imaginary parts
        real = coeffs[..., :self.num_coeffs]
        imag = coeffs[..., self.num_coeffs:]
        
        # Combine into complex coefficients
        return torch.complex(real, imag)
    
    def compute_symmetries(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Compute modular symmetry parameters.
        
        Args:
            x: Input tensor of shape [batch_size, hidden_dim]
            
        Returns:
            Dictionary containing symmetry parameters
        """
        params = self.symmetry_net(x)
        
        return {
            'translation': params[..., 0],  # Translation parameter
            'inversion': params[..., 1]     # Inversion parameter
        }
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """Compute modular form.
        
        Args:
            x: Input tensor of shape [batch_size, hidden_dim]
            
        Returns:
            - q-expansion coefficients
            - Dictionary of symmetry parameters and metrics
        """
        # Compute q-expansion
        q_coeffs = self.compute_q_expansion(x)
        
        # Compute symmetries
        symmetries = self.compute_symmetries(x)
        
        # Compute metrics
        metrics = {
            'weight': self.weight,
            'level': self.level,
            'q_norm': torch.norm(q_coeffs, dim=-1).mean().item(),
            'translation_param': symmetries['translation'].mean().item(),
            'inversion_param': symmetries['inversion'].mean().item()
        }
        
        return q_coeffs, metrics
    





# class ArithmeticDynamics:
#     """Implement arithmetic dynamics for attention evolution."""

#     def __init__(self, hidden_dim: int, motive_rank: int, num_primes: int = 8, dtype: Optional[torch.dtype] = None):
#         self.hidden_dim = hidden_dim
#         self.motive_rank = motive_rank
#         self.num_primes = num_primes
#         self.dtype = dtype if dtype is not None else torch.float32

#         # Project to hidden dimension first using adaptive pooling
#         self.hidden_proj = nn.Sequential(
#             nn.AdaptiveAvgPool1d(hidden_dim),  # Handle variable input sizes
#             nn.Linear(hidden_dim, hidden_dim, dtype=self.dtype)
#         )

#         # L-function computation
#         self.l_function = nn.Sequential(
#             nn.Linear(hidden_dim, hidden_dim // 2, dtype=self.dtype),
#             nn.Tanh(),
#             nn.Linear(hidden_dim // 2, motive_rank, dtype=self.dtype)
#         )

#         # Flow computation
#         self.flow = nn.Linear(motive_rank, motive_rank, dtype=self.dtype)

#     def compute_dynamics(self, state: torch.Tensor) -> torch.Tensor:
#         """Compute one step of arithmetic dynamics.
#         
#         Args:
#             state: Input tensor of shape [batch_size, *]
#             
#         Returns:
#             Tensor of shape [batch_size, motive_rank]
#         """
#         # Ensure input is 2D: [batch_size, features]
#         if state.dim() == 1:
#             state = state.unsqueeze(0)
#             
#         # Flatten all dimensions after batch
#         batch_size = state.shape[0]
#         state_flat = state.reshape(batch_size, -1)  # [batch_size, num_features]
#         
#         # Add channel dimension for adaptive pooling
#         state_channels = state_flat.unsqueeze(1)  # [batch_size, 1, num_features]
#         
#         # Project to hidden dimension using adaptive pooling
#         hidden_state = self.hidden_proj[0](state_channels)  # [batch_size, 1, hidden_dim]
#         hidden_state = hidden_state.squeeze(1)  # [batch_size, hidden_dim]
#         hidden_state = self.hidden_proj[1](hidden_state)  # [batch_size, hidden_dim]
#         
#         # Compute L-function values
#         l_values = self.l_function(hidden_state)  # [batch_size, motive_rank]

#         # Evolve using flow
#         evolved = self.flow(l_values)  # [batch_size, motive_rank]

#         return evolved

