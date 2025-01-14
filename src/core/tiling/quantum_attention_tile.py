"""Quantum Motivic Attention Tile.

This module implements quantum attention with motivic structure.
"""

from typing import Dict, List, Optional, Tuple, Union, Any, cast
import math
import torch
import torch.nn.functional as F
from torch import nn, Tensor

from ..patterns.fiber_types import LocalChart as PatternSection
from ..common.enums import ResolutionStrategy

MetricsDict = Dict[str, Union[float, List[float]]]

class QuantumMotivicTile(nn.Module):
    """Quantum attention with motivic structure."""

    def __init__(
        self,
        size: int,
        hidden_dim: int,
        num_heads: int = 1,
        dropout: float = 0.1,
        resolution: float = 1.0,
        cohomology_dim: int = 8,  # Dimension of cohomological structure
        motive_rank: int = 4,  # Rank of quantum motive
        dtype: torch.dtype = torch.float32
    ) -> None:
        """Initialize quantum motivic attention tile.
        
        Args:
            size: Size of the attention window
            hidden_dim: Hidden dimension size
            num_heads: Number of attention heads
            dropout: Dropout probability
            resolution: Initial resolution factor
            cohomology_dim: Dimension of cohomological structure
            motive_rank: Rank of quantum motive
            dtype: Data type for computations
            
        Raises:
            ValueError: If input parameters are invalid
        """
        super().__init__()

        # Validate input parameters
        if size <= 0:
            raise ValueError(f"Size must be positive, got {size}")
        if hidden_dim <= 0:
            raise ValueError(f"Hidden dimension must be positive, got {hidden_dim}")
        if num_heads <= 0:
            raise ValueError(f"Number of heads must be positive, got {num_heads}")
        if hidden_dim % num_heads != 0:
            raise ValueError(f"Hidden dimension {hidden_dim} must be divisible by number of heads {num_heads}")
        if not (0 <= dropout <= 1):
            raise ValueError(f"Dropout must be between 0 and 1, got {dropout}")
        if resolution <= 0:
            raise ValueError(f"Resolution must be positive, got {resolution}")
        if cohomology_dim <= 0:
            raise ValueError(f"Cohomology dimension must be positive, got {cohomology_dim}")
        if motive_rank <= 0:
            raise ValueError(f"Motive rank must be positive, got {motive_rank}")
        if dtype not in [torch.float32, torch.float64, torch.complex64, torch.complex128]:
            raise ValueError(f"Unsupported dtype: {dtype}")

        # Base attention parameters
        self.size = size
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.dropout = dropout
        self.resolution = resolution
        self.dtype = dtype
        self.cohomology_dim = cohomology_dim
        self.motive_rank = motive_rank

        # Initialize linear projections with correct dimensions
        self.input_proj = nn.Linear(self.head_dim, self.head_dim, bias=True)
        self.query = nn.Linear(self.head_dim, self.head_dim, bias=True)
        self.key = nn.Linear(self.head_dim, self.head_dim, bias=True)
        self.value = nn.Linear(self.head_dim, self.head_dim, bias=True)

        # Initialize dropout layer
        self.dropout_layer = nn.Dropout(dropout)

        # Convert weights to specified dtype
        with torch.no_grad():
            for layer in [self.input_proj, self.query, self.key, self.value]:
                if not isinstance(layer.weight, torch.Tensor):
                    raise ValueError(f"Expected weight to be tensor, got {type(layer.weight)}")
                if not isinstance(layer.bias, torch.Tensor):
                    raise ValueError(f"Expected bias to be tensor, got {type(layer.bias)}")
                layer.weight.data = layer.weight.data.to(dtype=self.dtype)
                layer.bias.data = layer.bias.data.to(dtype=self.dtype)

        # Initialize metrics dictionary
        self._metrics: MetricsDict = {}
        self._metrics_log: List[MetricsDict] = []
        self._neighbors: List['QuantumMotivicTile'] = []

        # Add cohomology projection
        self.cohomology_proj = nn.Linear(self.head_dim, cohomology_dim, dtype=dtype)
        self.cohomology_proj_inv = nn.Linear(cohomology_dim, self.head_dim, dtype=dtype)
        
        # Convert weights and bias to specified dtype
        with torch.no_grad():
            for layer in [self.cohomology_proj, self.cohomology_proj_inv]:
                if not isinstance(layer.weight, torch.Tensor):
                    raise ValueError(f"Expected weight to be tensor, got {type(layer.weight)}")
                layer.weight.data = layer.weight.data.to(dtype=self.dtype)
                if layer.bias is not None:
                    if not isinstance(layer.bias, torch.Tensor):
                        raise ValueError(f"Expected bias to be tensor, got {type(layer.bias)}")
                    layer.bias.data = layer.bias.data.to(dtype=self.dtype)

        # Add motive projection
        self.motive_proj = nn.Linear(self.head_dim, motive_rank, dtype=dtype)
        self.motive_proj_inv = nn.Linear(motive_rank, self.head_dim, dtype=dtype)
        
        # Convert weights and bias to specified dtype
        with torch.no_grad():
            for layer in [self.motive_proj, self.motive_proj_inv]:
                if not isinstance(layer.weight, torch.Tensor):
                    raise ValueError(f"Expected weight to be tensor, got {type(layer.weight)}")
                layer.weight.data = layer.weight.data.to(dtype=self.dtype)
                if layer.bias is not None:
                    if not isinstance(layer.bias, torch.Tensor):
                        raise ValueError(f"Expected bias to be tensor, got {type(layer.bias)}")
                    layer.bias.data = layer.bias.data.to(dtype=self.dtype)

        # Load balancing state
        self._load_factor = 1.0

    def _initialize_quantum_structure(self) -> None:
        """Initialize quantum structure and metrics components.
        
        Raises:
            ValueError: If tensor initialization fails
        """
        # Initialize base metrics
        self._metrics = cast(MetricsDict, {
            "ifq": 0.0,  # Information flow quotient
            "cer": 0.0,  # Cohomological error rate
            "ae": 0.0,   # Arithmetic entropy
            "quantum_entropy": 0.0,
            "motive_height": 0.0,
            "l_function_value": 0.01,
            "adelic_norm": 0.1,
            "resolution_history": [self.resolution],
            "flow": 0.0,
            "load_distribution": 1.0
        })

        # Initialize quantum state parameters
        quantum_state = torch.zeros(self.size, self.hidden_dim, dtype=self.dtype)
        if not torch.is_tensor(quantum_state):
            raise ValueError(f"Failed to initialize quantum_state tensor")
        self.register_buffer("quantum_state", quantum_state)

        # Initialize cohomology parameters
        cohomology_basis = torch.eye(self.cohomology_dim, dtype=self.dtype)
        if not torch.is_tensor(cohomology_basis):
            raise ValueError(f"Failed to initialize cohomology_basis tensor")
        self.register_buffer("cohomology_basis", cohomology_basis)

        # Initialize motive parameters
        motive_basis = torch.eye(self.motive_rank, dtype=self.dtype)
        if not torch.is_tensor(motive_basis):
            raise ValueError(f"Failed to initialize motive_basis tensor")
        self.register_buffer("motive_basis", motive_basis)

        # Validate initialized tensors
        for name, tensor in [
            ("quantum_state", self.quantum_state),
            ("cohomology_basis", self.cohomology_basis),
            ("motive_basis", self.motive_basis)
        ]:
            if not torch.is_tensor(tensor):
                raise ValueError(f"Expected {name} to be tensor, got {type(tensor)}")
            if tensor.dtype != self.dtype:
                raise ValueError(f"Expected {name} dtype {self.dtype}, got {tensor.dtype}")

    def get_metrics(self) -> MetricsDict:
        """Get current metrics.
        
        Returns:
            Copy of current metrics dictionary
        
        Raises:
            ValueError: If metrics dictionary is invalid
        """
        if not isinstance(self._metrics, dict):
            raise ValueError(f"Expected metrics to be dict, got {type(self._metrics)}")
        return self._metrics.copy()

    def _process_impl(self, x: torch.Tensor, update_metrics: bool = False) -> torch.Tensor:
        """Process input and update metrics.
        
        Args:
            x: Input tensor [batch_size, seq_len, hidden_dim]
            update_metrics: Whether to update metrics during processing
            
        Returns:
            Processed tensor
            
        Raises:
            ValueError: If input tensor has invalid shape or type
        """
        # Validate input
        if not torch.is_tensor(x):
            raise ValueError(f"Expected input to be tensor, got {type(x)}")
        if x.dim() != 3:
            raise ValueError(f"Expected 3D input tensor, got {x.dim()}D")
        if x.size(-1) != self.hidden_dim:
            raise ValueError(f"Expected hidden dimension {self.hidden_dim}, got {x.size(-1)}")
        if not torch.isfinite(x).all():
            raise ValueError("Input tensor contains inf or nan values")
            
        # Forward pass through attention mechanism
        output = self.forward(x)
        output_tensor = cast(torch.Tensor, output.coordinates if isinstance(output, PatternSection) else output)
        
        # Validate output tensor
        if not torch.is_tensor(output_tensor):
            raise ValueError(f"Expected output to be tensor, got {type(output_tensor)}")
        if output_tensor.shape != x.shape:
            raise ValueError(f"Expected output shape {x.shape}, got {output_tensor.shape}")
        if not torch.isfinite(output_tensor).all():
            raise ValueError("Output tensor contains inf or nan values")
        
        if update_metrics:
            # Update quantum metrics
            with torch.no_grad():
                # Validate scaling factors
                if not isinstance(self.resolution, (int, float)):
                    raise ValueError(f"Expected resolution to be numeric, got {type(self.resolution)}")
                if not isinstance(self._load_factor, (int, float)):
                    raise ValueError(f"Expected load_factor to be numeric, got {type(self._load_factor)}")
                
                # Update quantum entropy with resolution scaling
                x_scaled = x * self.resolution * self._load_factor
                
                # Validate scaled input
                if not torch.isfinite(x_scaled).all():
                    raise ValueError("Scaled input tensor contains inf or nan values")
                
                # Compute attention weights
                q_proj = self.query(x_scaled)
                k_proj = self.key(x_scaled)
                
                # Validate projections
                for name, tensor in [("query", q_proj), ("key", k_proj)]:
                    if not torch.is_tensor(tensor):
                        raise ValueError(f"Expected {name} projection to be tensor, got {type(tensor)}")
                    if tensor.shape != x_scaled.shape:
                        raise ValueError(f"Expected {name} shape {x_scaled.shape}, got {tensor.shape}")
                    if not torch.isfinite(tensor).all():
                        raise ValueError(f"{name} projection contains inf or nan values")
                
                attn_weights = torch.softmax(
                    torch.matmul(q_proj, k_proj.transpose(-2, -1)) 
                    / math.sqrt(self.head_dim), dim=-1
                )
                
                # Validate attention weights
                if not torch.is_tensor(attn_weights):
                    raise ValueError(f"Expected attention weights to be tensor, got {type(attn_weights)}")
                if not torch.isfinite(attn_weights).all():
                    raise ValueError("Attention weights contain inf or nan values")
                
                entropy = float(-(attn_weights * torch.log(attn_weights + 1e-9)).sum(-1).mean())
                self._metrics = cast(MetricsDict, {**self._metrics, "quantum_entropy": entropy})
                
                # Update cohomological error rate
                # Project to cohomology space with resolution scaling
                x_cohom = self.cohomology_proj(x_scaled)  # [batch_size, seq_len, cohomology_dim]
                
                # Validate cohomology projection
                if not torch.is_tensor(x_cohom):
                    raise ValueError(f"Expected cohomology projection to be tensor, got {type(x_cohom)}")
                if x_cohom.size(-1) != self.cohomology_dim:
                    raise ValueError(f"Expected cohomology dimension {self.cohomology_dim}, got {x_cohom.size(-1)}")
                if not torch.isfinite(x_cohom).all():
                    raise ValueError("Cohomology projection contains inf or nan values")
                
                x_proj = torch.matmul(
                    torch.matmul(x_cohom, self.cohomology_basis.T),
                    self.cohomology_basis
                )  # [batch_size, seq_len, cohomology_dim]
                
                # Validate projected tensor
                if not torch.is_tensor(x_proj):
                    raise ValueError(f"Expected projected tensor to be tensor, got {type(x_proj)}")
                if x_proj.shape != x_cohom.shape:
                    raise ValueError(f"Expected projection shape {x_cohom.shape}, got {x_proj.shape}")
                if not torch.isfinite(x_proj).all():
                    raise ValueError("Projected tensor contains inf or nan values")
                
                x_recon = self.cohomology_proj_inv(x_proj) / (self.resolution * self._load_factor)  # [batch_size, seq_len, hidden_dim]
                
                # Validate reconstruction
                if not torch.is_tensor(x_recon):
                    raise ValueError(f"Expected reconstruction to be tensor, got {type(x_recon)}")
                if x_recon.shape != x.shape:
                    raise ValueError(f"Expected reconstruction shape {x.shape}, got {x_recon.shape}")
                if not torch.isfinite(x_recon).all():
                    raise ValueError("Reconstruction contains inf or nan values")
                
                proj_error = torch.norm(x - x_recon)
                self._metrics = cast(MetricsDict, {**self._metrics, "cer": float(proj_error)})
                
                # Update arithmetic entropy
                # Project to motive space
                x_motive = self.motive_proj(x_scaled)  # [batch_size, seq_len, motive_rank]
                
                # Validate motive projection
                if not torch.is_tensor(x_motive):
                    raise ValueError(f"Expected motive projection to be tensor, got {type(x_motive)}")
                if x_motive.size(-1) != self.motive_rank:
                    raise ValueError(f"Expected motive rank {self.motive_rank}, got {x_motive.size(-1)}")
                if not torch.isfinite(x_motive).all():
                    raise ValueError("Motive projection contains inf or nan values")
                
                motive_coords = torch.matmul(
                    torch.matmul(x_motive, self.motive_basis.T),
                    self.motive_basis
                )  # [batch_size, seq_len, motive_rank]
                
                # Validate motive coordinates
                if not torch.is_tensor(motive_coords):
                    raise ValueError(f"Expected motive coordinates to be tensor, got {type(motive_coords)}")
                if motive_coords.shape != x_motive.shape:
                    raise ValueError(f"Expected motive coordinates shape {x_motive.shape}, got {motive_coords.shape}")
                if not torch.isfinite(motive_coords).all():
                    raise ValueError("Motive coordinates contain inf or nan values")
                
                ae_value = float(torch.norm(motive_coords) / (x.shape[1] * self.motive_rank))
                self._metrics = cast(MetricsDict, {**self._metrics, "ae": ae_value})
                
                # Update motive height
                height = float(torch.max(torch.abs(motive_coords)))
                self._metrics = cast(MetricsDict, {**self._metrics, "motive_height": height})
                
                # Update information flow quotient
                ifq = float(torch.norm(output_tensor) / (torch.norm(x) + 1e-9))
                self._metrics = cast(MetricsDict, {**self._metrics, "ifq": ifq})
                
                # Update L-function value (simplified)
                # Reshape motive coordinates to [batch_size * seq_len, motive_rank]
                motive_coords_flat = motive_coords.view(-1, self.motive_rank)
                
                # Validate flattened coordinates
                if not torch.is_tensor(motive_coords_flat):
                    raise ValueError(f"Expected flattened coordinates to be tensor, got {type(motive_coords_flat)}")
                if motive_coords_flat.size(-1) != self.motive_rank:
                    raise ValueError(f"Expected flattened rank {self.motive_rank}, got {motive_coords_flat.size(-1)}")
                
                l_value = max(0.01, float(torch.det(torch.eye(self.motive_rank) + 0.1 * motive_coords_flat.T @ motive_coords_flat)))
                self._metrics = cast(MetricsDict, {**self._metrics, "l_function_value": l_value})
                
                # Update adelic norm with scaling
                inf_norm = float(torch.norm(motive_coords, p=float('inf')))
                l1_norm = float(torch.norm(motive_coords, p=1))
                adelic = max(0.1, min(1.0, inf_norm / (l1_norm / (x.shape[1] * self.motive_rank) + 1e-9)))
                self._metrics = cast(MetricsDict, {**self._metrics, "adelic_norm": adelic})
                
                # Update flow metric if there are neighbors
                if self._neighbors:
                    neighbor_outputs = []
                    for n in self._neighbors:
                        if not isinstance(n, QuantumMotivicTile):
                            raise ValueError(f"Expected neighbor to be QuantumMotivicTile, got {type(n)}")
                        n_out = n.forward(x)
                        if isinstance(n_out, PatternSection):
                            neighbor_outputs.append(n_out.coordinates)
                        else:
                            neighbor_outputs.append(n_out)
                            
                    # Validate neighbor outputs
                    for i, n_out in enumerate(neighbor_outputs):
                        if not torch.is_tensor(n_out):
                            raise ValueError(f"Expected neighbor output {i} to be tensor, got {type(n_out)}")
                        if n_out.shape != output_tensor.shape:
                            raise ValueError(f"Expected neighbor output {i} shape {output_tensor.shape}, got {n_out.shape}")
                        if not torch.isfinite(n_out).all():
                            raise ValueError(f"Neighbor output {i} contains inf or nan values")
                    
                    flow = sum(torch.norm(n_out - output_tensor) for n_out in neighbor_outputs)
                    self._metrics = cast(MetricsDict, {**self._metrics, "flow": float(flow / len(self._neighbors))})
                
                # Store metrics history
                self._metrics_log.append(self._metrics.copy())
        
        return output_tensor

    def add_neighbor(self, neighbor: 'QuantumMotivicTile') -> None:
        """Add a neighboring tile.
        
        Args:
            neighbor: Neighboring quantum motivic tile
            
        Raises:
            ValueError: If neighbor is invalid
        """
        if not isinstance(neighbor, QuantumMotivicTile):
            raise ValueError(f"Expected neighbor to be QuantumMotivicTile, got {type(neighbor)}")
        if neighbor not in self._neighbors:
            self._neighbors.append(neighbor)

    def adapt_resolution(self, density_metric: float, strategy: ResolutionStrategy) -> None:
        """Adapt tile resolution based on density metric.
        
        Args:
            density_metric: Metric for adapting resolution
            strategy: Resolution adaptation strategy
            
        Raises:
            ValueError: If input parameters are invalid
        """
        if not isinstance(density_metric, (int, float)):
            raise ValueError(f"Expected density_metric to be numeric, got {type(density_metric)}")
        if not isinstance(strategy, ResolutionStrategy):
            raise ValueError(f"Expected strategy to be ResolutionStrategy, got {type(strategy)}")

        if strategy == ResolutionStrategy.ADAPTIVE:
            # Validate density metric
            if not (0 <= density_metric <= 1):
                raise ValueError(f"Density metric must be between 0 and 1, got {density_metric}")
                
            # Adjust resolution based on density
            new_resolution = self.resolution * (1.0 + 0.1 * (density_metric - 0.5))
            new_resolution = max(0.1, min(2.0, new_resolution))
            
            # Update resolution
            self.resolution = new_resolution
            resolution_history = cast(List[float], self._metrics["resolution_history"])
            if not isinstance(resolution_history, list):
                raise ValueError(f"Expected resolution_history to be list, got {type(resolution_history)}")
            resolution_history.append(new_resolution)

    def balance_load(self, neighbors: List['QuantumMotivicTile']) -> None:
        """Balance computational load with neighbors.
        
        Args:
            neighbors: List of neighboring tiles
        """
        # Add load distribution metric
        total_load = len(self._metrics_log) + sum(len(n._metrics_log) for n in neighbors)
        if total_load > 0:
            load_dist = float(len(self._metrics_log)) / total_load
            self._metrics = cast(MetricsDict, {**self._metrics, "load_distribution": load_dist})

            # Update load factor based on load distribution
            self._load_factor = 1.0 + 0.2 * (load_dist - 0.5)

        # Update neighbors
        self._neighbors = neighbors.copy()

    def forward(
        self,
        x: Union[torch.Tensor, PatternSection],
        return_metrics: bool = False
    ) -> Union[PatternSection, Tuple[PatternSection, Dict[str, Any]]]:
        """Process input through quantum attention tile.
        
        Args:
            x: Input tensor [batch_size, seq_len, hidden_dim] or PatternSection
            return_metrics: Whether to return metrics
            
        Returns:
            Processed pattern section and optional metrics
            
        Raises:
            ValueError: If input tensor has invalid shape or type
        """
        # Extract coordinates and transition maps
        if isinstance(x, PatternSection):
            coords = x.coordinates
            transition_maps = x.transition_maps
        else:
            coords = x
            transition_maps = {}

        # Validate input tensor
        if not torch.is_tensor(coords):
            raise ValueError(f"Expected torch.Tensor, got {type(coords)}")
            
        # Ensure input has correct shape
        if coords.dim() < 2 or coords.dim() > 3:
            raise ValueError(f"Expected 2D or 3D tensor, got {coords.dim()}D")
            
        # Add batch dimension if needed
        if coords.dim() == 2:
            coords = coords.unsqueeze(0)
            
        # Get dimensions
        batch_size, seq_len, hidden_dim = coords.shape
        
        # Validate dimensions
        if hidden_dim % self.head_dim != 0:
            raise ValueError(f"Hidden dimension {hidden_dim} must be divisible by head dimension {self.head_dim}")
            
        # Convert to correct dtype if needed
        if coords.dtype != self.dtype:
            coords = coords.to(dtype=self.dtype)
        
        # Ensure hidden_dim matches expected dimension
        if hidden_dim != self.hidden_dim:
            coords = coords.view(batch_size, seq_len, -1, self.head_dim)
            coords = coords.mean(dim=2)  # Average across extra dimensions

        # Project input through linear layers
        coords = self.input_proj(coords)  # [batch_size, seq_len, hidden_dim]
        q = self.query(coords)  # [batch_size, seq_len, hidden_dim]
        k = self.key(coords)    # [batch_size, seq_len, hidden_dim]
        v = self.value(coords)  # [batch_size, seq_len, hidden_dim]

        # Validate projected tensors
        for name, tensor in [("query", q), ("key", k), ("value", v)]:
            if not torch.is_tensor(tensor):
                raise ValueError(f"Expected {name} to be torch.Tensor, got {type(tensor)}")
            if tensor.shape != coords.shape:
                raise ValueError(f"Expected {name} shape {coords.shape}, got {tensor.shape}")

        # Reshape for multi-head attention
        q = q.reshape(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)  # [batch_size, num_heads, seq_len, head_dim]
        k = k.reshape(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)  # [batch_size, num_heads, seq_len, head_dim]
        v = v.reshape(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)  # [batch_size, num_heads, seq_len, head_dim]

        # Compute attention scores
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)  # [batch_size, num_heads, seq_len, seq_len]
        
        # Validate attention scores
        if not torch.isfinite(scores).all():
            raise ValueError("Attention scores contain inf or nan values")
        
        # Convert scores to real for softmax
        scores_real = scores.abs()
        attn = torch.softmax(scores_real, dim=-1)
        attn = self.dropout_layer(attn)

        # Apply attention
        output = torch.matmul(attn, v)  # [batch_size, num_heads, seq_len, head_dim]

        # Reshape back to original dimensions
        output = output.transpose(1, 2).reshape(batch_size, seq_len, -1)  # [batch_size, seq_len, hidden_dim]
        
        # Validate output tensor
        if not torch.isfinite(output).all():
            raise ValueError("Output contains inf or nan values")
        if output.shape != (batch_size, seq_len, self.hidden_dim):
            raise ValueError(f"Expected output shape {(batch_size, seq_len, self.hidden_dim)}, got {output.shape}")

        # Create new pattern section
        new_pattern = PatternSection(
            coordinates=output,
            dimension=output.shape[-1],
            transition_maps=transition_maps
        )
        
        if return_metrics:
            metrics = {
                'attention_scores': scores.detach(),
                'attention_probs': attn.detach(),
                'output_norm': output.norm().item(),
                'attention_entropy': -(attn * torch.log(attn + 1e-9)).sum(-1).mean().item()
            }
            return new_pattern, metrics
        
        return new_pattern
