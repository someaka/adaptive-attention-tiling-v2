from dataclasses import dataclass
from typing import Callable, List, Tuple, Dict, Any, Union, Optional
import gc

import numpy as np
import torch
from torch import nn

from src.utils.memory_management import register_tensor
from src.core.crystal.scale_classes.complextanh import ComplexTanh
from src.core.crystal.scale_classes.memory_utils import memory_manager, memory_efficient_computation




@dataclass
class ScaleConnectionData:
    """Data class for scale connection results."""
    source_scale: Union[float, torch.Tensor]
    target_scale: Union[float, torch.Tensor]
    connection_map: torch.Tensor
    holonomy: torch.Tensor



@dataclass
class AnomalyPolynomial:
    """Represents an anomaly polynomial."""

    coefficients: torch.Tensor  # Polynomial coefficients
    variables: List[str]  # Variable names
    degree: int  # Polynomial degree
    type: str  # Type of anomaly





class ScaleConnection:
    """Implementation of scale connections with optimized memory usage."""

    def __init__(self, dim: int, num_scales: int = 4, dtype=torch.float32):
        """Initialize scale connection with memory-efficient setup.
        
        Args:
            dim: Dimension of the space (must be positive)
            num_scales: Number of scale levels (must be at least 2)
            dtype: Data type for tensors
            
        Raises:
            ValueError: If dim <= 0 or num_scales < 2
        """
        if dim <= 0:
            raise ValueError(f"Dimension must be positive, got {dim}")
        if num_scales < 2:
            raise ValueError(f"Number of scales must be at least 2, got {num_scales}")
            
        self.dim = dim
        self.num_scales = num_scales
        self.dtype = dtype

        # Initialize scale connections
        self.connections = nn.ModuleList([
            nn.Linear(dim, dim, dtype=dtype)
            for _ in range(num_scales - 1)
        ])

        # Register parameters for memory tracking
        for connection in self.connections:
            connection.weight.data = register_tensor(connection.weight.data)
            if connection.bias is not None:
                connection.bias.data = register_tensor(connection.bias.data)

        # Holonomy computation with optimized architecture
        activation = ComplexTanh() if dtype == torch.complex64 else nn.Tanh()
        
        # Create and register holonomy computer layers
        self.holonomy_in = nn.Linear(dim * 2, dim * 4, dtype=dtype)
        self.holonomy_out = nn.Linear(dim * 4, dim * dim, dtype=dtype)
        
        # Register parameters
        self.holonomy_in.weight.data = register_tensor(self.holonomy_in.weight.data)
        self.holonomy_out.weight.data = register_tensor(self.holonomy_out.weight.data)
        if self.holonomy_in.bias is not None:
            self.holonomy_in.bias.data = register_tensor(self.holonomy_in.bias.data)
        if self.holonomy_out.bias is not None:
            self.holonomy_out.bias.data = register_tensor(self.holonomy_out.bias.data)
            
        self.holonomy_computer = nn.Sequential(
            self.holonomy_in,
            activation,
            self.holonomy_out
        )
        
        # Initialize memory manager for tensor operations
        self._memory_manager = memory_manager

    def connect_scales(
        self, source_state: torch.Tensor, source_scale: float, target_scale: float
    ) -> torch.Tensor:
        """Connect states at different scales with memory optimization."""
        with memory_efficient_computation("connect_scales"):
            # Ensure input tensors have correct dtype
            source_state = source_state.to(dtype=self.dtype)
            
            scale_idx = int(np.log2(target_scale / source_scale))
            if scale_idx >= len(self.connections):
                raise ValueError("Scale difference too large")

            # Use in-place operations where possible
            result = torch.empty_like(source_state)
            self.connections[scale_idx](source_state, out=result)
            return result

    def compute_holonomy(self, states: List[torch.Tensor]) -> torch.Tensor:
        """Compute holonomy with improved memory efficiency."""
        with memory_efficient_computation("compute_holonomy"):
            # Ensure states have correct dtype and batch dimension
            states_batch = []
            for s in states:
                state = s.to(dtype=self.dtype)
                if state.dim() == 2:
                    state = state.unsqueeze(0)
                states_batch.append(state)
            
            # Concatenate initial and final states along feature dimension
            # Use in-place operations where possible
            loop_states = torch.cat([states_batch[0], states_batch[-1]], dim=-1)
            
            # Compute holonomy and reshape to dim x dim matrix
            # Use pre-allocated tensors for reshaping
            holonomy = self.holonomy_computer(loop_states)
            result = holonomy.view(-1, self.dim, self.dim)
            
            # Remove batch dimension if added
            if result.size(0) == 1:
                result = result.squeeze(0)
            
            return result

    def _cleanup(self):
        """Clean up memory when the connection is no longer needed."""
        for connection in self.connections:
            del connection
        del self.holonomy_computer
        torch.cuda.empty_cache()  # Clean GPU memory if available
        gc.collect()  # Trigger garbage collection

    def __del__(self):
        """Ensure proper cleanup of resources."""
        self._cleanup()






class AnomalyDetector:
    """Detection and analysis of anomalies with optimized performance."""

    def __init__(self, dim: int, max_degree: int = 4, dtype=torch.float32):
        self.dim = dim
        self.max_degree = max_degree
        self.dtype = dtype

        # Anomaly detection network with optimized architecture
        activation = ComplexTanh() if dtype == torch.complex64 else nn.Tanh()
        
        # Create and register detector layers
        self.detector_in = nn.Linear(dim, dim * 2, dtype=dtype)
        self.detector_out = nn.Linear(dim * 2, max_degree + 1, dtype=dtype)
        
        # Register parameters for memory tracking
        self.detector_in.weight.data = register_tensor(self.detector_in.weight.data)
        self.detector_out.weight.data = register_tensor(self.detector_out.weight.data)
        if self.detector_in.bias is not None:
            self.detector_in.bias.data = register_tensor(self.detector_in.bias.data)
        if self.detector_out.bias is not None:
            self.detector_out.bias.data = register_tensor(self.detector_out.bias.data)
            
        self.detector = nn.Sequential(
            self.detector_in,
            activation,
            self.detector_out
        )

        # Variable names for polynomials
        self.variables = [f"x_{i}" for i in range(dim)]
        
        # Initialize memory manager
        self._memory_manager = memory_manager
        
        # Cache for polynomial computations
        self._poly_cache: Dict[str, List[AnomalyPolynomial]] = {}

    def detect_anomalies(self, state: torch.Tensor) -> List[AnomalyPolynomial]:
        """Detect anomalies in quantum state with improved efficiency."""
        with memory_efficient_computation("detect_anomalies"):
            # Check cache first
            state_key = self._get_state_key(state)
            if state_key in self._poly_cache:
                return self._poly_cache[state_key]
            
            anomalies = []

            # Analyze state for different polynomial degrees
            # Use in-place operations where possible
            coefficients = torch.empty((self.max_degree + 1,), dtype=self.dtype)
            self.detector(state, out=coefficients)

            # Process coefficients efficiently
            for degree in range(self.max_degree + 1):
                coeff_slice = coefficients[degree:]
                if torch.norm(coeff_slice) > 1e-6:
                    anomalies.append(
                        AnomalyPolynomial(
                            coefficients=coeff_slice.clone(),  # Clone to preserve data
                            variables=self.variables[: degree + 1],
                            degree=degree,
                            type=self._classify_anomaly(degree)
                        )
                    )

            # Cache results
            self._poly_cache[state_key] = anomalies
            return anomalies

    def _get_state_key(self, state: torch.Tensor) -> str:
        """Generate cache key for state tensor."""
        with torch.no_grad():
            if state.is_complex():
                return "_".join(
                    f"{x.real:.6f}_{x.imag:.6f}" 
                    for x in state.detach().flatten()
                )
            return "_".join(
                f"{x:.6f}" 
                for x in state.detach().flatten()
            )

    def _classify_anomaly(self, degree: int) -> str:
        """Classify type of anomaly based on degree."""
        if degree == 1:
            return "linear"
        if degree == 2:
            return "quadratic"
        if degree == 3:
            return "cubic"
        return f"degree_{degree}"

    def _cleanup(self):
        """Clean up resources."""
        del self.detector
        self._poly_cache.clear()
        torch.cuda.empty_cache()
        gc.collect()

    def __del__(self):
        """Ensure proper cleanup."""
        self._cleanup()

