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
    """Represents an anomaly polynomial with consistency checks."""

    coefficients: torch.Tensor  # Polynomial coefficients
    variables: List[str]  # Variable names
    degree: int  # Polynomial degree
    type: str  # Type of anomaly
    winding_number: Optional[float] = None  # Winding number for U(1) symmetries
    is_consistent: Optional[bool] = None  # Whether anomaly satisfies consistency conditions

    def __post_init__(self):
        """Compute winding number and check consistency if not provided."""
        if self.winding_number is None and self.coefficients is not None:
            # Compute winding number from phases of coefficients
            phases = torch.angle(self.coefficients[self.coefficients.abs() > 1e-6])
            self.winding_number = float(torch.sum(torch.diff(phases)) / (2 * torch.pi))
            
        if self.is_consistent is None and self.coefficients is not None:
            # Check basic consistency conditions
            self.is_consistent = self._check_consistency()
            
    def _check_consistency(self) -> bool:
        """Check if anomaly satisfies basic consistency conditions."""
        # 1. Check coefficient normalization
        norm = torch.norm(self.coefficients)
        if norm < 1e-6:
            return False
            
        # 2. Check phase consistency for U(1)
        if self.type == "U1":
            phases = torch.angle(self.coefficients[self.coefficients.abs() > 1e-6])
            if phases.numel() <= 1:  # Handle single coefficient case
                return True
            phase_diffs = torch.diff(phases)
            if phase_diffs.numel() == 0:  # Handle no phase differences
                return True
            return torch.allclose(phase_diffs, phase_diffs[0], rtol=1e-2)
            
        return True





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
        
        # Create and register detector layers with symmetry preservation
        self.detector_in = nn.Linear(dim, dim * 4, dtype=dtype)  # Increased width
        self.detector_hidden = nn.Linear(dim * 4, dim * 4, dtype=dtype)  # Added hidden layer
        self.detector_out = nn.Linear(dim * 4, max_degree + 1, dtype=dtype)
        
        # Register parameters for memory tracking
        self.detector_in.weight.data = register_tensor(self.detector_in.weight.data)
        self.detector_hidden.weight.data = register_tensor(self.detector_hidden.weight.data)
        self.detector_out.weight.data = register_tensor(self.detector_out.weight.data)
        
        if self.detector_in.bias is not None:
            self.detector_in.bias.data = register_tensor(self.detector_in.bias.data)
        if self.detector_hidden.bias is not None:
            self.detector_hidden.bias.data = register_tensor(self.detector_hidden.bias.data)
        if self.detector_out.bias is not None:
            self.detector_out.bias.data = register_tensor(self.detector_out.bias.data)
            
        self.detector = nn.Sequential(
            self.detector_in,
            activation,
            self.detector_hidden,
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
        """Detect anomalies in quantum state with improved efficiency and mathematical consistency."""
        with memory_efficient_computation("detect_anomalies"):
            # Check cache first
            state_key = self._get_state_key(state)
            if state_key in self._poly_cache:
                return self._poly_cache[state_key]
            
            anomalies = []

            # Get coefficients from detector
            coefficients = self.detector(state)
            if coefficients.dim() > 1:
                coefficients = coefficients.squeeze()

            # Check if this is a U(1) symmetry by verifying that the state preserves U(1) structure
            # For U(1), the transformed state should have constant magnitude and smooth phase variation
            is_u1 = False
            if state.is_complex():
                magnitudes = torch.abs(state)
                mean_mag = torch.mean(magnitudes)
                mag_variation = torch.std(magnitudes)
                # Check if magnitudes are approximately constant
                if mag_variation / mean_mag < 1e-3:
                    # Check if phases vary smoothly
                    phases = torch.angle(state)
                    phase_diffs = torch.diff(phases)
                    # Unwrap phase differences to [-π, π]
                    phase_diffs = torch.where(phase_diffs > np.pi, phase_diffs - 2*np.pi, phase_diffs)
                    phase_diffs = torch.where(phase_diffs < -np.pi, phase_diffs + 2*np.pi, phase_diffs)
                    # Check if phase differences are approximately constant
                    mean_diff = torch.mean(phase_diffs)
                    diff_variation = torch.std(phase_diffs)
                    if diff_variation / (torch.abs(mean_diff) + 1e-6) < 1e-2:
                        is_u1 = True

            # Process coefficients with symmetry preservation
            for degree in range(self.max_degree + 1):
                coeff_slice = coefficients[degree:]
                norm = torch.norm(coeff_slice)
                
                if norm > 1e-6:
                    # Project onto the space of consistent anomalies
                    coeff_slice = self._project_to_consistent_space(coeff_slice, degree)
                    
                    # Normalize after projection
                    norm = torch.norm(coeff_slice)
                    if norm > 0:
                        coeff_slice = coeff_slice / norm
                    
                    # Create anomaly polynomial with proper type
                    anomalies.append(
                        AnomalyPolynomial(
                            coefficients=coeff_slice.clone(),
                            variables=self.variables[: degree + 1],
                            degree=degree,
                            type="U1" if is_u1 else "general"  # Set type based on symmetry check
                        )
                    )

            # Ensure global consistency
            anomalies = self._ensure_global_consistency(anomalies)
            
            # Cache results
            self._poly_cache[state_key] = anomalies
            return anomalies

    def _project_to_consistent_space(self, coeffs: torch.Tensor, degree: int) -> torch.Tensor:
        """Project coefficients onto space of consistent anomalies."""
        n = coeffs.shape[0]
        result = torch.zeros_like(coeffs)
        
        # For U(1) symmetries, handle phase composition
        if coeffs.is_complex():
            # Get significant coefficients
            mask = coeffs.abs() > 1e-6
            if mask.any():
                # Get phases and magnitudes
                phases = torch.angle(coeffs[mask])
                magnitudes = coeffs[mask].abs()
                
                # Sort by magnitude to identify corresponding terms
                sorted_idx = torch.argsort(magnitudes, descending=True)
                sorted_phases = phases[sorted_idx]
                sorted_mags = magnitudes[sorted_idx]
                
                # For each coefficient
                for i in range(n):
                    if degree == 1:
                        # For degree 1, use antisymmetric phase composition
                        j = (i + 1) % n
                        phase = sorted_phases[i % len(sorted_phases)] - sorted_phases[j % len(sorted_phases)]
                        mag = torch.sqrt(sorted_mags[i % len(sorted_mags)] * sorted_mags[j % len(sorted_mags)])
                        result[i] = mag * torch.exp(1j * phase)
                    elif degree == 2:
                        # For degree 2, use cyclic phase composition
                        j = (i + 1) % n
                        k = (i - 1) % n
                        phase = sorted_phases[i % len(sorted_phases)] + sorted_phases[j % len(sorted_phases)] - sorted_phases[k % len(sorted_phases)]
                        mag = torch.pow(sorted_mags[i % len(sorted_mags)] * sorted_mags[j % len(sorted_mags)] * sorted_mags[k % len(sorted_mags)], 1/3)
                        result[i] = mag * torch.exp(1j * phase)
                    else:
                        # For higher degrees, use standard phase addition
                        for j in range(n):
                            for k in range(n):
                                if (j + k) % n == i:
                                    j_idx = j % len(sorted_phases) if j < len(sorted_phases) else 0
                                    k_idx = k % len(sorted_phases) if k < len(sorted_phases) else 0
                                    phase = sorted_phases[j_idx] + sorted_phases[k_idx]
                                    mag = torch.sqrt(sorted_mags[j_idx] * sorted_mags[k_idx])
                                    result[i] += mag * torch.exp(1j * phase)
                
                # Normalize result
                norm = torch.norm(result)
                if norm > 1e-6:
                    result = result / norm
            return result
        else:
            # For non-U(1) symmetries, use standard group operations
            if degree == 1:
                # Antisymmetric operation
                result = coeffs - torch.flip(coeffs, [0])
            elif degree == 2:
                # Cyclic operation
                for i in range(n):
                    result[i] = coeffs[i] + coeffs[(i+1) % n] - coeffs[(i-1) % n]
            else:
                # Standard cocycle condition
                for i in range(n):
                    for j in range(n):
                        k = (i + j) % n
                        result[k] = coeffs[j] + coeffs[(i+j) % n] - coeffs[i]
            
            # Normalize by number of terms
            norm = torch.norm(result)
            if norm > 1e-6:
                result = result / norm
            return result

    def _ensure_global_consistency(self, anomalies: List[AnomalyPolynomial]) -> List[AnomalyPolynomial]:
        """Ensure global consistency across all anomalies."""
        if not anomalies:
            return anomalies
            
        # Find maximum coefficient size
        max_size = max(a.coefficients.shape[-1] for a in anomalies)
        
        # Pad all coefficients to same size
        padded_coeffs = []
        for anomaly in anomalies:
            coeffs = anomaly.coefficients
            if coeffs.dim() == 1:
                coeffs = coeffs.unsqueeze(0)  # Add batch dimension if needed
                
            if coeffs.shape[-1] < max_size:
                padding_shape = list(coeffs.shape)
                padding_shape[-1] = max_size - coeffs.shape[-1]
                padding = torch.zeros(
                    padding_shape,
                    dtype=coeffs.dtype,
                    device=coeffs.device
                )
                padded = torch.cat([coeffs, padding], dim=-1)
            else:
                padded = coeffs[..., :max_size]
            padded_coeffs.append(padded)
            
        # Stack coefficients and compute global phase
        stacked = torch.cat(padded_coeffs, dim=0)
        
        # For U(1) symmetries, ensure proper phase composition
        if all(a.type == "U1" for a in anomalies):
            # Get significant coefficients
            mask = stacked.abs() > 1e-6
            if mask.any():
                # Get phases and magnitudes
                phases = torch.angle(stacked[mask])
                magnitudes = stacked[mask].abs()
                
                # Sort by magnitude to identify corresponding terms
                sorted_idx = torch.argsort(magnitudes, descending=True)
                sorted_phases = phases[sorted_idx]
                sorted_mags = magnitudes[sorted_idx]
                
                # For each anomaly
                start_idx = 0
                for i, anomaly in enumerate(anomalies):
                    end_idx = start_idx + (anomaly.coefficients.abs() > 1e-6).sum()
                    if end_idx > start_idx:
                        # Get phases for this anomaly
                        anomaly_phases = sorted_phases[start_idx:end_idx]
                        anomaly_mags = sorted_mags[start_idx:end_idx]
                        
                        # Compute phase differences
                        if len(anomaly_phases) > 1:
                            phase_diffs = torch.diff(anomaly_phases)
                            # Unwrap to [-π, π]
                            phase_diffs = torch.where(phase_diffs > np.pi, phase_diffs - 2*np.pi, phase_diffs)
                            phase_diffs = torch.where(phase_diffs < -np.pi, phase_diffs + 2*np.pi, phase_diffs)
                            # Use mean phase difference
                            mean_diff = torch.mean(phase_diffs)
                            
                            # Reconstruct phases
                            new_phases = torch.zeros_like(anomaly_phases)
                            new_phases[0] = anomaly_phases[0]  # Keep first phase
                            for j in range(1, len(new_phases)):
                                new_phases[j] = new_phases[j-1] + mean_diff
                                
                            # Apply new phases
                            stacked[mask][start_idx:end_idx] = anomaly_mags * torch.exp(1j * new_phases)
                    start_idx = end_idx
        
        # Normalize globally
        norms = torch.norm(stacked, dim=-1, keepdim=True)
        norms = torch.where(norms > 1e-6, norms, torch.ones_like(norms))
        stacked = stacked / norms
        
        # Create new anomalies with consistent coefficients
        result = []
        for i, anomaly in enumerate(anomalies):
            coeffs = stacked[i, :anomaly.coefficients.shape[-1]]
            if anomaly.coefficients.dim() == 1:
                coeffs = coeffs.squeeze(0)  # Remove batch dimension if original was 1D
            result.append(
                AnomalyPolynomial(
                    coefficients=coeffs.clone(),
                    variables=anomaly.variables,
                    degree=anomaly.degree,
                    type=anomaly.type
                )
            )
        
        return result

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
        gc.collect()

    def __del__(self):
        """Ensure proper cleanup."""
        self._cleanup()

