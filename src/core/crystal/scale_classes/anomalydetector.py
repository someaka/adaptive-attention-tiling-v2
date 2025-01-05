from dataclasses import dataclass
from typing import Callable, List, Tuple, Dict, Any, Union, Optional
import gc

import numpy as np
import torch
from torch import nn

from src.utils.memory_management_util import register_tensor
from src.core.crystal.scale_classes.complextanh import ComplexTanh
from src.core.crystal.scale_classes.memory_utils import memory_manager, memory_efficient_computation
from src.core.quantum.state_space import QuantumState, HilbertSpace
from src.core.patterns.operadic_structure import OperadicOperation
from src.core.patterns.operadic_handler import OperadicStructureHandler




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
    berry_phase: Optional[float] = None  # Berry phase for geometric verification

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
        if self.type == "U(1)":
            phases = torch.angle(self.coefficients[self.coefficients.abs() > 1e-6])
            if phases.numel() <= 1:  # Handle single coefficient case
                return True
            phase_diffs = torch.diff(phases)
            if phase_diffs.numel() == 0:  # Handle no phase differences
                return True
            
            # If we have a Berry phase, verify it matches winding
            if self.berry_phase is not None:
                berry_winding = float(self.berry_phase / (2 * torch.pi))
                if not torch.allclose(torch.tensor(self.winding_number), torch.tensor(berry_winding), rtol=1e-2):
                    return False
            
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
    """Detection of anomalies through direct mathematical structure."""

    def __init__(self, dim: int, max_degree: int = 4, dtype=torch.float32):
        self.dim = dim
        self.max_degree = max_degree  # Store max_degree as instance variable
        self.dtype = dtype

        # Initialize operadic structure for composition
        from src.core.patterns.operadic_structure import AttentionOperad
        self.operad = AttentionOperad(
            base_dim=dim,
            preserve_symplectic=True,
            preserve_metric=True,
            dtype=dtype
        )

        # Initialize operadic handler
        self.operadic_handler = OperadicStructureHandler(
            base_dim=dim,
            hidden_dim=dim,
            motive_rank=4,
            preserve_symplectic=True,
            preserve_metric=True,
            dtype=dtype
        )

        # Variable names for polynomials
        self.variables = [f"x_{i}" for i in range(dim)]

        # Add a dummy detector attribute for compatibility
        self.detector = nn.Sequential()

        # Initialize cache for polynomial computations
        self._poly_cache = {}

    def _get_state_key(self, state: torch.Tensor) -> str:
        """Get a unique key for caching state computations."""
        # Convert state to numpy for hashing
        state_np = state.detach().cpu().numpy()
        # Use string representation of rounded values for stable hashing
        rounded = np.round(state_np, decimals=6)
        return str(rounded.tobytes())

    def detect_anomalies(self, state: torch.Tensor) -> List[AnomalyPolynomial]:
        """Detect anomalies by finding obstructions to operadic composition."""
        # Handle batched input
        if state.dim() == 2:
            return [self._detect_single(s) for s in state]
        return [self._detect_single(state)]

    def _detect_single(self, state: torch.Tensor) -> AnomalyPolynomial:
        """Detect anomalies for a single state."""
        # Check cache first
        cache_key = self._get_state_key(state)
        if cache_key in self._poly_cache:
            return self._poly_cache[cache_key]

        # Normalize state
        state_norm = torch.norm(state)
        if state_norm > 1e-6:
            state = state / state_norm

        # Create composition law matrix from state vector
        # We use outer product to create a rank-1 matrix
        composition_law = torch.outer(state, state.conj())

        # Create basic operadic operation from state
        base_op = OperadicOperation(
            source_dim=self.dim,
            target_dim=self.dim,
            composition_law=composition_law
        )

        # Try to compose with itself - this reveals anomalies
        composed_op, metrics = self.operadic_handler.compose_operations(
            operations=[base_op, base_op],
            with_motivic=True
        )

        # Get composition law and its eigendecomposition
        comp_law = composed_op.composition_law
        eigenvals, eigenvecs = torch.linalg.eigh(comp_law)

        # Sort by absolute value
        idx = torch.argsort(torch.abs(eigenvals), descending=True)
        eigenvals = eigenvals[idx]
        eigenvecs = eigenvecs[:, idx]

        # Check for U(1) symmetry
        if state.is_complex():
            phases = torch.angle(state)
            phase_diffs = torch.diff(phases)
            phase_diffs = torch.where(phase_diffs > np.pi, phase_diffs - 2*np.pi, phase_diffs)
            phase_diffs = torch.where(phase_diffs < -np.pi, phase_diffs + 2*np.pi, phase_diffs)
            
            # If phase differences are approximately constant, we have U(1)
            if torch.std(phase_diffs) < 1e-2:
                # Project eigenvalues to consistent space
                coeffs = self._project_to_consistent_space(eigenvals, degree=1)
                result = AnomalyPolynomial(
                    coefficients=coeffs,
                    variables=self.variables[:len(coeffs)],
                    degree=1,  # U(1) anomalies are degree 1
                    type="U(1)",
                    winding_number=float(torch.sum(phase_diffs) / (2 * torch.pi))
                )
                self._poly_cache[cache_key] = result
                return result

        # Check for general anomalies through eigenvalue gaps
        gaps = torch.diff(torch.abs(eigenvals))
        significant_gaps = torch.where(gaps > 1e-2)[0]
        
        if len(significant_gaps) > 0:
            # Take the first significant gap
            end_idx = significant_gaps[0] + 1
            coeffs = eigenvals[:end_idx]
            
            # Project coefficients to consistent space
            coeffs = self._project_to_consistent_space(coeffs, degree=len(coeffs)-1)
            
            result = AnomalyPolynomial(
                coefficients=coeffs,
                variables=self.variables[:len(coeffs)],
                degree=len(coeffs) - 1,  # Degree is one less than number of coefficients
                type="general"
            )
            self._poly_cache[cache_key] = result
            return result

        # No significant anomalies found
        result = AnomalyPolynomial(
            coefficients=eigenvals[:1],  # Just take first eigenvalue
            variables=self.variables[:1],
            degree=0,
            type="trivial"
        )
        self._poly_cache[cache_key] = result
        return result

    def _project_to_consistent_space(self, coeffs: torch.Tensor, degree: int) -> torch.Tensor:
        """Project coefficients to a space consistent with the given degree.
        
        This ensures that:
        1. The number of coefficients matches the degree
        2. Coefficients with similar magnitudes are grouped together
        3. Phase relationships are preserved
        4. Normalization is preserved
        """
        # Ensure we have the right number of coefficients
        n_coeffs = degree + 1
        if len(coeffs) < n_coeffs:
            # Pad with zeros if needed
            coeffs = torch.nn.functional.pad(coeffs, (0, n_coeffs - len(coeffs)))
        elif len(coeffs) > n_coeffs:
            # Truncate if too many
            coeffs = coeffs[:n_coeffs]

        # Get magnitudes and phases
        mags = torch.abs(coeffs)
        phases = torch.angle(coeffs)

        # For single coefficients, just normalize and return
        if len(coeffs) <= 1:
            norm = torch.norm(coeffs)
            if norm > 1e-6:
                return coeffs / norm
            return coeffs

        # Calculate phase differences between consecutive coefficients
        phase_diffs = torch.diff(phases)
        # Normalize phase differences to [-π, π]
        phase_diffs = (phase_diffs + torch.pi) % (2 * torch.pi) - torch.pi
        # Use the first phase difference as the target
        target_phase_diff = phase_diffs[0]

        # Find groups of similar magnitudes
        groups = []
        used = set()
        
        # Sort by magnitude for stable grouping
        sorted_idx = torch.argsort(mags, descending=True)
        sorted_mags = mags[sorted_idx]

        # Group coefficients with similar magnitudes
        current_group = []
        current_mag = None
        threshold = 0.1  # 10% difference threshold

        for i, mag in enumerate(sorted_mags):
            if i in used:
                continue

            if current_mag is None:
                current_mag = mag
                current_group = [int(sorted_idx[i].item())]  # Convert to Python int
                used.add(i)
            else:
                ratio = mag / current_mag
                if abs(ratio - 1.0) < threshold:
                    current_group.append(int(sorted_idx[i].item()))  # Convert to Python int
                    used.add(i)
                else:
                    if len(current_group) > 0:
                        groups.append(current_group)
                    current_group = [int(sorted_idx[i].item())]  # Convert to Python int
                    current_mag = mag
                    used.add(i)

        if len(current_group) > 0:
            groups.append(current_group)

        # Create new coefficients with consistent magnitudes and phases
        new_coeffs = torch.zeros_like(coeffs)
        base_phase = phases[0]  # Use first coefficient's phase as base

        # Process each group
        for group in groups:
            # Use average magnitude for the group
            group_mags = mags[group]
            avg_mag = torch.mean(group_mags)
            
            # Assign phases with consistent differences
            for j, idx in enumerate(group):
                # Calculate phase based on position in sequence
                new_phase = base_phase + idx * target_phase_diff
                new_coeffs[idx] = avg_mag * torch.exp(1j * new_phase)

        # Normalize the coefficients
        norm = torch.norm(new_coeffs)
        if norm > 1e-6:
            new_coeffs = new_coeffs / norm

        return new_coeffs

    def compose_anomalies(self, anomalies1: List[AnomalyPolynomial], anomalies2: List[AnomalyPolynomial]) -> List[AnomalyPolynomial]:
        """Compose two lists of anomaly polynomials using operadic structure."""
        # If either list is empty, return empty list
        if not anomalies1 or not anomalies2:
            return []

        # Group anomalies by degree
        max_degree = max(max(a.degree for a in anomalies1), max(a.degree for a in anomalies2))
        result = []

        for degree in range(max_degree + 1):
            # Get anomalies of current degree
            a1s = [a for a in anomalies1 if a.degree == degree]
            a2s = [a for a in anomalies2 if a.degree == degree]

            if not a1s or not a2s:
                continue

            # For each pair of anomalies of the same degree
            for a1 in a1s:
                for a2 in a2s:
                    # Create operadic operations from coefficients
                    # Pad coefficients to match dimensions
                    coeffs1 = torch.nn.functional.pad(a1.coefficients, (0, self.dim - len(a1.coefficients)))
                    coeffs2 = torch.nn.functional.pad(a2.coefficients, (0, self.dim - len(a2.coefficients)))

                    # Ensure coefficients have correct dtype
                    coeffs1 = coeffs1.to(dtype=self.dtype)
                    coeffs2 = coeffs2.to(dtype=self.dtype)

                    op1 = OperadicOperation(
                        source_dim=self.dim,
                        target_dim=self.dim,
                        composition_law=torch.diag(coeffs1)
                    )
                    op2 = OperadicOperation(
                        source_dim=self.dim,
                        target_dim=self.dim,
                        composition_law=torch.diag(coeffs2)
                    )

                    # Compose operations
                    composed_op, _ = self.operadic_handler.compose_operations(
                        operations=[op1, op2],
                        with_motivic=True
                    )

                    # Extract diagonal coefficients
                    coeffs = torch.diagonal(composed_op.composition_law)

                    # Project to consistent space
                    coeffs = self._project_to_consistent_space(coeffs, degree=degree)

                    # Create new anomaly polynomial
                    result.append(AnomalyPolynomial(
                        coefficients=coeffs,
                        variables=self.variables[:len(coeffs)],
                        degree=degree,
                        type="composed"
                    ))

        return result

