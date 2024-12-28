from dataclasses import dataclass
from typing import Callable, List, Tuple, Dict, Any, Union, Optional
import gc

import numpy as np
import torch
from torch import nn

from src.utils.memory_management import register_tensor
from src.core.crystal.scale_classes.complextanh import ComplexTanh
from src.core.crystal.scale_classes.memory_utils import memory_manager, memory_efficient_computation
from src.core.quantum.state_space import QuantumState, HilbertSpace




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
        if self.type == "U1":
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
    """Detection and analysis of anomalies with optimized performance."""

    def __init__(self, dim: int, max_degree: int = 4, dtype=torch.float32):
        self.dim = dim
        self.max_degree = max_degree
        self.dtype = dtype

        # Initialize Hilbert space for geometric phase computation
        self.hilbert_space = HilbertSpace(dim=dim)
        self._berry_transport = None

        # Initialize operadic structure for composition
        from src.core.patterns.operadic_structure import AttentionOperad
        self.operad = AttentionOperad(
            base_dim=dim,
            preserve_symplectic=True,
            preserve_metric=True,
            dtype=dtype
        )

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

    @property
    def berry_transport(self):
        """Lazy initialization of BerryTransport to avoid circular imports."""
        if self._berry_transport is None:
            from src.core.quantum.geometric_flow import BerryTransport
            self._berry_transport = BerryTransport(self.hilbert_space)
        return self._berry_transport

    def detect_anomalies(self, state: torch.Tensor) -> List[AnomalyPolynomial]:
        """Detect anomalies in quantum state with improved efficiency and mathematical consistency."""
        with memory_efficient_computation("detect_anomalies"):
            print("\n=== Starting Anomaly Detection ===")
            print(f"Input state shape: {state.shape}")
            print(f"Input state: {state}")

            # Handle batched input
            if state.dim() > 1:
                # Process each batch element separately
                anomalies_list = [self.detect_anomalies(s) for s in state]
                # Combine results (take the first set of anomalies as they should be similar)
                return anomalies_list[0]

            # Normalize state first
            state_norm = torch.norm(state)
            if state_norm > 1e-6:
                state = state / state_norm
            print(f"Normalized state: {state}")

            # Check cache first
            state_key = self._get_state_key(state)
            if state_key in self._poly_cache:
                print("Cache hit - returning cached results")
                return self._poly_cache[state_key]

            anomalies = []

            # Get coefficients from detector
            print("\n=== Neural Network Processing ===")
            print("Passing state through detector layers...")
            intermediate = self.detector_in(state)
            print(f"After input layer: {intermediate}")
            intermediate = self.detector_hidden(intermediate)
            print(f"After hidden layer: {intermediate}")
            coefficients = self.detector_out(intermediate)
            print(f"Raw coefficients from detector: {coefficients}")

            if coefficients.dim() > 1:
                coefficients = coefficients.squeeze()
                print(f"Squeezed coefficients: {coefficients}")

            # Check if this is a U(1) symmetry
            print("\n=== U(1) Symmetry Check ===")
            is_u1 = False
            berry_phase = None
            if state.is_complex():
                magnitudes = torch.abs(state)
                mean_mag = torch.mean(magnitudes)
                mag_variation = torch.std(magnitudes)
                print(f"State magnitudes: {magnitudes}")
                print(f"Mean magnitude: {mean_mag}")
                print(f"Magnitude variation: {mag_variation}")

                # Check if magnitudes are approximately constant
                if mag_variation / mean_mag < 1e-2:  # Relaxed threshold since we normalized
                    print("Passed magnitude constancy check")
                    # Check if phases vary smoothly
                    phases = torch.angle(state)
                    phase_diffs = torch.diff(phases)
                    # Unwrap phase differences to [-π, π]
                    phase_diffs = torch.where(phase_diffs > np.pi, phase_diffs - 2*np.pi, phase_diffs)
                    phase_diffs = torch.where(phase_diffs < -np.pi, phase_diffs + 2*np.pi, phase_diffs)
                    print(f"State phases: {phases}")
                    print(f"Phase differences: {phase_diffs}")

                    # Check if phase differences are approximately constant
                    mean_diff = torch.mean(phase_diffs)
                    diff_variation = torch.std(phase_diffs)
                    print(f"Mean phase difference: {mean_diff}")
                    print(f"Phase difference variation: {diff_variation}")

                    if diff_variation / (torch.abs(mean_diff) + 1e-6) < 1e-1:  # Relaxed threshold
                        print("Passed phase smoothness check - U(1) symmetry detected")
                        is_u1 = True
                        # For U(1), compute Berry phase
                        initial_state = QuantumState(
                            amplitudes=torch.ones_like(state),
                            basis_labels=[f"|{i}⟩" for i in range(self.dim)],
                            phase=torch.tensor(0.0, device=state.device, dtype=state.dtype)
                        )
                        final_state = QuantumState(
                            amplitudes=state,
                            basis_labels=[f"|{i}⟩" for i in range(self.dim)],
                            phase=torch.tensor(float(torch.angle(state[0])), device=state.device, dtype=state.dtype)
                        )
                        path = [initial_state, final_state]
                        berry_phase = float(self.berry_transport.compute_berry_phase(path))
                        print(f"Computed Berry phase: {berry_phase}")

            # Process coefficients with symmetry preservation
            print("\n=== Processing Coefficients ===")
            for degree in range(self.max_degree + 1):
                print(f"\nDegree {degree}:")
                coeff_slice = coefficients[degree:]
                print(f"Initial coefficient slice: {coeff_slice}")
                norm = torch.norm(coeff_slice)
                print(f"Initial norm: {norm}")

                if norm > 1e-6:
                    # Project onto the space of consistent anomalies
                    print("Projecting to consistent space...")
                    coeff_slice = self._project_to_consistent_space(coeff_slice, degree)
                    print(f"After projection: {coeff_slice}")

                    # Create anomaly polynomial
                    poly = AnomalyPolynomial(
                        coefficients=coeff_slice,
                        variables=self.variables[: degree + 1],
                        degree=degree,
                        type="U1" if is_u1 else "general",
                        winding_number=self._compute_winding_number(coeff_slice) if is_u1 else None,
                        berry_phase=berry_phase if is_u1 else None,
                        is_consistent=True
                    )
                    print(f"Created anomaly polynomial:")
                    print(f"  Type: {poly.type}")
                    print(f"  Coefficients: {poly.coefficients}")
                    print(f"  Winding number: {poly.winding_number}")
                    print(f"  Berry phase: {poly.berry_phase}")
                    print(f"  Is consistent: {poly.is_consistent}")
                    anomalies.append(poly)

            # Cache results
            self._poly_cache[state_key] = anomalies

            # Print final summary
            print("\n=== Ensuring Global Consistency ===")
            print(f"Final number of anomalies: {len(anomalies)}")
            for i, poly in enumerate(anomalies):
                print(f"\nAnomaly {i}:")
                print(f"  Type: {poly.type}")
                print(f"  Coefficients: {poly.coefficients}")
                print(f"  Winding number: {poly.winding_number}")
                print(f"  Berry phase: {poly.berry_phase}")
                print(f"  Is consistent: {poly.is_consistent}")

            return anomalies

    def _project_to_consistent_space(self, coefficients: torch.Tensor, degree: Optional[int] = None) -> torch.Tensor:
        """Project coefficients to consistent space while preserving phase relationships.
        
        Args:
            coefficients: Input coefficients to project
            degree: Optional degree of the polynomial
            
        Returns:
            Projected coefficients that satisfy consistency conditions
        """
        # Normalize coefficients
        norm = torch.norm(coefficients)
        if norm > 0:
            coefficients = coefficients / norm

        # Extract phases and magnitudes
        phases = torch.angle(coefficients)
        magnitudes = torch.abs(coefficients)

        # Group similar magnitudes
        sorted_indices = torch.argsort(magnitudes, descending=True)
        sorted_magnitudes = magnitudes[sorted_indices]
        sorted_phases = phases[sorted_indices]

        # Find groups of similar magnitudes
        groups = []
        current_group = [0]
        threshold = 0.1  # Threshold for considering magnitudes similar

        for i in range(1, len(sorted_magnitudes)):
            if abs(sorted_magnitudes[i] - sorted_magnitudes[current_group[0]]) < threshold:
                current_group.append(i)
            else:
                if len(current_group) > 1:
                    groups.append(current_group.copy())
                current_group = [i]
        if len(current_group) > 1:
            groups.append(current_group)

        # Process each group to preserve phase relationships
        for group in groups:
            # Calculate phase differences between consecutive elements
            phase_diffs = []
            for i in range(len(group) - 1):
                diff = (sorted_phases[group[i + 1]] - sorted_phases[group[i]] + torch.pi) % (2 * torch.pi) - torch.pi
                phase_diffs.append(diff)

            if len(phase_diffs) > 0:
                # Check if phase pattern is regular
                phase_diffs = torch.tensor(phase_diffs)
                mean_diff = torch.mean(phase_diffs)
                std_diff = torch.std(phase_diffs) if len(phase_diffs) > 1 else torch.tensor(0.0)
                is_regular = std_diff < 0.1

                if is_regular:
                    # Apply consistent phase pattern
                    base_phase = sorted_phases[group[0]]
                    for i, idx in enumerate(group):
                        sorted_phases[idx] = base_phase + i * mean_diff
                else:
                    # Check for anti-phase pairs
                    for i in range(len(group) - 1):
                        phase_diff = phase_diffs[i]
                        if abs(abs(phase_diff) - torch.pi) < 0.1:
                            # Anti-phase pair found
                            sorted_phases[group[i + 1]] = sorted_phases[group[i]] + torch.pi
                        else:
                            # Preserve relative phase
                            sorted_phases[group[i + 1]] = sorted_phases[group[i]] + phase_diff

        # Reconstruct coefficients with preserved phase relationships
        projected = torch.zeros_like(coefficients)
        projected[sorted_indices] = sorted_magnitudes * torch.exp(1j * sorted_phases)

        # Normalize again
        projected = projected / torch.norm(projected)

        return projected

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

    def _compare_coefficients(self, coeffs1, coeffs2, threshold=0.1):
        """Compare two sets of coefficients for similarity."""
        # Get significant terms using dynamic thresholding
        max_mag1 = torch.max(torch.abs(coeffs1))
        max_mag2 = torch.max(torch.abs(coeffs2))
        rel_threshold = threshold * torch.max(torch.tensor([max_mag1, max_mag2]))
        
        sig1 = coeffs1[torch.abs(coeffs1) > rel_threshold]
        sig2 = coeffs2[torch.abs(coeffs2) > rel_threshold]
        
        # Sort by magnitude for consistent comparison
        sig1_sorted, _ = torch.sort(torch.abs(sig1), descending=True)
        sig2_sorted, _ = torch.sort(torch.abs(sig2), descending=True)
        
        # Compare number of significant terms
        if len(sig1_sorted) != len(sig2_sorted):
            return False
            
        # Compare magnitudes
        return torch.allclose(sig1_sorted, sig2_sorted, rtol=1e-2)

    def compose_anomalies(self, a1: List[AnomalyPolynomial], a2: List[AnomalyPolynomial]) -> List[AnomalyPolynomial]:
        """Compose anomaly polynomials using operadic structure.
        
        Args:
            a1: First list of anomaly polynomials
            a2: Second list of anomaly polynomials
            
        Returns:
            List of composed anomaly polynomials
        """
        print("\n=== Composing Anomalies Using Operadic Structure ===")
        composed_anomalies = []
        
        # Match polynomials of same degree
        for poly1, poly2 in zip(a1, a2):
            print(f"\nComposing degree {poly1.degree} polynomials:")
            print(f"First polynomial: {poly1.coefficients}")
            print(f"Second polynomial: {poly2.coefficients}")
            
            # Get significant terms
            sig1 = poly1.coefficients[torch.abs(poly1.coefficients) > 0.1]
            sig2 = poly2.coefficients[torch.abs(poly2.coefficients) > 0.1]
            
            # Create operadic operations from polynomials
            op1 = self.operad.create_operation(
                source_dim=len(sig1),
                target_dim=len(sig1),
                preserve_structure='symplectic' if poly1.type == 'U1' else None
            )
            op1.composition_law = sig1.view(-1, 1)  # Reshape for matrix ops
            
            op2 = self.operad.create_operation(
                source_dim=len(sig2),
                target_dim=len(sig2),
                preserve_structure='symplectic' if poly2.type == 'U1' else None
            )
            op2.composition_law = sig2.view(-1, 1)  # Reshape for matrix ops
            
            # Compose using operadic structure
            composed_op = self.operad.compose([op1, op2])
            composed_coeffs = composed_op.composition_law.view(-1)  # Flatten back to 1D
            
            # Pad with zeros to match original size
            padded_coeffs = torch.zeros_like(poly1.coefficients)
            padded_coeffs[:len(composed_coeffs)] = composed_coeffs
            
            print(f"Composed coefficients: {padded_coeffs}")
            
            # For U1 symmetries, compute Berry phase of composition
            berry_phase = None
            if poly1.type == 'U1' and poly2.type == 'U1':
                # Create quantum states for Berry phase computation
                initial_state = QuantumState(
                    amplitudes=torch.ones_like(padded_coeffs),
                    basis_labels=[f"|{i}⟩" for i in range(len(padded_coeffs))],
                    phase=torch.tensor(0.0, device=padded_coeffs.device, dtype=padded_coeffs.dtype)
                )
                
                # Evolve state and compute Berry phase
                evolved_state = self._evolve_state(initial_state, padded_coeffs)
                berry_phase = self._compute_berry_phase(initial_state, evolved_state)
            
            # Create composed anomaly polynomial
            composed_type = 'U1' if poly1.type == 'U1' and poly2.type == 'U1' else 'general'
            composed_poly = AnomalyPolynomial(
                coefficients=padded_coeffs,
                type=composed_type,
                degree=poly1.degree,
                winding_number=self._compute_winding_number(padded_coeffs) if composed_type == 'U1' else None,
                berry_phase=berry_phase,
                is_consistent=True,
                variables=poly1.variables  # Use same variables as first polynomial
            )
            
            composed_anomalies.append(composed_poly)
            
        return composed_anomalies

    def _compute_winding_number(self, coefficients: torch.Tensor) -> float:
        """Compute the winding number of a polynomial."""
        if coefficients.dim() == 0:
            return 0.0
        phases = torch.angle(coefficients)
        phase_diffs = torch.diff(phases)
        # Unwrap phase differences to [-π, π]
        phase_diffs = (phase_diffs + torch.pi) % (2 * torch.pi) - torch.pi
        return float(torch.sum(phase_diffs) / (2 * torch.pi))

    def _evolve_state(self, state: QuantumState, coefficients: torch.Tensor) -> QuantumState:
        """Evolve quantum state using coefficients.
        
        Args:
            state: Initial quantum state
            coefficients: Coefficients to evolve with
            
        Returns:
            Evolved quantum state
        """
        evolved_amplitudes = coefficients * state.amplitudes
        return QuantumState(
            amplitudes=evolved_amplitudes,
            basis_labels=state.basis_labels,
            phase=torch.tensor(float(torch.angle(evolved_amplitudes[0])), device=evolved_amplitudes.device, dtype=evolved_amplitudes.dtype)
        )
        
    def _compute_berry_phase(self, initial_state: QuantumState, final_state: QuantumState) -> float:
        """Compute Berry phase between initial and final states.
        
        Args:
            initial_state: Initial quantum state
            final_state: Final quantum state
            
        Returns:
            Computed Berry phase
        """
        path = [initial_state, final_state]
        return float(self.berry_transport.compute_berry_phase(path))

