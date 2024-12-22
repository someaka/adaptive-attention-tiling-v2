"""Pattern formation module.

This module implements pattern formation dynamics and analysis tools.
It integrates symplectic geometry, fiber bundles, and quantum geometric tensors
for analyzing pattern dynamics and bifurcations.
"""

import torch
import numpy as np
from typing import Dict, Any, Optional, List, Tuple, Union, Protocol
from dataclasses import dataclass
import logging

from .symplectic import SymplecticStructure, SymplecticForm
from .operadic_structure import AttentionOperad, EnrichedAttention

logger = logging.getLogger(__name__)

@dataclass
class BifurcationMetrics:
    """Metrics for bifurcation analysis."""
    stability_margin: float
    max_eigenvalue: float
    symplectic_invariant: float
    quantum_metric: torch.Tensor
    pattern_height: float
    geometric_flow: torch.Tensor

class BifurcationAnalyzer:
    """Analyzer for bifurcation points in pattern dynamics with geometric structure."""
    
    def __init__(
        self,
        threshold: float = 0.1,
        window_size: int = 10,
        symplectic: Optional[SymplecticStructure] = None,
        preserve_structure: bool = True,
        wave_enabled: bool = True,
        dtype: torch.dtype = torch.float32
    ):
        """Initialize bifurcation analyzer with geometric structure.
        
        Args:
            threshold: Threshold for detecting bifurcations
            window_size: Window size for temporal analysis
            symplectic: Optional symplectic structure for Hamiltonian dynamics
            preserve_structure: Whether to preserve geometric structure
            wave_enabled: Whether to enable wave packet evolution
            dtype: Data type for tensors
        """
        self.threshold = threshold
        self.window_size = window_size
        self.preserve_structure = preserve_structure
        self.wave_enabled = wave_enabled
        self.dtype = dtype
        
        # Initialize geometric structures
        if symplectic is None:
            # Default to 2D symplectic structure if none provided
            self.symplectic = SymplecticStructure(
                dim=2,
                preserve_structure=preserve_structure,
                wave_enabled=wave_enabled,
                dtype=dtype
            )
        else:
            self.symplectic = symplectic
            
        # Initialize enriched structure for wave behavior
        self.enriched = EnrichedAttention(
            base_category="SymplecticVect",
            wave_enabled=wave_enabled,
            dtype=dtype
        )
        self.enriched.wave_enabled = wave_enabled
        
        # Initialize operadic structure for transitions
        self.operadic = AttentionOperad(
            base_dim=2,
            preserve_symplectic=preserve_structure,
            preserve_metric=True,
            dtype=dtype
        )
        
    def detect_bifurcations(
        self,
        pattern: torch.Tensor,
        parameter: torch.Tensor
    ) -> List[float]:
        """Detect bifurcation points in pattern evolution.
        
        Args:
            pattern: Pattern evolution tensor [time, ...]
            parameter: Control parameter values
            
        Returns:
            List of bifurcation points
        """
        logger.info("Starting bifurcation detection")
        batch_size = len(parameter)
        pattern_flat = pattern.reshape(batch_size, -1)
        
        # Pre-allocate arrays for metrics
        stability_margins = torch.zeros(batch_size, device=pattern.device)
        max_eigenvalues = torch.zeros(batch_size, device=pattern.device)
        symplectic_invariants = torch.zeros(batch_size, device=pattern.device)
        quantum_metrics = []
        pattern_heights = torch.zeros(batch_size, device=pattern.device)
        geometric_flows = []
        
        # Process patterns in chunks for better memory efficiency
        chunk_size = min(batch_size, 16)  # Reduced chunk size
        
        logger.info(f"Processing {batch_size} patterns in chunks of {chunk_size}")
        
        # Use torch.no_grad() for inference
        with torch.no_grad():
            try:
                for i in range(0, batch_size, chunk_size):
                    end_idx = min(i + chunk_size, batch_size)
                    chunk_patterns = pattern_flat[i:end_idx]
                    
                    logger.debug(f"Processing chunk {i//chunk_size + 1}/{(batch_size-1)//chunk_size + 1}")
                    
                    # Compute metrics for chunk
                    chunk_metrics = []
                    for p in chunk_patterns:
                        try:
                            metrics = self._compute_stability_metrics(p)
                            chunk_metrics.append(metrics)
                        except Exception as e:
                            logger.error(f"Error computing metrics for pattern: {e}")
                            # Use default metrics on error
                            chunk_metrics.append(BifurcationMetrics(
                                stability_margin=0.0,
                                max_eigenvalue=0.0,
                                symplectic_invariant=0.0,
                                quantum_metric=torch.zeros(1),
                                pattern_height=0.0,
                                geometric_flow=torch.zeros(1)
                            ))
                    
                    # Extract metrics
                    for j, metrics in enumerate(chunk_metrics):
                        idx = i + j
                        stability_margins[idx] = metrics.stability_margin
                        max_eigenvalues[idx] = metrics.max_eigenvalue
                        symplectic_invariants[idx] = metrics.symplectic_invariant
                        quantum_metrics.append(metrics.quantum_metric)
                        pattern_heights[idx] = metrics.pattern_height
                        geometric_flows.append(metrics.geometric_flow)
                        
                logger.info("Finished processing all patterns")
                
                # Detect bifurcations using vectorized operations
                # Compute differences in metrics
                margin_diff = torch.abs(stability_margins[1:] - stability_margins[:-1])
                eigenval_diff = torch.abs(max_eigenvalues[1:] - max_eigenvalues[:-1])
                invariant_diff = torch.abs(symplectic_invariants[1:] - symplectic_invariants[:-1])
                height_diff = torch.abs(pattern_heights[1:] - pattern_heights[:-1])
                
                # Combine differences with weights
                total_diff = (
                    margin_diff * 0.3 +
                    eigenval_diff * 0.3 +
                    invariant_diff * 0.2 +
                    height_diff * 0.2
                )
                
                # Find points where difference exceeds threshold
                bifurcation_mask = total_diff > self.threshold
                bifurcation_indices = torch.nonzero(bifurcation_mask).squeeze(-1)
                
                # Convert to parameter values
                bifurcations = [float(parameter[i+1].item()) for i in bifurcation_indices]
                
                logger.info(f"Found {len(bifurcations)} bifurcation points")
                return bifurcations
                
            except Exception as e:
                logger.error(f"Error in bifurcation detection: {e}")
                return []
        
    def _compute_stability_metrics(
        self,
        pattern: torch.Tensor
    ) -> BifurcationMetrics:
        """Compute stability metrics with geometric structure.
        
        Args:
            pattern: Pattern tensor
            
        Returns:
            BifurcationMetrics containing stability information
        """
        logger.debug("Computing stability metrics")
        
        # Handle pattern dimensions through symplectic structure
        pattern_symplectic = self.symplectic._handle_dimension(pattern)
        
        # Compute temporal derivatives with structure preservation
        if pattern.dim() > 1:
            # Use finite differences instead of gradient for speed
            grad = (pattern_symplectic[1:] - pattern_symplectic[:-1])
            mean_rate = torch.mean(torch.abs(grad)).item()
            max_rate = torch.max(torch.abs(grad)).item()
        else:
            mean_rate = 0.0
            max_rate = 0.0
            
        # Compute quantum geometric tensor (cache result for reuse)
        Q = self.symplectic.compute_quantum_geometric_tensor(pattern_symplectic)
        g = Q.real  # Metric part
        omega = Q.imag  # Symplectic part
        
        # Compute symplectic invariant
        symplectic_invariant = torch.abs(
            torch.einsum('...i,...ij,...j->...', pattern_symplectic, omega, pattern_symplectic)
        ).item()
        
        # Compute pattern height using cached tensor
        pattern_height = torch.norm(pattern_symplectic).item()
        
        # Compute geometric flow with reduced steps but maintained accuracy
        # Use adaptive step size based on pattern magnitude
        pattern_norm = pattern_height
        dt = min(0.1, 1.0 / (1.0 + pattern_norm))
        steps = 5  # Reduced number of steps
        
        flow = self.symplectic.quantum_ricci_flow(
            pattern_symplectic,
            time=0.5,  # Reduced total time
            dt=dt,
            steps=steps
        )
        
        logger.debug("Finished computing stability metrics")
        
        return BifurcationMetrics(
            stability_margin=mean_rate,
            max_eigenvalue=max_rate,
            symplectic_invariant=symplectic_invariant,
            quantum_metric=g,
            pattern_height=pattern_height,
            geometric_flow=flow
        )
        
    def _is_bifurcation(
        self,
        metrics1: BifurcationMetrics,
        metrics2: BifurcationMetrics
    ) -> bool:
        """Determine if there is a bifurcation between two sets of metrics.
        
        Args:
            metrics1: First set of metrics
            metrics2: Second set of metrics
            
        Returns:
            True if bifurcation detected, False otherwise
        """
        # Compute weighted differences
        margin_diff = abs(metrics2.stability_margin - metrics1.stability_margin)
        eigenval_diff = abs(metrics2.max_eigenvalue - metrics1.max_eigenvalue)
        invariant_diff = abs(metrics2.symplectic_invariant - metrics1.symplectic_invariant)
        height_diff = abs(metrics2.pattern_height - metrics1.pattern_height)
        
        # Combine differences with weights
        total_diff = (
            margin_diff * 0.3 +
            eigenval_diff * 0.3 +
            invariant_diff * 0.2 +
            height_diff * 0.2
        )
        
        return total_diff > self.threshold
        
    def analyze_stability(
        self,
        pattern: torch.Tensor,
        parameter_range: Tuple[float, float],
        num_points: int = 100
    ) -> Dict[str, Any]:
        """Analyze pattern stability across parameter range with geometric structure.
        
        Args:
            pattern: Initial pattern state
            parameter_range: Range of parameter values
            num_points: Number of points to sample
            
        Returns:
            Dictionary with stability analysis results
        """
        # Generate parameter values
        parameters = torch.linspace(
            parameter_range[0],
            parameter_range[1],
            num_points
        )
        
        # Evolve pattern across parameter range with structure preservation
        evolution = []
        for param in parameters:
            state = self._evolve_pattern(pattern, param)
            evolution.append(state)
            
        evolution = torch.stack(evolution)
        
        # Find bifurcation points using geometric structure
        bifurcations = self.detect_bifurcations(evolution, parameters)
        
        # Compute stability metrics with geometric structure
        stability = []
        for state in evolution:
            metrics = self._compute_stability_metrics(state)
            stability.append(metrics)
            
        return {
            "bifurcation_points": bifurcations,
            "stability_metrics": stability,
            "parameter_values": parameters,
            "evolution": evolution
        }
        
    def _evolve_pattern(
        self,
        pattern: torch.Tensor,
        parameter: Union[float, torch.Tensor],
        time_steps: int = 100
    ) -> torch.Tensor:
        """Evolve pattern with geometric structure preservation.
        
        Args:
            pattern: Initial pattern state
            parameter: Evolution parameter
            time_steps: Number of time steps
            
        Returns:
            Evolved pattern state
        """
        if isinstance(parameter, torch.Tensor):
            parameter = parameter.item()
            
        # Handle pattern dimensions
        pattern_symplectic = self.symplectic._handle_dimension(pattern)
        
        if self.wave_enabled:
            # Create wave packet
            n = pattern_symplectic.shape[-1] // 2
            position = pattern_symplectic[..., :n]
            momentum = pattern_symplectic[..., n:]
            wave_packet = self.enriched.create_wave_packet(position, momentum)
            
            # Evolve through wave operator
            evolved = self.enriched.create_morphism(
                pattern=wave_packet,
                operation=self.operadic.create_operation(
                    source_dim=wave_packet.shape[-1],
                    target_dim=pattern_symplectic.shape[-1],
                    preserve_structure='symplectic'
                ),
                include_wave=True
            )
        else:
            # Direct evolution with structure preservation
            evolved = pattern_symplectic + parameter * torch.randn_like(pattern_symplectic)
            
        # Verify structure preservation
        if self.preserve_structure:
            form_before = self.symplectic.compute_form(pattern_symplectic)
            form_after = self.symplectic.compute_form(evolved)
            if not torch.allclose(
                form_before.evaluate(pattern_symplectic, pattern_symplectic),
                form_after.evaluate(evolved, evolved),
                rtol=1e-5
            ):
                raise ValueError("Symplectic structure not preserved during evolution")
                
        return evolved

class PatternFormation:
    """Pattern formation dynamics with geometric structure.
    
    This class implements pattern formation through reaction-diffusion dynamics
    while preserving geometric structure (symplectic form, fiber bundle structure).
    It integrates:
    1. Symplectic geometry for Hamiltonian dynamics
    2. Wave packet evolution for quantum behavior
    3. Geometric flow for pattern evolution
    4. Structure preservation for stability
    """
    
    def __init__(
        self, 
        dim: int = 3,
        dt: float = 0.1,
        diffusion_coeff: float = 0.1,
        reaction_coeff: float = 1.0,
        symplectic: Optional[SymplecticStructure] = None,
        preserve_structure: bool = True,
        wave_enabled: bool = True
    ):
        """Initialize pattern formation with geometric structure.
        
        Args:
            dim: Dimension of pattern space
            dt: Time step for integration
            diffusion_coeff: Diffusion coefficient
            reaction_coeff: Reaction coefficient
            symplectic: Optional symplectic structure
            preserve_structure: Whether to preserve structure
            wave_enabled: Whether to enable wave behavior
        """
        self.dim = dim
        self.dt = dt
        self.diffusion_coeff = diffusion_coeff
        self.reaction_coeff = reaction_coeff
        self.preserve_structure = preserve_structure
        self.wave_enabled = wave_enabled
        
        # Initialize geometric structures
        if symplectic is None:
            self.symplectic = SymplecticStructure(
                dim=dim,
                preserve_structure=preserve_structure,
                wave_enabled=wave_enabled
            )
        else:
            self.symplectic = symplectic
            
        # Initialize enriched structure
        self.enriched = EnrichedAttention()
        self.enriched.wave_enabled = wave_enabled
        
        # Initialize operadic structure
        self.operadic = AttentionOperad(
            base_dim=dim,
            preserve_symplectic=preserve_structure,
            preserve_metric=True
        )
        
        # Initialize diffusion kernel with structure preservation
        self.diffusion_kernel = self._create_structured_kernel()
        
    def _create_structured_kernel(self) -> torch.Tensor:
        """Create diffusion kernel that preserves geometric structure."""
        # Basic kernel [0.2, 0.6, 0.2]
        kernel = torch.tensor([[[0.2, 0.6, 0.2]]], dtype=torch.cfloat)

        # Ensure kernel preserves symplectic structure
        if self.preserve_structure:
            form = self.symplectic.compute_form(kernel)
            # Reshape kernel to match form matrix dimensions
            kernel = kernel.view(-1)  # Flatten to 1D
            form_matrix = form.matrix.view(form.matrix.shape[-2:])  # Get 2D matrix
            # Pad or truncate kernel to match form matrix size
            target_size = form_matrix.shape[0]
            if kernel.shape[0] < target_size:
                padding = torch.zeros(target_size - kernel.shape[0], device=kernel.device, dtype=kernel.dtype)
                kernel = torch.cat([kernel, padding])
            else:
                kernel = kernel[:target_size]
            # Reshape for matrix multiplication
            kernel = kernel.view(1, -1)
            # Apply form matrix
            kernel = torch.matmul(kernel, form_matrix)
            # Reshape back to original format
            kernel = kernel.view(1, 1, -1)
            # Convert back to real tensor
            kernel = torch.real(kernel)

        return kernel
        
    def _project_orthogonal(self, matrix: torch.Tensor) -> torch.Tensor:
        """Project matrix to orthogonal group O(n) using polar decomposition.
        
        Args:
            matrix: Input matrix to project
            
        Returns:
            Projected orthogonal matrix
        """
        # Compute polar decomposition
        U, S, V = torch.linalg.svd(matrix)
        # Return orthogonal factor
        return torch.matmul(U, V)
        
    def evolve(
        self,
        pattern: torch.Tensor,
        time_steps: int
    ) -> torch.Tensor:
        """Evolve pattern according to reaction-diffusion dynamics with structure preservation.
        
        Args:
            pattern: Initial pattern tensor of shape (batch_size, dim)
            time_steps: Number of time steps to evolve
            
        Returns:
            torch.Tensor: Evolved pattern trajectory of shape (batch_size, time_steps, dim)
        """
        batch_size = pattern.size(0)
        
        # Handle pattern dimensions through symplectic structure
        pattern_symplectic = self.symplectic._handle_dimension(pattern)
        
        # Initialize trajectory tensor
        trajectory = torch.zeros(
            batch_size,
            time_steps,
            pattern_symplectic.shape[-1],
            device=pattern.device,
            dtype=pattern.dtype
        )
        trajectory[:, 0] = pattern_symplectic.clone()
        
        # Initialize wave packet if enabled
        if self.wave_enabled:
            n = pattern_symplectic.shape[-1] // 2
            position = pattern_symplectic[..., :n]
            momentum = pattern_symplectic[..., n:]
            wave_packet = self.enriched.create_wave_packet(position, momentum)
            trajectory[:, 0] = wave_packet.view(batch_size, -1).clone()
        
        # Evolve pattern with structure preservation
        for t in range(1, time_steps):
            # Compute quantum geometric tensor
            Q = self.symplectic.compute_quantum_geometric_tensor(trajectory[:, t-1])
            g = Q.real  # Metric part
            omega = Q.imag  # Symplectic part
            
            # Diffusion term with structure preservation
            current = trajectory[:, t-1].view(batch_size, 1, -1)  # Reshape for conv1d
            diffusion = torch.nn.functional.conv1d(
                current,
                self.diffusion_kernel,
                padding=1
            ).view(batch_size, -1)
            
            # Ensure dimensions match
            if diffusion.shape[-1] != trajectory.shape[-1]:
                target_size = trajectory.shape[-1]
                if diffusion.shape[-1] < target_size:
                    padding = torch.zeros(batch_size, target_size - diffusion.shape[-1],
                                        device=diffusion.device, dtype=diffusion.dtype)
                    diffusion = torch.cat([diffusion, padding], dim=-1)
                else:
                    diffusion = diffusion[..., :target_size]
            
            # Reaction term with structure preservation
            reaction = self.reaction_coeff * trajectory[:, t-1] * (1 - trajectory[:, t-1])
            
            # Update pattern with structure preservation
            update = self.dt * (self.diffusion_coeff * diffusion + reaction)
            
            # Project update to preserve energy
            energy_before = torch.sum(trajectory[:, t-1] ** 2, dim=-1, keepdim=True)
            next_state = trajectory[:, t-1] + update
            energy_after = torch.sum(next_state ** 2, dim=-1, keepdim=True)
            scale_factor = torch.sqrt(energy_before / energy_after)
            next_state = next_state * scale_factor
            
            # Project to preserve orthogonality if structure preservation is enabled
            if self.preserve_structure:
                # Reshape to square matrix if possible
                n = int(torch.sqrt(torch.tensor(trajectory.shape[-1])))
                if n * n == trajectory.shape[-1]:
                    # Reshape to batch of square matrices
                    matrices = next_state.view(batch_size, n, n)
                    # Project each matrix to O(n)
                    for i in range(batch_size):
                        U, S, V = torch.linalg.svd(matrices[i])
                        matrices[i] = torch.matmul(U, V)
                    # Reshape back
                    next_state = matrices.view(batch_size, -1)
            
            # Store the result
            trajectory[:, t] = next_state.clone()
        
        return trajectory
        
    def compute_energy(self, pattern: torch.Tensor) -> torch.Tensor:
        """Compute energy of pattern with geometric structure.
        
        Args:
            pattern: Pattern tensor of shape (batch_size, dim)
            
        Returns:
            torch.Tensor: Energy of pattern
        """
        # Handle pattern dimensions
        pattern_symplectic = self.symplectic._handle_dimension(pattern)
        
        # Compute quantum geometric tensor
        Q = self.symplectic.compute_quantum_geometric_tensor(pattern_symplectic)
        g = Q.real  # Metric part
        omega = Q.imag  # Symplectic part
        
        # Compute gradient term with metric structure
        grad = torch.diff(pattern_symplectic, dim=-1)
        grad_energy = torch.einsum('...i,...ij,...j->...', grad, g, grad)
        
        # Compute potential term (double-well potential) with symplectic structure
        potential = 0.25 * pattern_symplectic**4 - 0.5 * pattern_symplectic**2
        potential_energy = torch.sum(potential, dim=-1)
        
        # Add symplectic contribution
        symplectic_energy = torch.abs(
            self.symplectic.compute_form(pattern_symplectic).evaluate(
                pattern_symplectic,
                pattern_symplectic
            )
        )
        
        return grad_energy + potential_energy + self.symplectic._SYMPLECTIC_WEIGHT * symplectic_energy
        
    def compute_stability(self, pattern: torch.Tensor) -> Dict[str, Any]:
        """Compute stability metrics with geometric structure.
        
        Args:
            pattern: Pattern tensor of shape (batch_size, dim)
            
        Returns:
            Dict containing stability metrics
        """
        # Enable anomaly detection
        torch.autograd.set_detect_anomaly(True)
        
        # Handle pattern dimensions and create a copy to avoid in-place modifications
        pattern_symplectic = self.symplectic._handle_dimension(pattern.clone())
        
        # Ensure pattern is reshaped to a square matrix if needed
        n = int(torch.sqrt(torch.tensor(pattern_symplectic.shape[-1])))
        if n * n != pattern_symplectic.shape[-1]:
            # Pad to next perfect square
            next_square = (n + 1) ** 2
            padding = torch.zeros(*pattern_symplectic.shape[:-1], next_square - pattern_symplectic.shape[-1],
                                device=pattern_symplectic.device, dtype=pattern_symplectic.dtype)
            pattern_symplectic = torch.cat([pattern_symplectic, padding], dim=-1)
            n = int(torch.sqrt(torch.tensor(pattern_symplectic.shape[-1])))
        
        # Reshape to square matrix
        pattern_symplectic = pattern_symplectic.view(-1, n, n)
        
        # Define evolution step function that avoids in-place operations
        def evolution_step(input_pattern):
            # Create a copy of the input pattern and enable gradients
            x = input_pattern.clone().requires_grad_(True)
            # Evolve pattern for 2 time steps
            batch_size = x.size(0)
            
            # Initialize trajectory tensor
            trajectory = torch.zeros(
                batch_size,
                2,
                n,
                n,
                device=x.device,
                dtype=x.dtype
            )
            trajectory[:, 0] = x.clone()
            
            # Compute quantum geometric tensor
            Q = self.symplectic.compute_quantum_geometric_tensor(x.view(batch_size, -1))
            g = Q.real  # Metric part
            omega = Q.imag  # Symplectic part
            
            # Diffusion term with structure preservation
            current = x.view(batch_size, 1, -1)  # Reshape for conv1d
            diffusion = torch.nn.functional.conv1d(
                current,
                self.diffusion_kernel,
                padding=1
            ).view(batch_size, n, n)
            
            # Reaction term with structure preservation
            reaction = self.reaction_coeff * x * (1 - x)
            
            # Update pattern with structure preservation
            update = self.dt * (self.diffusion_coeff * diffusion + reaction)
            
            # Project update to preserve energy
            energy_before = torch.sum(x ** 2, dim=(1, 2), keepdim=True)
            next_state = x + update
            energy_after = torch.sum(next_state ** 2, dim=(1, 2), keepdim=True)
            scale_factor = torch.sqrt(energy_before / energy_after)
            next_state = next_state * scale_factor
            
            # Store the result
            trajectory[:, 1] = next_state.clone()
            
            return trajectory[:, 1]
        
        try:
            # Compute Jacobian with structure preservation
            jac = torch.autograd.functional.jacobian(evolution_step, pattern_symplectic)
            
            # Convert jacobian to proper tensor shape
            if isinstance(jac, tuple):
                jac = torch.stack(list(jac))
            
            # Reshape jacobian to square matrix
            jac = jac.view(pattern_symplectic.size(0), n * n, n * n)
            
            # Compute eigenvalues with geometric structure
            eigenvals = torch.linalg.eigvals(jac)
            
            # Compute stability metrics
            max_eigenval = torch.max(eigenvals.real)
            stability_margin = -max_eigenval.item()
            
            # Compute quantum geometric tensor
            Q = self.symplectic.compute_quantum_geometric_tensor(pattern_symplectic.view(pattern_symplectic.size(0), -1))
            g = Q.real  # Metric part
            omega = Q.imag  # Symplectic part
            
            # Compute symplectic invariants
            symplectic_form = self.symplectic.compute_form(pattern_symplectic.view(pattern_symplectic.size(0), -1))
            symplectic_invariant = torch.abs(
                symplectic_form.evaluate(pattern_symplectic.view(pattern_symplectic.size(0), -1),
                                       pattern_symplectic.view(pattern_symplectic.size(0), -1))
            ).item()
            
            return {
                'stability_margin': stability_margin,
                'max_eigenvalue': max_eigenval.item(),
                'eigenvalues': eigenvals.detach(),
                'quantum_metric': g.detach(),
                'symplectic_form': omega.detach(),
                'symplectic_invariant': symplectic_invariant
            }
        finally:
            # Disable anomaly detection
            torch.autograd.set_detect_anomaly(False)
