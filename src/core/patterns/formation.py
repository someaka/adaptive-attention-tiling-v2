"""Pattern formation module.

This module implements pattern formation dynamics and analysis tools.
It integrates symplectic geometry, fiber bundles, and quantum geometric tensors
for analyzing pattern dynamics and bifurcations.
"""

import torch
import numpy as np
from typing import Dict, Any, Optional, List, Tuple, Union, Protocol
from dataclasses import dataclass

from .symplectic import SymplecticStructure, SymplecticForm
from .operadic_structure import AttentionOperad, EnrichedAttention

@dataclass
class BifurcationMetrics:
    """Metrics for bifurcation analysis with geometric structure."""
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
        # Compute stability metrics along parameter range
        stability_metrics = []
        for i in range(len(parameter)):
            metrics = self._compute_stability_metrics(pattern[i])
            stability_metrics.append(metrics)
            
        # Detect significant changes in stability using geometric structure
        bifurcations = []
        for i in range(1, len(stability_metrics)):
            if self._is_bifurcation(
                stability_metrics[i-1],
                stability_metrics[i]
            ):
                bifurcations.append(float(parameter[i].item()))
                
        return bifurcations
        
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
        # Handle pattern dimensions through symplectic structure
        pattern_symplectic = self.symplectic._handle_dimension(pattern)
        
        # Compute temporal derivatives with structure preservation
        if pattern.dim() > 1:
            grad = torch.gradient(pattern_symplectic)[0]
            mean_rate = torch.mean(torch.abs(grad)).item()
            max_rate = torch.max(torch.abs(grad)).item()
        else:
            mean_rate = 0.0
            max_rate = 0.0
            
        # Compute quantum geometric tensor
        Q = self.symplectic.compute_quantum_geometric_tensor(pattern_symplectic)
        g = Q.real  # Metric part
        omega = Q.imag  # Symplectic part
        
        # Compute symplectic invariants
        symplectic_form = self.symplectic.compute_form(pattern_symplectic)
        symplectic_invariant = torch.abs(
            symplectic_form.evaluate(pattern_symplectic, pattern_symplectic)
        ).item()
        
        # Compute geometric flow for stability
        flow = self.symplectic.quantum_ricci_flow(
            pattern_symplectic,
            time=1.0,
            dt=0.1,
            steps=10
        )
        
        return BifurcationMetrics(
            stability_margin=mean_rate,
            max_eigenvalue=max_rate,
            symplectic_invariant=symplectic_invariant,
            quantum_metric=g,
            pattern_height=torch.mean(torch.abs(pattern_symplectic)).item(),
            geometric_flow=flow
        )
        
    def _is_bifurcation(
        self,
        metrics1: BifurcationMetrics,
        metrics2: BifurcationMetrics
    ) -> bool:
        """Check if transition between states is a bifurcation using geometric structure."""
        # Check standard metric changes
        if abs(metrics2.stability_margin - metrics1.stability_margin) > self.threshold:
            return True
            
        # Check symplectic structure preservation
        if abs(metrics2.symplectic_invariant - metrics1.symplectic_invariant) > self.threshold:
            return True
            
        # Check quantum geometric tensor evolution
        metric_diff = torch.norm(metrics2.quantum_metric - metrics1.quantum_metric)
        if metric_diff > self.threshold:
            return True
            
        # Check geometric flow stability
        flow_diff = torch.norm(metrics2.geometric_flow - metrics1.geometric_flow)
        if flow_diff > self.threshold:
            return True
            
        return False
        
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
        kernel = torch.tensor([[[0.2, 0.6, 0.2]]])
        
        # Ensure kernel preserves symplectic structure
        if self.preserve_structure:
            form = self.symplectic.compute_form(kernel)
            kernel = kernel * form.matrix
            
        return kernel
        
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
        trajectory[:, 0] = pattern_symplectic
        
        # Initialize wave packet if enabled
        if self.wave_enabled:
            n = pattern_symplectic.shape[-1] // 2
            position = pattern_symplectic[..., :n]
            momentum = pattern_symplectic[..., n:]
            wave_packet = self.enriched.create_wave_packet(position, momentum)
            trajectory[:, 0] = wave_packet
        
        # Evolve pattern with structure preservation
        for t in range(1, time_steps):
            # Compute quantum geometric tensor
            Q = self.symplectic.compute_quantum_geometric_tensor(trajectory[:, t-1])
            g = Q.real  # Metric part
            omega = Q.imag  # Symplectic part
            
            # Diffusion term with structure preservation
            diffusion = torch.nn.functional.conv1d(
                trajectory[:, t-1:t].unsqueeze(1),
                self.diffusion_kernel,
                padding=1
            ).squeeze(1)
            
            # Reaction term (cubic nonlinearity) with structure preservation
            reaction = trajectory[:, t-1] * (1 - trajectory[:, t-1]**2)
            
            # Update pattern with structure preservation
            trajectory[:, t] = trajectory[:, t-1] + self.dt * (
                self.diffusion_coeff * diffusion + 
                self.reaction_coeff * reaction
            )
            
            # Apply quantum Ricci flow for stability
            trajectory[:, t] = self.symplectic.quantum_ricci_flow(
                trajectory[:, t],
                time=self.dt,
                dt=self.dt/10,
                steps=5
            )
            
            # Verify structure preservation
            if self.preserve_structure:
                form_before = self.symplectic.compute_form(trajectory[:, t-1])
                form_after = self.symplectic.compute_form(trajectory[:, t])
                if not torch.allclose(
                    form_before.evaluate(trajectory[:, t-1], trajectory[:, t-1]),
                    form_after.evaluate(trajectory[:, t], trajectory[:, t]),
                    rtol=1e-5
                ):
                    raise ValueError("Symplectic structure not preserved during evolution")
            
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
        # Handle pattern dimensions
        pattern_symplectic = self.symplectic._handle_dimension(pattern)
        
        # Compute Jacobian with structure preservation
        x = pattern_symplectic.requires_grad_(True)
        y = self.evolve(x.unsqueeze(0), time_steps=2)[:, 1]
        jac = torch.autograd.functional.jacobian(
            lambda x: self.evolve(x.unsqueeze(0), time_steps=2)[:, 1],
            pattern_symplectic
        )
        
        # Convert jacobian to proper tensor shape
        if isinstance(jac, tuple):
            jac = torch.stack(list(jac))
        
        # Compute eigenvalues with geometric structure
        eigenvals = torch.linalg.eigvals(jac)
        
        # Compute stability metrics
        max_eigenval = torch.max(eigenvals.real)
        stability_margin = -max_eigenval.item()
        
        # Compute quantum geometric tensor
        Q = self.symplectic.compute_quantum_geometric_tensor(pattern_symplectic)
        g = Q.real  # Metric part
        omega = Q.imag  # Symplectic part
        
        # Compute symplectic invariants
        symplectic_form = self.symplectic.compute_form(pattern_symplectic)
        symplectic_invariant = torch.abs(
            symplectic_form.evaluate(pattern_symplectic, pattern_symplectic)
        ).item()
        
        return {
            'stability_margin': stability_margin,
            'max_eigenvalue': max_eigenval.item(),
            'eigenvalues': eigenvals.detach(),
            'quantum_metric': g.detach(),
            'symplectic_form': omega.detach(),
            'symplectic_invariant': symplectic_invariant
        }
