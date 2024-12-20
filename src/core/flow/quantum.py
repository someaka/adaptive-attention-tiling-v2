"""Quantum Geometric Flow Implementation.

This module provides a specialized implementation of geometric flows for quantum systems,
incorporating quantum corrections and uncertainty principles into the flow evolution.
"""

from typing import Dict, List, Tuple, Any, Optional, cast
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn, Tensor

from .base import BaseGeometricFlow
from .protocol import FlowMetrics, QuantumFlowMetrics, SingularityInfo

# Quantum and Wave Packet imports
from ..patterns.enriched_structure import EnrichedMorphism, EnrichedTransition
from ..patterns.evolution import PatternEvolution, PatternEvolutionMetrics
from ..patterns.symplectic import SymplecticStructure
from ..quantum.state_space import HilbertSpace
from ..quantum.types import QuantumState
from ..interfaces.quantum import GeometricFlow, QuantumOperation, QuantumObservable
from ...validation.quantum.state import (
    StateValidator,
    StatePreparationValidator,
    EntanglementMetrics,
    TomographyValidator
)
from ...neural.attention.pattern.quantum import QuantumGeometricTensor

@dataclass
class AnalyzerMetrics:
    """Comprehensive metrics from flow analysis."""
    ricci_scalar: float
    mean_curvature: float
    berry_phase: float
    holonomy: float
    quantum_corrections: float
    stability: float
    convergence: float
    energy_conservation: float

class QuantumGeometricFlow(BaseGeometricFlow):
    """Quantum-specific implementation of geometric flow.
    
    This class extends the base geometric flow implementation with:
    1. Quantum corrections to the metric
    2. Uncertainty principle constraints
    3. Quantum state normalization
    4. Entanglement-aware transport
    5. Berry phase and holonomy tracking
    6. Mean curvature flow
    """
    
    def __init__(
        self,
        manifold_dim: int,
        hidden_dim: int,
        hbar: float = 1.0,
        dt: float = 0.1,
        stability_threshold: float = 1e-6,
        hilbert_space: Optional[HilbertSpace] = None,
        decoherence_rates: Optional[Dict[str, float]] = None
    ):
        """Initialize quantum geometric flow.
        
        Args:
            manifold_dim: Dimension of the base manifold
            hidden_dim: Hidden dimension for computations
            hbar: Reduced Planck constant
            dt: Time step for flow integration
            stability_threshold: Threshold for stability checks
            hilbert_space: Optional HilbertSpace for quantum operations
            decoherence_rates: Optional dictionary with T1, T2 decoherence rates
        """
        super().__init__(
            manifold_dim=manifold_dim,
            hidden_dim=hidden_dim,
            dt=dt,
            stability_threshold=stability_threshold
        )
        self.hbar = hbar
        self.hilbert_space = hilbert_space
        self.decoherence_rates = decoherence_rates or {"T1": 1.0, "T2": 0.5}
        self._points: Optional[Tensor] = None
        self._path_history: List[Tensor] = []
        self._current_state: Optional[QuantumState] = None
        
        # Initialize validators if HilbertSpace provided
        if hilbert_space is not None:
            self.state_validator = StateValidator(tolerance=stability_threshold)
            self.tomography_validator = TomographyValidator()
        
        # Quantum correction networks
        self.uncertainty_net = nn.Sequential(
            nn.Linear(manifold_dim * 2, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, manifold_dim * manifold_dim)
        )
        
        self.entanglement_net = nn.Sequential(
            nn.Linear(manifold_dim * 3, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, manifold_dim * manifold_dim)
        )

        # Mean curvature network
        self.mean_curvature_net = nn.Sequential(
            nn.Linear(manifold_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, manifold_dim)
        )

        # Berry phase tracking
        self.berry_connection = nn.Parameter(
            torch.zeros(manifold_dim, manifold_dim, dtype=torch.complex64)
        )

        # Additional quantum geometric networks
        self.fubini_study_net = nn.Sequential(
            nn.Linear(manifold_dim * 2, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, manifold_dim * manifold_dim),
            nn.Softplus()  # Ensure positive definiteness
        )
        
        self.quantum_transport_net = nn.Sequential(
            nn.Linear(manifold_dim * 2, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, manifold_dim),
            nn.Tanh()  # Bounded transport coefficients
        )
        
        self.state_reconstruction_net = nn.Sequential(
            nn.Linear(manifold_dim * 2, hidden_dim * 2),
            nn.ReLU(),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, manifold_dim * 2)  # Real and imaginary parts
        )

    def flow_step(
        self,
        metric: Tensor,
        ricci: Optional[Tensor] = None,
        timestep: float = 0.1,
        quantum_state: Optional[QuantumState] = None
    ) -> Tuple[Tensor, QuantumFlowMetrics]:
        """Perform quantum-aware flow step."""
        # Get base flow step
        new_metric, base_metrics = super().flow_step(metric, ricci, timestep)
        
        # Get quantum metrics
        quantum_metrics = self.compute_quantum_metrics(quantum_state) if quantum_state else {}
        
        # Create QuantumFlowMetrics instance
        metrics = QuantumFlowMetrics(
            flow_magnitude=base_metrics.flow_magnitude,
            metric_determinant=base_metrics.metric_determinant,
            ricci_scalar=base_metrics.ricci_scalar,
            energy=base_metrics.energy + self.hbar * base_metrics.flow_magnitude,
            singularity=base_metrics.singularity,
            normalized_flow=torch.linalg.det(new_metric).mean().item(),
            quantum_entropy=quantum_metrics.get("von_neumann_entropy", torch.tensor(0.0, device=metric.device)),
            berry_phase=quantum_metrics.get("berry_phase", torch.tensor(0.0, device=metric.device)),
            mean_curvature=quantum_metrics.get("mean_curvature", torch.tensor(0.0, device=metric.device)),
            quantum_corrections=quantum_metrics.get("quantum_corrections", torch.tensor(0.0, device=metric.device))
        )
        
        return new_metric, metrics

    def compute_quantum_metrics(
        self,
        quantum_state: Optional[QuantumState]
    ) -> Dict[str, Optional[Tensor]]:
        """Compute quantum-specific metrics from state."""
        if quantum_state is None or self.hilbert_space is None:
            return {}
            
        # Get density matrix
        rho = quantum_state.density_matrix()
        
        # Compute von Neumann entropy
        entropy = self.hilbert_space.compute_entropy(quantum_state)
        
        # Compute purity
        purity = torch.trace(torch.matmul(rho, rho)).real
        
        # Get entanglement metrics if state is multipartite
        try:
            # Use HilbertSpace's compute_negativity directly
            negativity = self.hilbert_space.compute_negativity(quantum_state)
            negativity = torch.tensor(float(negativity), device=rho.device)
        except:
            negativity = torch.tensor(0.0, device=rho.device)
            
        # Convert dt to tensor for exp operation
        dt_tensor = torch.tensor(self.dt, device=rho.device)
        T2_tensor = torch.tensor(self.decoherence_rates["T2"], device=rho.device)
            
        return {
            "von_neumann_entropy": entropy,
            "purity": purity,
            "negativity": negativity,
            "decoherence_factor": torch.exp(-dt_tensor / T2_tensor),
            "berry_phase": torch.tensor(0.0, device=rho.device),  # Placeholder
            "mean_curvature": torch.tensor(0.0, device=rho.device),  # Placeholder
            "quantum_corrections": torch.tensor(0.0, device=rho.device)  # Placeholder
        }

    def set_points(self, points: Tensor) -> None:
        """Set current points for flow computation."""
        self._points = points
        self._path_history.append(points)

    def compute_metric(
        self,
        points: Tensor,
        connection: Optional[Tensor] = None
    ) -> Tensor:
        """Compute quantum-corrected metric tensor.
        
        Incorporates uncertainty principle constraints into the metric.
        """
        # Get classical metric
        metric = super().compute_metric(points, connection)
        
        # Compute quantum corrections
        batch_size = points.shape[0]
        uncertainty = self.uncertainty_net(
            torch.cat([points, points * self.hbar], dim=-1)
        )
        uncertainty = uncertainty.view(batch_size, self.manifold_dim, self.manifold_dim)
        
        # Add quantum corrections
        metric = metric + self.hbar * uncertainty
        
        # Ensure Heisenberg constraints
        eye = torch.eye(self.manifold_dim, device=points.device).unsqueeze(0)
        min_uncertainty = self.hbar * 0.5 * eye
        metric = torch.where(metric < min_uncertainty, min_uncertainty, metric)
        
        return metric

    def compute_mean_curvature(self, points: Tensor) -> Tensor:
        """Compute mean curvature vector field."""
        return self.mean_curvature_net(points)

    def compute_berry_phase(self, path: List[Tensor]) -> Tensor:
        """Compute Berry phase along a path in parameter space."""
        total_phase = torch.zeros(1, dtype=torch.complex64, device=path[0].device)
        
        for i in range(len(path)-1):
            # Compute local connection
            delta = path[i+1] - path[i]
            phase = torch.einsum('ij,j->i', self.berry_connection, delta)
            total_phase = total_phase + torch.sum(phase)
            
        return total_phase

    def compute_connection(
        self,
        metric: Tensor,
        points: Optional[Tensor] = None
    ) -> Tensor:
        """Compute quantum-aware connection coefficients."""
        connection = super().compute_connection(metric, points)
        
        if points is not None:
            # Add quantum phase corrections
            batch_size = points.shape[0]
            phase = torch.angle(points + 1j * points.roll(1, dims=-1))
            phase_correction = self.hbar * phase.unsqueeze(-1).unsqueeze(-1)
            connection = connection + phase_correction
        
        return connection

    def compute_ricci_tensor(
        self,
        metric: Tensor,
        points: Optional[Tensor] = None,
        connection: Optional[Tensor] = None
    ) -> Tensor:
        """Compute Ricci tensor with quantum corrections."""
        # Get classical Ricci tensor
        ricci = super().compute_ricci_tensor(metric, points, connection)
        
        if points is not None:
            # Add entanglement corrections
            batch_size = points.shape[0]
            entanglement = self.entanglement_net(
                torch.cat([
                    points,
                    points.roll(1, dims=-1),
                    points.roll(-1, dims=-1)
                ], dim=-1)
            )
            entanglement = entanglement.view(
                batch_size, self.manifold_dim, self.manifold_dim
            )
            ricci = ricci + self.hbar * entanglement
        
        return ricci

    def compute_decoherence_correction(
        self,
        metric: Tensor,
        quantum_state: Optional[QuantumState] = None
    ) -> Tensor:
        """Compute decoherence corrections to the metric tensor."""
        if quantum_state is None or self.hilbert_space is None:
            return metric
            
        # Get density matrix
        rho = quantum_state.density_matrix()
        
        # Compute decoherence factors
        T1, T2 = self.decoherence_rates["T1"], self.decoherence_rates["T2"]
        gamma = 1.0 / T1
        dephasing = 1.0 / T2 - 0.5 / T1
        
        # Apply decoherence corrections to metric
        correction = torch.zeros_like(metric)
        for i in range(self.manifold_dim):
            for j in range(self.manifold_dim):
                # Population decay correction
                correction[..., i, j] += gamma * torch.diagonal(rho, dim1=-2, dim2=-1).mean()
                
                # Dephasing correction
                if i != j:
                    correction[..., i, j] += dephasing * torch.abs(rho[..., i, j])
        
        return metric + self.hbar * correction

    def parallel_transport(
        self,
        vector: Tensor,
        start_point: Tensor,
        end_point: Tensor,
        connection: Optional[Tensor] = None
    ) -> Tensor:
        """Parallel transport with quantum phase correction."""
        # Get classical transport
        transported = super().parallel_transport(
            vector, start_point, end_point, connection
        )
        
        # Add quantum phase
        phase = torch.angle(end_point) - torch.angle(start_point)
        transported = transported * torch.exp(1j * phase * self.hbar)
        
        return transported

    def compute_geodesic(
        self,
        start_point: Tensor,
        end_point: Tensor,
        num_steps: int = 10
    ) -> Tensor:
        """Compute quantum-corrected geodesic."""
        # Get classical geodesic
        path = super().compute_geodesic(start_point, end_point, num_steps)
        
        # Apply quantum corrections to path
        phases = torch.linspace(0, 1, num_steps + 1, device=path.device)
        phases = phases * torch.angle(end_point / (start_point + 1e-8))
        path = path * torch.exp(1j * phases.unsqueeze(-1) * self.hbar)
        
        return path 

    def compute_quantum_metric_tensor(
        self,
        state: QuantumState,
        metric: Optional[Tensor] = None
    ) -> Tensor:
        """Compute quantum geometric tensor using Fubini-Study metric.
        
        Args:
            state: Current quantum state
            metric: Optional classical metric tensor
            
        Returns:
            Quantum geometric tensor
        """
        if self.hilbert_space is None:
            raise ValueError("HilbertSpace required for quantum metric tensor")
            
        # Get tangent space at state
        tangent = self.hilbert_space.quantum_tangent_vector(state)
        
        # Compute Fubini-Study metric components using neural network
        batch_size = tangent.shape[0]
        fubini_input = torch.cat([
            tangent.real.view(batch_size, -1),
            tangent.imag.view(batch_size, -1)
        ], dim=-1)
        
        metric_components = self.fubini_study_net(fubini_input)
        metric_tensor = metric_components.view(
            batch_size, self.manifold_dim, self.manifold_dim
        )
        
        # Add quantum corrections from classical metric if provided
        if metric is not None:
            metric_tensor = metric_tensor + self.hbar * metric
            
        return metric_tensor

    def parallel_transport_state(
        self,
        state: QuantumState,
        vector: Tensor,
        connection: Optional[Tensor] = None
    ) -> QuantumState:
        """Parallel transport quantum state along vector field.
        
        Args:
            state: Quantum state to transport
            vector: Tangent vector field
            connection: Optional connection coefficients
            
        Returns:
            Transported quantum state
        """
        if self.hilbert_space is None:
            raise ValueError("HilbertSpace required for parallel transport")
            
        # Get current tangent space
        tangent = self.hilbert_space.quantum_tangent_vector(state)
        
        # Compute transport coefficients using neural network
        transport_input = torch.cat([
            tangent.view(-1, self.manifold_dim),
            vector
        ], dim=-1)
        transport_coeffs = self.quantum_transport_net(transport_input)
        
        # Apply transport to state amplitudes
        transported = state.amplitudes + transport_coeffs * vector
        
        # Apply quantum phase correction if connection provided
        if connection is not None:
            phase = torch.einsum('ijk,j,k->i', connection, vector, vector)
            transported = transported * torch.exp(1j * self.hbar * phase)
            
        return QuantumState(
            amplitudes=transported,
            basis_labels=state.basis_labels,
            phase=state.phase
        )

    def reconstruct_state(
        self,
        measurements: Dict[str, Tensor],
        bases: Optional[List[Tensor]] = None
    ) -> Optional[QuantumState]:
        """Reconstruct quantum state from measurements.
        
        Args:
            measurements: Dictionary of measurement results
            bases: Optional measurement bases
            
        Returns:
            Reconstructed quantum state if successful
        """
        if self.hilbert_space is None:
            raise ValueError("HilbertSpace required for state reconstruction")
            
        try:
            # Prepare input for reconstruction network
            measurement_tensor = torch.cat([
                measurements[key] for key in sorted(measurements.keys())
            ], dim=-1)
            
            # Add basis information if provided
            if bases is not None:
                basis_tensor = torch.cat(bases, dim=-1)
                reconstruction_input = torch.cat([
                    measurement_tensor,
                    basis_tensor
                ], dim=-1)
            else:
                reconstruction_input = measurement_tensor
                
            # Use neural network for reconstruction
            reconstructed = self.state_reconstruction_net(reconstruction_input)
            
            # Split into real and imaginary parts
            split_point = reconstructed.shape[-1] // 2
            real_part = reconstructed[..., :split_point]
            imag_part = reconstructed[..., split_point:]
            
            # Combine into complex amplitudes
            amplitudes = real_part + 1j * imag_part
            
            # Normalize the state
            norm = torch.sqrt(torch.sum(torch.abs(amplitudes) ** 2, dim=-1, keepdim=True))
            amplitudes = amplitudes / (norm + 1e-8)
            
            # Use measurement keys as basis labels
            basis_labels = sorted(measurements.keys())
            
            return QuantumState(
                amplitudes=amplitudes,
                basis_labels=basis_labels,
                phase=torch.angle(amplitudes[..., 0])  # Use first amplitude for phase
            )
            
        except Exception as e:
            print(f"State reconstruction failed: {e}")
            return None

class GeometricFlowAnalyzer:
    """Complete geometric flow analysis system."""

    def __init__(
        self,
        flow: QuantumGeometricFlow,
        hilbert_space: Optional[HilbertSpace] = None,
        tolerance: float = 1e-6
    ):
        """Initialize analyzer with flow instance.
        
        Args:
            flow: QuantumGeometricFlow instance to analyze
            hilbert_space: Optional HilbertSpace for quantum analysis
            tolerance: Numerical tolerance for validation checks
        """
        self.flow = flow
        self.hilbert_space = hilbert_space
        self.history: List[QuantumFlowMetrics] = []
        
        # Initialize validators
        self.state_validator = StateValidator(tolerance=tolerance)
        self.preparation_validator = StatePreparationValidator(tolerance=tolerance)
        
    def analyze_step(
        self,
        metric: Tensor,
        points: Optional[Tensor] = None,
        timestep: float = 0.1,
        quantum_state: Optional[QuantumState] = None
    ) -> Tuple[Tensor, AnalyzerMetrics]:
        """Analyze single flow step with quantum validation.
        
        Args:
            metric: Current metric tensor
            points: Optional points tensor
            timestep: Time step size
            quantum_state: Optional quantum state for validation
            
        Returns:
            Tuple of (evolved metric, analyzer metrics)
        """
        # Set points if provided
        if points is not None:
            self.flow.set_points(points)
            
        # Perform flow step
        new_metric, metrics = self.flow.flow_step(metric, timestep=timestep)
        self.history.append(metrics)
        
        # Validate quantum state if provided
        quantum_metrics = {}
        if quantum_state is not None and self.hilbert_space is not None:
            # Validate state properties
            state_props = self.state_validator.validate_state(quantum_state)
            
            # Validate uncertainty relations
            uncertainty = self.state_validator.validate_uncertainty(quantum_state)
            
            # Store quantum metrics
            quantum_metrics.update({
                "state_normalized": state_props.is_normalized,
                "state_pure": state_props.is_pure,
                "state_purity": state_props.purity,
                "position_uncertainty": uncertainty.position_uncertainty,
                "momentum_uncertainty": uncertainty.momentum_uncertainty,
                "heisenberg_product": uncertainty.heisenberg_product
            })
        
        # Compute stability from history
        if len(self.history) > 1:
            stability = torch.norm(
                self.history[-1].flow_magnitude - self.history[-2].flow_magnitude
            ).item()
        else:
            stability = 0.0
            
        # Compute convergence
        if len(self.history) > 10:
            recent_flows = [m.flow_magnitude for m in self.history[-10:]]
            convergence = torch.std(torch.tensor(recent_flows)).item()
        else:
            convergence = 1.0
            
        # Energy conservation
        if len(self.history) > 1:
            energy_conservation = abs(
                self.history[-1].energy - self.history[-2].energy
            ) / (abs(self.history[-1].energy) + 1e-8)
        else:
            energy_conservation = 1.0
            
        # Get tensor values safely
        mean_curvature_val = metrics.mean_curvature.norm().item() if metrics.mean_curvature is not None else 0.0
        berry_phase_val = metrics.berry_phase.abs().item() if metrics.berry_phase is not None else 0.0
        quantum_corrections_val = metrics.quantum_corrections.norm().item() if metrics.quantum_corrections is not None else 0.0
            
        # Compile analyzer metrics
        analyzer_metrics = AnalyzerMetrics(
            ricci_scalar=float(metrics.ricci_scalar),
            mean_curvature=mean_curvature_val,
            berry_phase=berry_phase_val,
            holonomy=float(metrics.normalized_flow),
            quantum_corrections=quantum_corrections_val,
            stability=float(stability),
            convergence=float(convergence),
            energy_conservation=float(energy_conservation)
        )
        
        return new_metric, analyzer_metrics
        
    def analyze_evolution(
        self,
        initial_metric: Tensor,
        num_steps: int = 100,
        dt: float = 0.01,
        points: Optional[Tensor] = None,
        initial_state: Optional[QuantumState] = None
    ) -> Tuple[List[AnalyzerMetrics], Optional[Dict[str, List[float]]]]:
        """Analyze complete flow evolution with quantum state tracking.
        
        Args:
            initial_metric: Initial metric tensor
            num_steps: Number of evolution steps
            dt: Time step size
            points: Optional points tensor
            initial_state: Optional initial quantum state to track
            
        Returns:
            Tuple of (metrics history, quantum metrics history)
        """
        metrics_history: List[AnalyzerMetrics] = []
        quantum_history: Optional[Dict[str, List[float]]] = None
        
        if initial_state is not None and self.hilbert_space is not None:
            quantum_history = {
                "state_purity": [],
                "heisenberg_product": [],
                "energy_uncertainty": []
            }
        
        current_metric = initial_metric
        current_state = initial_state
        
        for _ in range(num_steps):
            current_metric, metrics = self.analyze_step(
                current_metric, points, dt, current_state
            )
            metrics_history.append(metrics)
            
            # Update quantum state if provided
            if (current_state is not None and 
                self.hilbert_space is not None and 
                quantum_history is not None):
                # Evolve state using geometric flow
                evolved_state = self._evolve_quantum_state(current_state, current_metric, dt)
                
                # Validate evolved state
                state_props = self.state_validator.validate_state(evolved_state)
                uncertainty = self.state_validator.validate_uncertainty(evolved_state)
                
                # Track quantum metrics
                quantum_history["state_purity"].append(float(state_props.purity))
                quantum_history["heisenberg_product"].append(float(uncertainty.heisenberg_product))
                quantum_history["energy_uncertainty"].append(float(uncertainty.energy_uncertainty))
                
                current_state = evolved_state
            
        return metrics_history, quantum_history
    
    def _evolve_quantum_state(
        self, 
        state: QuantumState, 
        metric: Tensor, 
        dt: float
    ) -> QuantumState:
        """Evolve quantum state using geometric flow."""
        if self.hilbert_space is None:
            raise ValueError("HilbertSpace required for quantum state evolution")
            
        # Construct effective Hamiltonian from metric
        hamiltonian = torch.einsum('ij,jk->ik', metric, metric.conj().T)
        
        # Add quantum corrections
        hamiltonian = hamiltonian + self.flow.hbar * torch.eye(
            self.flow.manifold_dim, 
            device=metric.device
        )
        
        # Evolve state
        evolved = self.hilbert_space.evolve_state(state, hamiltonian, dt)
        # Handle potential List[QuantumState] return
        if isinstance(evolved, list):
            return evolved[-1]  # Return final state
        return evolved
        
    def get_convergence_stats(self) -> Dict[str, float]:
        """Get statistical analysis of flow convergence.
        
        Returns:
            Dictionary of convergence statistics
        """
        if not self.history:
            return {
                "mean_flow": 0.0,
                "flow_std": 0.0,
                "final_energy": 0.0,
                "energy_conservation": 0.0,
                "berry_phase_accumulated": 0.0
            }
            
        flow_magnitudes = torch.tensor([m.flow_magnitude for m in self.history])
        energies = torch.tensor([m.energy for m in self.history])
        berry_phases = torch.tensor([
            m.berry_phase.abs().item() if m.berry_phase is not None else 0.0
            for m in self.history
        ])
        
        return {
            "mean_flow": float(flow_magnitudes.mean().item()),
            "flow_std": float(flow_magnitudes.std().item()),
            "final_energy": float(energies[-1].item()),
            "energy_conservation": float(
                torch.std(energies).item() / (torch.mean(torch.abs(energies)) + 1e-8)
            ),
            "berry_phase_accumulated": float(berry_phases.sum().item())
        }

    def validate_entanglement(self, state: QuantumState) -> EntanglementMetrics:
        """Validate entanglement properties of quantum state.
        
        Args:
            state: Quantum state to analyze
            
        Returns:
            EntanglementMetrics containing validation results
        """
        if self.hilbert_space is None:
            raise ValueError("HilbertSpace required for entanglement validation")
            
        # Get density matrix
        rho = state.density_matrix()
        
        # Compute entanglement metrics
        concurrence = self.preparation_validator._compute_concurrence(state)
        entropy = self.hilbert_space.compute_entropy(state)
        negativity = self.hilbert_space.compute_negativity(state)
        log_negativity = torch.log2(2 * negativity + 1)
        
        # Check PPT criterion
        ppt_criterion = bool(negativity.item() < 1e-6)
        
        # Compute entanglement witness
        witness_value = self.hilbert_space.evaluate_entanglement_witness(state)
        
        return EntanglementMetrics(
            concurrence=float(concurrence.item()),
            von_neumann_entropy=float(entropy.item()),
            negativity=float(negativity.item()),
            log_negativity=float(log_negativity.item()),
            ppt_criterion=ppt_criterion,
            witness_value=float(witness_value.item())
        )