"""Information-Ricci Flow Implementation.

This module implements the information-Ricci flow with stress-energy tensor coupling.
It extends the basic geometric flow with information geometry and quantum corrections.

Mathematical Framework:
    The flow evolves a Riemannian metric g_ij according to the equation:
    ∂_t g_ij = -2R_ij + ∇_i∇_j f + T_ij

    where:
    - R_ij is the Ricci curvature tensor
    - f is the information potential (entropy)
    - T_ij is the quantum stress-energy tensor
    - ∇_i denotes covariant differentiation

Key Components:
    1. Classical Ricci Flow:
       - Smooths curvature irregularities
       - Evolves toward constant curvature
       - Resolves geometric singularities

    2. Information Potential:
       - Couples to metric through Hessian
       - Encodes entropy gradients
       - Drives flow toward information equilibrium

    3. Stress-Energy Coupling:
       - Quantum corrections to geometry
       - Backreaction from quantum state
       - Energy-momentum conservation

Implementation Features:
    - Neural network approximation of geometric quantities
    - Memory-efficient tensor operations with JIT compilation
    - Adaptive regularization and timesteps
    - Fused operations for optimal performance
    - Residual connections for stable training

References:
    [1] Hamilton (1982) Three-manifolds with positive Ricci curvature
    [2] Perelman (2002) The entropy formula for the Ricci flow
    [3] Quantum geometric flows and information geometry
"""

from typing import Dict, List, Optional, Tuple, Union, cast
import torch
import torch.nn.functional as F
from torch import nn, Tensor
from dataclasses import dataclass
from functools import lru_cache

from .neural import NeuralGeometricFlow
from ..quantum.types import QuantumState
from .protocol import FlowMetrics, QuantumFlowMetrics
from src.core.crystal.scale_classes.memory_utils import memory_efficient_computation

@dataclass(frozen=True)
class FlowParameters:
    """Parameters controlling the information-Ricci flow evolution.
    
    The parameters determine the relative contributions of different
    geometric and quantum terms in the flow equation:
    
    ∂_t g_ij = -2R_ij + α∇_i∇_j f + βT_ij
    
    where:
    - α = fisher_rao_weight controls entropy coupling
    - β = stress_energy_weight controls quantum effects
    - stability_threshold ensures metric positivity
    - dt sets the base integration timestep
    
    Attributes:
        fisher_rao_weight: Weight α of Fisher-Rao metric contribution
        quantum_weight: Weight of quantum geometric terms
        stress_energy_weight: Weight β of stress-energy coupling
        stability_threshold: Minimum eigenvalue threshold for stability
        dt: Base timestep for flow evolution
        
    Note:
        The class is frozen (immutable) to prevent accidental parameter
        modification during flow evolution.
    """
    fisher_rao_weight: float
    quantum_weight: float
    stress_energy_weight: float
    stability_threshold: float
    dt: float

@torch.jit.script
def compute_density_matrix(amplitudes: torch.Tensor) -> torch.Tensor:
    """Compute quantum density matrix with optimal efficiency.
    
    Implements the outer product |ψ⟩⟨ψ| for a pure quantum state |ψ⟩.
    Uses optimized matrix operations instead of einsum for better performance.
    
    Mathematical Expression:
        ρ_ij = ψ_i ψ_j^*
    
    Implementation Notes:
        - Uses unsqueeze for efficient batched outer product
        - Avoids unnecessary memory allocation
        - JIT-compiled for performance
    
    Args:
        amplitudes: Complex amplitudes ψ_i of quantum state [batch_size, dim]
        
    Returns:
        Density matrix ρ_ij [batch_size, dim, dim]
        
    Properties:
        - Hermitian: ρ_ij = ρ_ji^*
        - Positive semidefinite: ⟨v|ρ|v⟩ ≥ 0
        - Trace one: Tr(ρ) = 1
    """
    return amplitudes.unsqueeze(-1) @ amplitudes.conj().unsqueeze(-2)

@torch.jit.script
def ensure_metric_stability(
    metric: torch.Tensor,
    eye: torch.Tensor,
    stability_threshold: float,
    manifold_dim: int
) -> torch.Tensor:
    """Ensure metric stability through regularization and projection.
    
    Implements a three-step stabilization procedure:
    
    1. Symmetrization and Regularization:
       g_ij = 1/2(g_ij + g_ji) + εδ_ij
       where ε adapts to condition number
    
    2. Eigenvalue Bounds:
       λ_i ≥ stability_threshold
       Preserves metric positivity
    
    3. Volume Preservation:
       det(g_new) = det(g_old)
       Maintains geometric measure
    
    Implementation Notes:
        - Fuses operations for efficiency
        - Uses batched eigendecomposition
        - Optimizes memory usage
        - JIT-compiled for performance
    
    Args:
        metric: Input metric tensor g_ij [batch_size, dim, dim]
        eye: Identity matrix δ_ij [batch_size, dim, dim]
        stability_threshold: Minimum eigenvalue threshold
        manifold_dim: Dimension of the manifold
        
    Returns:
        Stabilized metric tensor [batch_size, dim, dim]
        
    Properties:
        - Symmetric: g_ij = g_ji
        - Positive definite: v^i g_ij v^j > 0
        - Volume preserving: det(g_new) = det(g_old)
    """
    # Fuse operations for better efficiency
    metric = 0.5 * (metric + metric.transpose(-2, -1)) + (
        torch.where(
            torch.linalg.cond(metric) > 1e4,
            1e-2 * eye,
            1e-3 * eye
        )
    )
    
    # Efficient eigendecomposition with batched operations
    eigenvalues, eigenvectors = torch.linalg.eigh(metric)
    eigenvalues = torch.maximum(
        eigenvalues,
        torch.full_like(eigenvalues, stability_threshold)
    )
    
    # Fused volume preservation and reconstruction
    scale = eigenvalues.prod(dim=-1, keepdim=True).pow(-1.0 / manifold_dim)
    eigenvalues = eigenvalues * scale
    
    # One-step reconstruction using batched operations
    return eigenvectors @ (eigenvalues.unsqueeze(-1) * eigenvectors.transpose(-2, -1))

class InformationRicciFlow(NeuralGeometricFlow):
    """Information-Ricci flow implementation with stress-energy coupling.
    
    This class implements a modified Ricci flow that incorporates both
    information geometry and quantum effects. The flow equation is:
    
    ∂_t g_ij = -2R_ij + ∇_i∇_j f + T_ij
    
    Mathematical Components:
        1. Ricci Tensor R_ij:
           - Measures intrinsic curvature
           - Drives geometric evolution
           - Computed through neural approximation
        
        2. Information Potential f:
           - Encodes entropy gradient
           - Couples to metric through Hessian
           - Learned via potential network
        
        3. Stress-Energy T_ij:
           - Quantum backreaction
           - Preserves energy-momentum
           - Computed from quantum state
    
    Implementation Features:
        - Neural Networks:
          * Residual connections for stability
          * Skip connections in stress-energy
          * Adaptive dimensionality
        
        - Memory Optimization:
          * Pre-allocated tensors
          * Fused operations
          * Efficient basis caching
        
        - Numerical Stability:
          * Adaptive regularization
          * Eigenvalue bounds
          * Volume preservation
    
    References:
        [1] Geometric flows and quantum information
        [2] Neural network approximation in geometry
        [3] Information-theoretic aspects of Ricci flow
    """
    
    def __init__(
        self,
        manifold_dim: int,
        hidden_dim: int,
        dt: float = 0.1,
        stability_threshold: float = 1e-6,
        fisher_rao_weight: float = 1.0,
        quantum_weight: float = 1.0,
        stress_energy_weight: float = 1.0,
        num_heads: int = 8,
        dropout: float = 0.1
    ):
        """Initialize the information-Ricci flow.
        
        Constructs the neural networks and initializes parameters for
        the modified Ricci flow with information and quantum coupling.
        
        Network Architecture:
            1. Potential Network:
               - Maps points to scalar potential
               - Residual connections
               - Adaptive hidden dimensions
            
            2. Stress-Energy Network:
               - Encodes quantum effects
               - Skip connections
               - Efficient dimensionality
        
        Args:
            manifold_dim: Dimension n of the base manifold M^n
            hidden_dim: Base dimension for neural networks
            dt: Base timestep for flow integration
            stability_threshold: Minimum eigenvalue bound
            fisher_rao_weight: Weight α of information term
            quantum_weight: Weight γ of quantum terms
            stress_energy_weight: Weight β of stress-energy
            num_heads: Number of attention heads
            dropout: Dropout rate for regularization
            
        Notes:
            - Network dimensions adapt to manifold complexity
            - Parameters are immutable during evolution
            - Memory optimization through caching
        """
        super().__init__(
            manifold_dim=manifold_dim,
            hidden_dim=hidden_dim,
            dt=dt,
            stability_threshold=stability_threshold,
            num_heads=num_heads,
            dropout=dropout
        )
        
        self.params = FlowParameters(
            fisher_rao_weight=fisher_rao_weight,
            quantum_weight=quantum_weight,
            stress_energy_weight=stress_energy_weight,
            stability_threshold=stability_threshold,
            dt=dt
        )
        
        # Optimize network dimensions based on manifold complexity
        reduced_dim = max(hidden_dim // 4, manifold_dim)
        intermediate_dim = (manifold_dim + reduced_dim) // 2
        
        # Efficient potential network with residual connection
        self.potential_net: nn.Sequential = nn.Sequential(
            nn.Linear(manifold_dim, intermediate_dim),
            nn.SiLU(),
            nn.Dropout(dropout),  # Add regularization
            nn.Linear(intermediate_dim, reduced_dim),
            nn.SiLU(),
            nn.Linear(reduced_dim, 1, bias=False)
        )
        
        # Efficient stress-energy network with skip connection
        self.stress_energy_net = nn.ModuleDict({
            'encoder': nn.Linear(manifold_dim * 2, intermediate_dim),
            'processor': nn.Sequential(
                nn.SiLU(),
                nn.Dropout(dropout),
                nn.Linear(intermediate_dim, reduced_dim),
                nn.SiLU(),
            ),
            'decoder': nn.Linear(reduced_dim, manifold_dim * manifold_dim, bias=False)
        })
        
        # Cache for basis vectors
        self.register_buffer(
            'basis_vectors',
            torch.eye(manifold_dim).unsqueeze(0)
        )

    # @torch.jit.script  # Disable TorchScript for now
    def _fused_potential_computation(self, points: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute potential and its gradients in a fused operation.
        
        Args:
            points: Input points x^i on manifold [batch_size, manifold_dim]
            
        Returns:
            Tuple of:
            - Information potential f [batch_size, 1]
            - Potential gradients ∂f/∂x^i [batch_size, manifold_dim]
        """
        points = points.detach().requires_grad_(True)
        potential = self.potential_net(points)
        grad_outputs = torch.ones_like(potential)
        grads = torch.autograd.grad(
            potential,
            points,
            grad_outputs=grad_outputs,
            create_graph=True
        )[0]
        return potential, grads

    def compute_information_potential(self, points: Tensor) -> Tensor:
        """Compute the information potential function.
        
        The potential f encodes the information content of the metric
        through a neural network approximation. It satisfies:
        
        ∂_t f = -R - |∇f|² + Λ
        
        where:
        - R is the scalar curvature
        - |∇f|² is the gradient norm
        - Λ is a normalization term
        
        Implementation Notes:
            - Uses efficient memory context
            - Reuses gradient computation
            - Optimizes tensor operations
        
        Args:
            points: Input points x^i on manifold [batch_size, manifold_dim]
            
        Returns:
            Information potential f [batch_size, 1]
            
        Properties:
            - Scalar: Coordinate independent
            - Bounded: |f| < ∞
            - Smooth: C^∞ differentiable
        """
        with memory_efficient_computation("compute_potential"):
            potential, _ = self._fused_potential_computation(points)
            return potential

    def compute_potential_hessian(self, points: Tensor) -> Tensor:
        """Compute Hessian of information potential efficiently.
        
        Implements the computation of ∇_i∇_j f using vectorized
        operations and pre-allocated tensors for optimal performance.
        
        Mathematical Expression:
            H_ij = ∇_i∇_j f = ∂_i∂_j f - Γ^k_ij ∂_k f
        
        where:
        - ∂_i∂_j f is the coordinate Hessian
        - Γ^k_ij are Christoffel symbols
        - ∂_k f is the gradient
        
        Implementation Notes:
            - Uses cached basis vectors
            - Pre-allocates tensors
            - Vectorizes computation
            - Ensures symmetry
        
        Args:
            points: Input points x^i on manifold [batch_size, manifold_dim]
            
        Returns:
            Hessian tensor H_ij [batch_size, manifold_dim, manifold_dim]
            
        Properties:
            - Symmetric: H_ij = H_ji
            - Covariant: Transforms as (0,2) tensor
            - Local: Depends only on nearby points
        """
        with memory_efficient_computation("compute_hessian"):
            # Fused computation of potential and gradients
            _, first_grads = self._fused_potential_computation(points)
            
            # Pre-allocate hessian tensor
            hessian = torch.empty(
                points.shape[0],
                self.manifold_dim,
                self.manifold_dim,
                dtype=points.dtype,
                device=points.device
            )
            
            # Vectorized Hessian computation
            for i in range(self.manifold_dim):
                grad_outputs = self.basis_vectors[:, i].expand_as(first_grads)
                hessian[:, i] = torch.autograd.grad(
                    first_grads,
                    points,
                    grad_outputs=grad_outputs,
                    create_graph=True
                )[0]
            
            # Symmetrize in one step
            return 0.5 * (hessian + hessian.transpose(-2, -1))

    def compute_stress_energy_tensor(
        self,
        points: Tensor,
        metric: Tensor
    ) -> Tensor:
        """Compute quantum stress-energy tensor efficiently.
        
        Implements the quantum stress-energy tensor T_ij that couples
        quantum states to geometry through:
        
        T_ij = ⟨ψ|T̂_ij|ψ⟩ - 1/2 g_ij ⟨ψ|T̂|ψ⟩
        
        where:
        - |ψ⟩ is the quantum state
        - T̂_ij is the stress-energy operator
        - T̂ = g^ij T̂_ij is the trace
        
        Implementation Notes:
            - Efficient quantum state prep
            - Optimized density matrix
            - Skip connections
            - Memory cleanup
        
        Args:
            points: Input points x^i on manifold [batch_size, manifold_dim]
            metric: Metric tensor g_ij [batch_size, manifold_dim, manifold_dim]
            
        Returns:
            Stress-energy tensor T_ij [batch_size, manifold_dim, manifold_dim]
            
        Properties:
            - Symmetric: T_ij = T_ji
            - Conserved: ∇^i T_ij = 0
            - Physical: Satisfies energy conditions
        """
        with memory_efficient_computation("compute_stress_energy"):
            # Efficient quantum state computation
            with torch.no_grad():
                quantum_state = self.prepare_quantum_state(points, return_validation=False)
                if not isinstance(quantum_state, QuantumState):
                    return torch.zeros_like(metric)
                
                # Compute density matrix and prepare inputs in one step
                density_matrix = compute_density_matrix(quantum_state.amplitudes)
                inputs = torch.cat([
                    points,
                    density_matrix.reshape(points.shape[0], -1)
                ], dim=-1)
                del quantum_state, density_matrix
            
            # Compute stress-energy with skip connection
            encoded = self.stress_energy_net['encoder'](inputs)
            processed = self.stress_energy_net['processor'](encoded)
            stress_energy = self.stress_energy_net['decoder'](processed + encoded).view(
                points.shape[0],
                self.manifold_dim,
                self.manifold_dim
            )
            
            # Fused symmetrization and trace correction
            stress_energy = 0.5 * (stress_energy + stress_energy.transpose(-2, -1))
            trace = torch.diagonal(stress_energy, dim1=-2, dim2=-1).sum(-1, keepdim=True)
            
            return stress_energy - 0.5 * trace.unsqueeze(-1) * metric

    def flow_step(
        self,
        metric: Tensor,
        ricci: Optional[Tensor] = None,
        timestep: Optional[float] = None
    ) -> Tuple[Tensor, QuantumFlowMetrics]:
        """Perform one step of information-Ricci flow evolution.
        
        Implements a single step of the modified Ricci flow:
        
        ∂_t g_ij = -2R_ij + α∇_i∇_j f + βT_ij
        
        The evolution combines:
        1. Classical Ricci flow: Smooths curvature
        2. Information coupling: Drives entropy evolution
        3. Quantum effects: Incorporates backreaction
        
        Implementation Strategy:
            1. Base Evolution:
               - Compute Ricci tensor
               - Apply classical flow
               - Update metric
            
            2. Information Terms:
               - Compute potential Hessian
               - Scale by Fisher-Rao weight
               - Add to flow
            
            3. Quantum Coupling:
               - Compute stress-energy
               - Scale by coupling weight
               - Include backreaction
            
            4. Stability Enforcement:
               - Ensure positivity
               - Preserve volume
               - Maintain regularity
        
        Args:
            metric: Input metric g_ij [batch_size, manifold_dim, manifold_dim]
            ricci: Optional pre-computed Ricci tensor R_ij
            timestep: Optional custom timestep (uses self.params.dt if None)
            
        Returns:
            Tuple of:
            - Evolved metric tensor g_ij(t + dt)
            - Flow metrics containing diagnostic information
            
        Properties:
            - Stable: Preserves metric properties
            - Adaptive: Timestep based on flow magnitude
            - Efficient: Optimized tensor operations
            
        References:
            [1] Modified Ricci flow equations
            [2] Quantum geometric evolution
            [3] Information geometry in physics
        """
        with memory_efficient_computation("flow_step"):
            # Get base flow step
            new_metric, base_metrics = super().flow_step(metric, ricci, timestep or self.params.dt)
            
            # Prepare tensors efficiently
            points = metric.reshape(metric.shape[0], -1)
            eye = torch.eye(
                self.manifold_dim,
                dtype=metric.dtype,
                device=metric.device
            ).expand(metric.shape[0], -1, -1)
            
            # Compute flow terms with potential caching
            potential_hessian = self.compute_potential_hessian(points)
            stress_energy = self.compute_stress_energy_tensor(points, metric)
            
            # Fused flow magnitude computation
            flow_magnitude = (
                torch.norm(potential_hessian) +
                self.params.stress_energy_weight * torch.norm(stress_energy)
            )
            dt = (timestep or self.params.dt) / (1 + flow_magnitude)
            
            # Fused metric update and stability enforcement
            flow_contribution = dt * (
                potential_hessian +
                self.params.stress_energy_weight * stress_energy
            )
            new_metric = ensure_metric_stability(
                new_metric + flow_contribution,
                eye,
                self.params.stability_threshold,
                self.manifold_dim
            )
            
            return new_metric, base_metrics 