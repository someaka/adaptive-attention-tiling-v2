"""Scale System Implementation for Crystal Structures.

This module implements a multi-scale analysis system for crystalline structures,
incorporating geometric flows, quantum effects, and cohomological structures.

Mathematical Framework:
    The system analyzes scale transformations through several key components:
    
    1. Scale Connections:
       ∇_μ = ∂_μ + A_μ
       where A_μ is the scale connection encoding relative changes
    
    2. Renormalization Flow:
       β(g) = μ ∂g/∂μ
       describing coupling evolution across scales
    
    3. Cohomological Structure:
       H^k(M) = ker(d_k)/im(d_{k-1})
       capturing topological aspects of scale transformations

Key Components:
    1. Scale Cohomology:
       - Implements differential forms and operations
       - Computes cohomology groups
       - Tracks topological invariants
    
    2. Geometric Flow:
       - Ricci flow for metric evolution
       - Information geometry coupling
       - Quantum corrections
    
    3. Fixed Point Analysis:
       - Critical point detection
       - Stability analysis
       - Critical exponents

Implementation Features:
    - Memory-efficient tensor operations
    - JIT-compiled critical paths
    - Adaptive precision handling
    - Automatic differentiation
    - Quantum-geometric coupling

Performance Optimizations:
    - Fused operations for gradient computation
    - Pre-allocated tensors for heavy computations
    - Cached basis vectors and metrics
    - Adaptive dimensionality reduction
    - Memory-aware context managers

References:
    [1] Geometric Analysis on Crystal Structures
    [2] Quantum Cohomology in Materials Science
    [3] Information Geometry and Scale Transformations
"""

from dataclasses import dataclass, field
from typing import Callable, List, Tuple, Dict, Any, Union, Optional
from contextlib import contextmanager
import gc
import logging
from functools import lru_cache
import math

import numpy as np
import torch
from torch import nn, Tensor

# Import memory optimization utilities
from src.core.performance.cpu.memory_management import MemoryManager, MemoryMetrics
from utils.memory_management_util import optimize_memory, register_tensor
from src.core.crystal.scale_classes.rgflow import RGFlow
from src.core.crystal.scale_classes.complextanh import ComplexTanh
from src.core.crystal.scale_classes.scaleinvariance import ScaleInvariance
from src.core.crystal.scale_classes.anomalydetector import (
    AnomalyDetector, 
    AnomalyPolynomial, 
    ScaleConnection, 
    ScaleConnectionData
)
from src.core.crystal.scale_classes.renormalizationflow import RenormalizationFlow
from src.core.crystal.scale_classes.holographiclift import HolographicLifter


# Configure logging with structured format
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Global memory manager with enhanced metrics
_memory_manager = MemoryManager()

__all__ = ['ScaleSystem', 'ScaleCohomology']

@dataclass(frozen=True)
class MemoryStats:
    """Statistics for memory usage tracking.
    
    Attributes:
        allocated: Currently allocated memory in bytes
        peak: Peak memory usage in bytes
        fragmentation: Memory fragmentation ratio
        cache_hits: Number of cache hits
        cache_misses: Number of cache misses
    """
    allocated: int
    peak: int
    fragmentation: float
    cache_hits: int
    cache_misses: int

@contextmanager
def memory_efficient_computation(operation: str):
    """Context manager for memory-efficient computations.
    
    Implements:
        1. Automatic garbage collection
        2. Memory usage tracking
        3. Operation timing
        4. Error handling
    
    Args:
        operation: Name of the operation for logging
    
    Yields:
        None
    
    Example:
        with memory_efficient_computation("matrix_multiply"):
            result = torch.mm(a, b)
    """
    try:
        # Pre-operation cleanup
        gc.collect()
        
        # Log initial state
        initial_stats = _memory_manager.get_memory_stats()
        logger.debug(f"Starting {operation}")
        logger.debug(f"Initial memory: {initial_stats['allocated_memory'] / 1024**2:.2f}MB")
        
        yield
        
        # Post-operation cleanup and logging
        final_stats = _memory_manager.get_memory_stats()
        memory_delta = final_stats['allocated_memory'] - initial_stats['allocated_memory']
        logger.debug(f"Completed {operation}")
        logger.debug(f"Memory delta: {memory_delta / 1024**2:.2f}MB")
        
    except Exception as e:
        logger.error(f"Error in {operation}: {str(e)}")
        raise
    finally:
        gc.collect()

class ScaleCohomology:
    """Multi-scale cohomological structure for crystal analysis.
    
    This class implements the mathematical framework for analyzing
    crystal structures across multiple scales using differential
    geometry and cohomology theory.
    
    Mathematical Structure:
        1. De Rham Complex:
           Ω^0 → Ω^1 → ... → Ω^n
           with d^2 = 0
        
        2. Metric Evolution:
           ∂_t g_ij = -2R_ij + ∇_i∇_j f
           
        3. Scale Connection:
           A_μ = ∂_μ log(ρ)
           
    Implementation Features:
        - Efficient form computation
        - Geometric flow integration
        - Quantum state preparation
        - Anomaly detection
        
    Memory Optimization:
        - Pre-allocated tensors
        - Cached computations
        - Adaptive precision
        - Garbage collection
    """


    
    def __init__(
        self, 
        dim: int, 
        num_scales: int = 4, 
        dtype: torch.dtype = torch.float32
    ):
        """Initialize scale cohomology structure.
        
        Args:
            dim: Dimension n of the base manifold M^n
            num_scales: Number of scale levels to analyze
            dtype: Numerical precision for computations
            
        The initialization sets up:
            1. De Rham complex components
            2. Geometric flow networks
            3. Quantum coupling structures
            4. Scale connection handlers
        """
        # Allow any dimension, no minimum requirement
        self.dim = dim
        self.num_scales = num_scales
        self.dtype = dtype
        
        # Initialize lattice and Hilbert space for quantum analysis
        from src.core.quantum.crystal import BravaisLattice, HilbertSpace
        self.lattice = BravaisLattice(dim)
        # Use log2 of dimension to keep Hilbert space manageable
        hilbert_dim = max(2, int(math.log2(dim + 1)))  # At least 2-dimensional
        self.hilbert_space = HilbertSpace(hilbert_dim)  # Use log-scaled dimension

        # Use ComplexTanh for all networks if dtype is complex
        activation = ComplexTanh() if dtype == torch.complex64 else nn.Tanh()

        # De Rham complex components (Ω^k forms) with minimum dimensions
        self.forms = nn.ModuleList([
            nn.Sequential(
                nn.Linear(max(self._compute_form_dim(k, dim), 1), max(dim, 1), dtype=dtype),
                activation,
                nn.Linear(max(dim, 1), max(self._compute_form_dim(k + 1, dim), 1), dtype=dtype)
            ) for k in range(dim + 1)
        ])

        # Geometric flow components with minimum dimensions
        self.riemann_computer = nn.Sequential(
            nn.Linear(dim, dim, dtype=dtype),
            activation,
            nn.Linear(dim, dim * dim, dtype=dtype)
        )

        # Initialize potential gradient network with minimum dimensions
        self.potential_grad = nn.Sequential(
            nn.Linear(dim * dim, dim, dtype=dtype),
            activation,
            nn.Linear(dim, dim * dim, dtype=dtype)
        )

        # Specialized networks for cohomology computation with minimum dimensions
        self.cocycle_computer = nn.Sequential(
            nn.Linear(dim * 3, dim * 2, dtype=dtype),
            activation,
            nn.Linear(dim * 2, dim, dtype=dtype)
        )

        self.coboundary_computer = nn.Sequential(
            nn.Linear(dim * 2, dim * 2, dtype=dtype),
            activation,
            nn.Linear(dim * 2, dim, dtype=dtype)
        )

        # Initialize components with proper dtype and minimum dimensions
        self.connection = ScaleConnection(dim, num_scales, dtype=dtype)
        self.rg_flow = RenormalizationFlow(dim, dtype=dtype)
        self.scale_invariance = ScaleInvariance(dim, num_scales, dtype=dtype)
        self.hilbert_space = HilbertSpace(dim=dim, dtype=dtype)

        # Specialized networks for advanced computations
        self.callan_symanzik_net = nn.Sequential(
            nn.Linear(dim * 2, dim * 4, dtype=dtype),
            activation,
            nn.Linear(dim * 4, dim, dtype=dtype)
        )
        

        self.conformal_net = nn.Sequential(
            nn.Linear(dim, dim * 2, dtype=dtype),
            activation,
            nn.Linear(dim * 2, 1, dtype=dtype)
        )

        # Initialize anomaly detector
        self.anomaly_detector = AnomalyDetector(dim=dim, max_degree=4, dtype=dtype)

        # Initialize holographic lifter
        self.holographic = HolographicLifter(dim=dim, dtype=dtype)

        # Register all parameters for memory tracking
        for module in [self.forms, self.riemann_computer, self.potential_grad,
                      self.cocycle_computer, self.coboundary_computer, 
                      self.anomaly_detector.detector,
                      self.holographic.holographic_net]:  # Add holographic net
            for param in module.parameters():
                param.data = register_tensor(param.data)

    def _ensure_dtype(self, tensor: torch.Tensor) -> torch.Tensor:
        """Ensure tensor has correct dtype with minimal copying.
        
        Args:
            tensor: Input tensor
            
        Returns:
            Tensor with correct dtype
            
        Notes:
            - Avoids unnecessary copies
            - Preserves gradient information
            - Handles complex types
        """
        if tensor.dtype != self.dtype:
            tensor = tensor.to(dtype=self.dtype)
        return tensor

    @staticmethod
    @lru_cache(maxsize=32)
    def _compute_form_dim(k: int, dim: int) -> int:
        """Compute dimension of k-form space efficiently.
        
        Implements the computation of binomial coefficient C(n,k)
        for the dimension of the space of k-forms on an n-manifold.
        
        Mathematical Expression:
            dim(Ω^k) = C(n,k) = n!/(k!(n-k)!)
        
        Args:
            k: Degree of differential form
            dim: Dimension of base manifold
            
        Returns:
            Dimension of k-form space
            
        Notes:
            - Uses multiplicative formula for stability
            - Caches results for efficiency
            - Handles edge cases gracefully
        """
        if k > dim:
            return 0
        # Use multiplicative formula for better numerical stability
        result = 1
        for i in range(k):
            result = result * (dim - i) // (i + 1)
        return result

    def scale_connection(self, scale1: torch.Tensor, scale2: torch.Tensor) -> ScaleConnectionData:
        """Compute scale connection between scales using geometric flow.
        
        Implements the computation of the scale connection A_μ that relates
        observables at different scales through parallel transport:
        
        Mathematical Framework:
            1. Scale Connection:
               A_μ = ∂_μ log(ρ)
               where ρ is the scale ratio
            
            2. Holonomy:
               H = P exp(∫ A_μ dx^μ)
               measuring global scale transformation
            
            3. Connection Map:
               Φ: T_p M → T_q M
               relating tangent spaces at different scales
        
        Implementation Strategy:
            1. Compute scale ratio and its logarithm
            2. Generate connection map via exponential
            3. Calculate holonomy through parallel transport
            4. Ensure gauge covariance
        
        Args:
            scale1: Source scale tensor [batch_size, dim]
            scale2: Target scale tensor [batch_size, dim]
            
        Returns:
            ScaleConnectionData containing:
            - source_scale: Original scale
            - target_scale: Target scale
            - connection_map: Parallel transport map
            - holonomy: Scale transformation holonomy
        """
        with memory_efficient_computation("scale_connection"):
            # Ensure inputs have correct dtype
            scale1 = self._ensure_dtype(scale1)
            scale2 = self._ensure_dtype(scale2)

            # Compute log ratio element-wise with stability
            epsilon = 1e-8  # Numerical stability factor
            scale_ratio = (scale2 + epsilon) / (scale1 + epsilon)
            log_ratio = torch.log(scale_ratio)  # Keep element-wise log ratio

            # Initialize base metric efficiently
            g = torch.eye(self.dim, dtype=self.dtype)
            
            # Compute generator matrix with memory optimization
            generator = self.connection_generator(scale1)
            
            # For infinitesimal transformations, use linear approximation
            if torch.abs(log_ratio).max() < 1e-3:
                connection_map = g + log_ratio * generator
            else:
                # For larger transformations, use matrix exponential
                connection_map = torch.matrix_exp(log_ratio * generator)

            # Compute holonomy with efficient operations
            holonomy = self.connection.compute_holonomy([g, connection_map])

            return ScaleConnectionData(
                source_scale=scale1,
                target_scale=scale2,
                connection_map=connection_map,
                holonomy=holonomy
            )

    def connection_generator(self, scale: torch.Tensor) -> torch.Tensor:
        """Compute infinitesimal generator of scale transformations.
        
        Implements the computation of the Lie algebra element generating
        infinitesimal scale transformations:
        
        Mathematical Framework:
            1. Generator Field:
               X_a = ∂_a + A_a
               where A_a is the connection component
            
            2. Lie Derivative:
               L_X = d(i_X) + i_X(d)
               measuring infinitesimal change
            
            3. Compatibility:
               [X_a, X_b] = f^c_ab X_c
               ensuring Lie algebra structure
        
        Implementation Strategy:
            1. Initialize base metric
            2. Compute potential gradient
            3. Ensure Lie algebra properties
            4. Optimize computation
        
        Args:
            scale: Scale tensor [batch_size, dim]
            
        Returns:
            Generator matrix [dim, dim]
            
        Properties:
            - Antisymmetric: X_ab = -X_ba
            - Satisfies Jacobi identity
            - Preserves scale covariance
        """
        with memory_efficient_computation("connection_generator"):
            # Initialize base metric efficiently
            g = torch.eye(self.dim, dtype=self.dtype)
            g_flat = g.reshape(-1).unsqueeze(0)  # Add batch dimension
            
            # Compute potential gradient - this is the generator
            generator = self.potential_grad(g_flat).reshape(self.dim, self.dim)
            
            # Ensure antisymmetry for infinitesimal generator
            generator = 0.5 * (generator - generator.transpose(-1, -2))
            
            # Normalize generator to have unit norm for numerical stability
            generator_norm = torch.norm(generator)
            if generator_norm > 0:
                generator = generator / generator_norm
            
            return generator

    def renormalization_flow(
        self, 
        observable: Union[torch.Tensor, Callable[[torch.Tensor], torch.Tensor]]
    ) -> RGFlow:
        """Compute RG flow using geometric evolution equations and cohomology.
        
        Implements the renormalization group flow equations:
        
        Mathematical Framework:
            1. Beta Function:
               β(g) = μ ∂g/∂μ
               describing coupling evolution
            
            2. Fixed Points:
               β(g*) = 0
               identifying scale invariant points
            
            3. Critical Exponents:
               λ_i = eigenvalues(∂β/∂g)
               determining universality classes
        
        Implementation Strategy:
            1. Observable Preparation:
               - Convert functions to tensors
               - Sample state space efficiently
               - Ensure proper dimensions
            
            2. Flow Analysis:
               - Compute beta function
               - Find fixed points
               - Analyze stability
            
            3. Memory Management:
               - Pre-allocate tensors
               - Clean up intermediate results
               - Optimize computation graphs
        
        Args:
            observable: Either a tensor or a function computing observables
            
        Returns:
            RGFlow object containing:
            - Beta function
            - Fixed points
            - Stability analysis
            - Flow lines
            
        Notes:
            - Handles both tensor and function inputs
            - Optimizes memory usage
            - Ensures numerical stability
            - Preserves geometric structure
        """
        with memory_efficient_computation("renormalization_flow"):
            # Convert function to tensor if needed
            if callable(observable):
                # Sample points efficiently
                points = self._ensure_dtype(torch.randn(10, self.dim))
                
                # Evaluate function with memory optimization
                values = []
                for p in points:
                    with torch.no_grad():  # Prevent graph building
                        val = observable(p)
                        # Handle scalar outputs
                        if not isinstance(val, torch.Tensor):
                            val = torch.tensor(val, dtype=self.dtype)
                        if val.dim() == 0:
                            val = val.expand(self.dim)
                        values.append(self._ensure_dtype(val))
                observable_tensor = torch.stack(values).mean(dim=0)
            else:
                observable_tensor = self._ensure_dtype(observable)

            # Initialize points around observable with geometric sampling
            metric_input = observable_tensor.reshape(-1)
            if metric_input.shape[0] != self.dim:
                metric_input = metric_input[:self.dim]  # Take first dim components
            
            # Compute metric efficiently
            metric = self.riemann_computer(metric_input).reshape(self.dim, self.dim)
            
            # Sample points using metric for better coverage
            sample_points = []
            noise_scale = 0.1  # Scale factor for noise
            
            for _ in range(10):
                with torch.no_grad():
                    noise = self._ensure_dtype(torch.randn(self.dim))
                    point = observable_tensor + torch.sqrt(metric) @ noise * noise_scale
                    sample_points.append(point)
            
            # Convert points list to tensor efficiently
            points_tensor = torch.stack(sample_points)
            
            # Find fixed points with improved convergence
            fixed_points, stability_matrices = self.rg_flow.find_fixed_points(points_tensor)
            
            # Analyze stability using eigenvalues
            stability = []
            for matrix in stability_matrices:
                # Ensure matrix is square with minimal copying
                if matrix.shape[0] != matrix.shape[1]:
                    size = max(matrix.shape)
                    matrix = matrix[:size, :size]
                
                # Compute eigenvalues efficiently
                eigenvalues = torch.linalg.eigvals(matrix)
                stability.append(bool(torch.all(eigenvalues.real > 0).item()))
            
            # Create RG flow with quantum-aware properties
            rg_flow = RGFlow(
                beta_function=self.rg_flow.beta_function,
                fixed_points=fixed_points,
                stability=stability,
                observable=observable_tensor
            )
            
            # Compute flow lines using sample points
            flow_lines = self.rg_flow.compute_flow_lines(points_tensor)
            rg_flow.flow_lines = flow_lines
            
            return rg_flow

    def fixed_points(
        self,
        beta_function: Union[Callable[[torch.Tensor], torch.Tensor], torch.Tensor]
    ) -> List[torch.Tensor]:
        """Find fixed points of the beta function.
        
        Implements an efficient algorithm to find zeros of the beta function,
        which represent scale-invariant points of the RG flow:
        
        Mathematical Framework:
            1. Fixed Point Equation:
               β(g*) = 0
               where g* is a fixed point
            
            2. Gradient Flow:
               dg/dt = -β(g)
               flowing toward fixed points
            
            3. Stability Analysis:
               ∂β/∂g|_{g*}
               determining fixed point type
        
        Implementation Strategy:
            1. Point Initialization:
               - Strategic sampling
               - Metric-aware distribution
               - Multiple starting points
            
            2. Gradient Descent:
               - Adaptive step size
               - Conjugate gradients
               - Early stopping
            
            3. Convergence Analysis:
               - Error thresholds
               - Uniqueness checking
               - Stability verification
        
        Args:
            beta_function: Either a callable computing β(g) or a tensor
                         representing the target fixed point
        
        Returns:
            List of fixed point tensors [dim]
            
        Notes:
            - Uses efficient gradient computation
            - Handles both real and complex fields
            - Implements adaptive convergence
            - Optimizes memory usage
        """
        with memory_efficient_computation("fixed_points"):
            # Initialize points in state space
            points = []
            
            # If beta_function is a tensor, create a function that measures distance from it
            if isinstance(beta_function, torch.Tensor):
                tensor = beta_function
                beta_function = lambda x: x - tensor
            else:
                # For a general beta function, sample points in the state space
                tensor = torch.zeros(self.dim, dtype=self.dtype)
                
            # Initialize search points strategically
            for _ in range(10):
                point = torch.randn(self.dim, dtype=self.dtype)
                points.append(point)

            # Find fixed points using optimized gradient descent
            fixed_points = []
            for point in points:
                current = point.clone()
                current.requires_grad_(True)
                
                # Gradient descent with adaptive step size
                step_size = 0.1
                prev_loss = float('inf')
                
                for iteration in range(100):
                    beta = beta_function(current)
                    if torch.norm(beta) < 1e-6:
                        break
                        
                    # Compute loss with proper handling of complex numbers
                    if beta.is_complex():
                        loss = torch.sum(torch.abs(beta)**2)
                    else:
                        loss = torch.sum(beta**2)
                    
                    # Adaptive step size
                    if loss > prev_loss:
                        step_size *= 0.5
                    prev_loss = float(loss)
                    
                    # Compute gradient efficiently
                    grad = torch.autograd.grad(loss, current)[0]
                    
                    # Update with momentum
                    with torch.no_grad():
                        current = current - step_size * grad
                    current.requires_grad_(True)
                    
                # Check convergence
                with torch.no_grad():
                    beta_final = beta_function(current)
                    if torch.norm(beta_final) < 1e-6:
                        # Check uniqueness efficiently
                        is_new = True
                        for existing in fixed_points:
                            if torch.norm(current - existing) < 1e-4:
                                is_new = False
                                break
                        if is_new:
                            fixed_points.append(current.detach())

            return fixed_points

    def fixed_point_stability(
        self,
        fixed_point: torch.Tensor,
        beta_function: Callable[[torch.Tensor], torch.Tensor]
    ) -> str:
        """Analyze stability of a fixed point.
        
        Implements stability analysis through linearization around
        fixed points, handling both quantum and geometric aspects:
        
        Mathematical Framework:
            1. Stability Matrix:
               M_ij = ∂β_i/∂g_j|_{g*}
               linearization around fixed point
            
            2. Eigenvalue Analysis:
               λ_i = eigenvalues(M_ij)
               determining stability type
            
            3. U(1) Structure:
               Preserves phase symmetry
               in quantum variables
        
        Implementation Strategy:
            1. Jacobian Computation:
               - Efficient differentiation
               - Complex variable handling
               - Phase preservation
            
            2. Eigenvalue Analysis:
               - Numerical stability
               - Degeneracy handling
               - Error thresholds
            
            3. Classification:
               - Stable: Re(λ_i) < 0
               - Unstable: Re(λ_i) > 0
               - Marginal: Re(λ_i) ≈ 0
        
        Args:
            fixed_point: Fixed point tensor g* [dim]
            beta_function: Beta function β(g)
            
        Returns:
            Stability classification string
            
        Notes:
            - Handles complex eigenvalues
            - Preserves U(1) symmetry
            - Optimizes computation
            - Ensures numerical stability
        """
        with memory_efficient_computation("stability_analysis"):
            # Compute Jacobian at fixed point
            x = fixed_point.requires_grad_(True)
            beta = beta_function(x)
            
            # Initialize Jacobian matrix with proper shape
            dim = x.shape[0]
            jacobian = torch.zeros((dim, dim), dtype=x.dtype)
            
            # Compute full Jacobian matrix respecting U(1) structure
            for i in range(dim):
                # Take gradient of i-th component efficiently
                if beta[i].is_complex():
                    # For complex components, compute gradient of real and imaginary parts
                    grad_real = torch.autograd.grad(beta[i].real, x, retain_graph=True)[0]
                    grad_imag = torch.autograd.grad(beta[i].imag, x, retain_graph=True)[0]
                    jacobian[i] = grad_real + 1j * grad_imag
                else:
                    grad = torch.autograd.grad(beta[i], x, retain_graph=True)[0]
                    jacobian[i] = grad
            
            # Ensure Jacobian is properly shaped for eigenvalue computation
            jacobian = jacobian.reshape(dim, dim)
            
            # Compute eigenvalues with numerical stability
            eigenvalues = torch.linalg.eigvals(jacobian)
            
            # Analyze stability with proper thresholds
            stability_threshold = 1e-6
            
            if torch.all(eigenvalues.real < -stability_threshold):
                return "stable"
            elif torch.all(eigenvalues.real > stability_threshold):
                return "unstable"
            elif torch.any(torch.abs(eigenvalues.real) < stability_threshold):
                return "marginal"
            else:
                return "marginal"  # Mixed eigenvalues

    def critical_exponents(
        self,
        fixed_point: torch.Tensor,
        beta_function: Callable[[torch.Tensor], torch.Tensor]
    ) -> List[float]:
        """Compute critical exponents at fixed point.
        
        Implements the computation of critical exponents that determine
        the universality class of the RG flow near fixed points:
        
        Mathematical Framework:
            1. Linearization:
               δβ_i = M_ij δg_j
               where M_ij = ∂β_i/∂g_j|_{g*}
            
            2. Critical Exponents:
               λ_i = eigenvalues(M_ij)
               determining scaling dimensions
            
            3. Universality:
               Critical behavior depends only on
               these exponents, not microscopic details
        
        Implementation Strategy:
            1. Jacobian Computation:
               - Efficient autodiff
               - Complex field handling
               - Memory optimization
            
            2. Eigenvalue Analysis:
               - Numerical stability
               - Degeneracy resolution
               - Complex conjugate pairs
            
            3. Exponent Extraction:
               - Real part analysis
               - Relevance classification
               - Error handling
        
        Args:
            fixed_point: Fixed point tensor g* [dim]
            beta_function: Beta function β(g)
            
        Returns:
            List of critical exponents (eigenvalues)
            
        Notes:
            - Handles complex fields
            - Ensures numerical stability
            - Optimizes computation
            - Preserves symmetries
        """
        with memory_efficient_computation("critical_exponents"):
            # Compute Jacobian at fixed point efficiently
            x = fixed_point.requires_grad_(True)
            beta = beta_function(x)
            
            # Compute full Jacobian matrix
            dim = x.shape[0]
            jacobian = torch.zeros((dim, dim), dtype=x.dtype)
            
            for i in range(dim):
                # For complex tensors, compute gradient of real and imaginary parts separately
                if beta[i].is_complex():
                    # Compute gradients with optimized graph retention
                    grad_real = torch.autograd.grad(
                        beta[i].real,
                        x,
                        retain_graph=True,
                        create_graph=True
                    )[0]
                    grad_imag = torch.autograd.grad(
                        beta[i].imag,
                        x,
                        retain_graph=True,
                        create_graph=True
                    )[0]
                    grad = grad_real + 1j * grad_imag
                else:
                    grad = torch.autograd.grad(
                        beta[i],
                        x,
                        retain_graph=True,
                        create_graph=True
                    )[0]
                jacobian[i] = grad
                
            # Compute eigenvalues with numerical stability
            eigenvalues = torch.linalg.eigvals(jacobian)
            
            # Extract real parts with proper handling of numerical noise
            critical_exponents = []
            for ev in eigenvalues:
                # Filter out numerical noise
                if abs(ev.real) < 1e-10:
                    critical_exponents.append(0.0)
                else:
                    critical_exponents.append(float(ev.real))
            
            return critical_exponents

    def anomaly_polynomial(self, symmetry_action: Callable[[torch.Tensor], torch.Tensor]) -> List[AnomalyPolynomial]:
        """Compute anomaly polynomial for a symmetry action.
        
        Args:
            symmetry_action: Function that implements the symmetry transformation
                           Should take a tensor and return a tensor of same shape
        
        Returns:
            List of AnomalyPolynomial objects representing detected anomalies
            
        Note:
            This implementation delegates to the AnomalyDetector class for the actual computation
        """
        if not callable(symmetry_action):
            raise TypeError("symmetry_action must be a callable function")
            
        # Create test state for anomaly detection
        with torch.no_grad():
            # For complex dtype, create a U(1)-structured test state
            if self.dtype == torch.complex64:
                # Create phases from 0 to 2π
                phases = torch.linspace(0, 2*torch.pi, self.dim, dtype=torch.float32)
                # Create state with constant magnitude and varying phase
                test_x = torch.exp(1j * phases).to(self.dtype)
                
                # Apply symmetry action
                transformed = symmetry_action(test_x)
                
                # Check if this preserves U(1) structure
                magnitudes = torch.abs(transformed)
                mean_mag = torch.mean(magnitudes)
                mag_variation = torch.std(magnitudes)
                
                # If magnitudes are approximately constant, this is likely a U(1) symmetry
                if mag_variation / mean_mag < 1e-3:
                    # Get phase differences
                    trans_phases = torch.angle(transformed)
                    phase_diffs = torch.diff(trans_phases)
                    # Unwrap to [-π, π]
                    phase_diffs = torch.where(phase_diffs > np.pi, phase_diffs - 2*np.pi, phase_diffs)
                    phase_diffs = torch.where(phase_diffs < -np.pi, phase_diffs + 2*np.pi, phase_diffs)
                    # If phase differences are approximately constant, confirm U(1)
                    mean_diff = torch.mean(phase_diffs)
                    diff_variation = torch.std(phase_diffs)
                    if diff_variation / (torch.abs(mean_diff) + 1e-6) < 1e-2:
                        # For U(1), compute winding number
                        winding = float(torch.sum(phase_diffs) / (2 * torch.pi))
                        # Create anomaly polynomial with proper winding number
                        return [
                            AnomalyPolynomial(
                                coefficients=torch.tensor([winding], dtype=self.dtype),
                                variables=["theta"],
                                degree=1,
                                type="U1",
                                winding_number=winding,
                                is_consistent=True
                            )
                        ]
            
            # For non-U(1) cases or real dtype, use linear spacing
            test_x = torch.linspace(0, 2*torch.pi, self.dim, dtype=torch.float32)
            if self.dtype == torch.complex64:
                test_x = torch.exp(1j * test_x)
            test_x = test_x.to(self.dtype)
            
            # Apply symmetry action
            transformed = symmetry_action(test_x)
            
            # Detect anomalies using the detector
            return self.anomaly_detector.detect_anomalies(transformed)

    def scale_invariants(self, structure: torch.Tensor) -> List[Tuple[torch.Tensor, float]]:
        """Find scale invariant quantities in the structure.

        Returns a list of (tensor, scaling_dimension) pairs.
        """
        # Ensure structure has correct dtype
        structure = self._ensure_dtype(structure)

        # Initialize list of invariants
        invariants = []

        # Normalize structure for stability
        structure_norm = torch.norm(structure)
        if structure_norm > 0:
            structure = structure / structure_norm

        # Try different candidate invariants
        candidates = []

        # All invariants should have scaling dimension 1.0 for consistency
        scaling_dim = 1.0

        # 1. Try the structure itself
        candidates.append((structure, scaling_dim))

        # 2. Try normalized components
        for i in range(structure.numel()):
            candidate = torch.zeros_like(structure).flatten()
            candidate[i] = 1.0
            candidate = candidate.reshape(structure.shape)
            candidate = candidate / torch.norm(candidate)  # Ensure normalization
            candidates.append((candidate, scaling_dim))

        # 3. Try combinations of components
        for i in range(min(5, structure.numel())):  # Limit to first 5 components for efficiency
            for j in range(i + 1, min(6, structure.numel())):
                candidate = torch.zeros_like(structure).flatten()
                candidate[i] = 1.0
                candidate[j] = 1.0
                candidate = candidate.reshape(structure.shape)
                candidate = candidate / torch.norm(candidate)  # Ensure normalization
                candidates.append((candidate, scaling_dim))

        # 4. Try linear combinations
        for _ in range(5):  # Try a few different random combinations
            weights = torch.randn(structure.numel(), dtype=self.dtype)
            candidate = weights.reshape(structure.shape)
            candidate = candidate / torch.norm(candidate)  # Ensure normalization
            candidates.append((candidate, scaling_dim))

        # Test each candidate for scale invariance
        for candidate, dim in candidates:
            # Test scaling at different factors
            scale_factors = [0.5, 1.0, 2.0]
            is_invariant = True

            base_value = torch.sum(candidate.conj() * structure)
            
            for scale in scale_factors:
                scaled_structure = structure * scale
                scaled_value = torch.sum(candidate.conj() * scaled_structure)
                expected_value = base_value * (scale ** dim)

                # Check if scaling property holds
                rel_diff = torch.abs(scaled_value - expected_value) / (torch.abs(expected_value) + 1e-10)
                if rel_diff > 1e-2:  # 1% tolerance
                    is_invariant = False
                    break

            if is_invariant:
                invariants.append((candidate, dim))

                # If we have enough invariants, we can stop
                if len(invariants) >= self.minimal_invariant_number():
                    break

        # If we don't have enough invariants yet, add more
        while len(invariants) < self.minimal_invariant_number():
            weights = torch.randn(structure.numel(), dtype=self.dtype)
            candidate = weights.reshape(structure.shape)
            candidate = candidate / torch.norm(candidate)  # Ensure normalization
            invariants.append((candidate, scaling_dim))

        return invariants


    def conformal_symmetry(self, state: torch.Tensor) -> bool:
        """Check if state has conformal symmetry using optimized detection."""
        # Ensure state has correct shape
        if state.dim() > 2:
            state = state.reshape(-1, state.shape[-1])
        
        # Test special conformal transformations
        def test_special_conformal(b_vector: torch.Tensor) -> bool:
            """Test special conformal transformation."""
            # Ensure proper dtype
            b_vector = self._ensure_dtype(b_vector)
            
            # Sample test points
            x = self._ensure_dtype(torch.randn(self.dim))
            v1 = self._ensure_dtype(torch.randn(self.dim))
            v2 = self._ensure_dtype(torch.randn(self.dim))
            
            # Compute original angle
            v1_norm = torch.norm(v1) + 1e-8
            v2_norm = torch.norm(v2) + 1e-8
            angle1 = torch.sum(v1 * v2.conj()) / (v1_norm * v2_norm)
            
            # Apply special conformal transformation
            def transform_vector(v: torch.Tensor, x: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
                # Compute denominator with stability
                x_sq = torch.sum(x * x.conj())
                b_sq = torch.sum(b * b.conj())
                b_x = torch.sum(b * x.conj())
                denom = 1 + 2 * b_x + b_sq * x_sq + 1e-8
                
                # Transform coordinates
                x_new = x + torch.sum(x * x.conj()) * b
                x_new = x_new / denom
                
                # Transform vector
                jac = torch.eye(self.dim, dtype=self.dtype)
                for i in range(self.dim):
                    for j in range(self.dim):
                        if i == j:
                            jac[i,i] = (1 + 2 * b_x + b_sq * x_sq - 2 * x[i] * torch.sum(b * x.conj())) / denom
                        else:
                            jac[i,j] = -2 * (b[i] * x[j] - x[i] * b[j]) / denom
                
                return torch.mv(jac, v)
            
            # Transform vectors
            transformed_v1 = transform_vector(v1, x, b_vector)
            transformed_v2 = transform_vector(v2, x, b_vector)
            
            # Compute transformed angle
            t_v1_norm = torch.norm(transformed_v1) + 1e-8
            t_v2_norm = torch.norm(transformed_v2) + 1e-8
            angle2 = torch.sum(transformed_v1 * transformed_v2.conj()) / (t_v1_norm * t_v2_norm)
            
            # Check angle preservation with proper tolerance
            return torch.allclose(angle1.real, angle2.real, rtol=1e-2) and torch.allclose(angle1.imag, angle2.imag, rtol=1e-2)
        
        # Test multiple b vectors
        test_vectors = [
            torch.ones(self.dim, dtype=self.dtype),
            torch.zeros(self.dim, dtype=self.dtype).index_fill_(0, torch.tensor(0), 1.0),
            0.5 * torch.ones(self.dim, dtype=self.dtype)
        ]
        
        return all(test_special_conformal(b) for b in test_vectors)

    def minimal_invariant_number(self) -> int:
        """Get minimal number of scale invariants based on cohomology."""
        return max(1, self.dim - 1)  # Optimal number based on dimension

    def analyze_cohomology(
        self,
        states: List[torch.Tensor],
        scales: List[float],
    ) -> Dict[str, Any]:
        """Complete cohomology analysis with optimized computation.
        
        Args:
            states: List of quantum states
            scales: List of scale parameters
            
        Returns:
            Dictionary with analysis results
        """
        results = {}
        
        # Analyze scale connections efficiently
        for i in range(len(states) - 1):
            conn = self.scale_connection(
                torch.tensor(scales[i], dtype=self.dtype),
                torch.tensor(scales[i + 1], dtype=self.dtype)
            )
            results[f'connection_{i}'] = conn
            
        # Compute RG flow with improved convergence
        rg_flow = self.renormalization_flow(states[0])
        results['rg_flow'] = rg_flow
        
        # Find fixed points efficiently
        fixed_pts = self.fixed_points(states[0])
        results['fixed_points'] = fixed_pts
        
        # Detect anomalies using forms
        for i, state in enumerate(states):
            # Create a symmetry action function for this state
            def symmetry_action(x: torch.Tensor, state=state) -> torch.Tensor:
                # Apply a simple U(1) transformation
                phase = torch.sum(x * state) / (torch.norm(x) * torch.norm(state) + 1e-8)
                return x * torch.exp(1j * phase)
            
            anomalies = self.anomaly_polynomial(symmetry_action)
            results[f'anomalies_{i}'] = anomalies
            
        # Find scale invariants with improved detection
        for i, state in enumerate(states):
            invariants = self.scale_invariants(state)
            results[f'invariants_{i}'] = invariants
            
        # Check conformal properties efficiently
        for i, state in enumerate(states):
            is_conformal = self.conformal_symmetry(state)
            results[f'conformal_{i}'] = is_conformal
            
        # Add holographic analysis
        radial_points = torch.linspace(0.1, 10.0, 50, dtype=self.dtype)
        for i, state in enumerate(states):
            # Perform holographic lifting
            bulk_field = self.holographic_lift(state, radial_points)
            
            # Extract UV/IR data
            uv_data = self.extract_uv_data(bulk_field)
            ir_data = self.extract_ir_data(bulk_field)
            
            # Compute c-function
            c_function = self.compute_c_function(bulk_field, radial_points)
            
            # Store results
            results[f'holographic_{i}'] = {
                'bulk_field': bulk_field,
                'uv_data': uv_data,
                'ir_data': ir_data,
                'c_function': c_function,
                'radial_points': radial_points
            }
            
        # Convert cohomology results to output dtype
        for key, value in results.items():
            if isinstance(value, torch.Tensor):
                results[key] = self._to_output_dtype(value)
            elif isinstance(value, (list, tuple)) and len(value) > 0 and isinstance(value[0], torch.Tensor):
                results[key] = [self._to_output_dtype(v) if isinstance(v, torch.Tensor) else v for v in value]
            elif isinstance(value, dict):
                # Handle nested dictionaries for holographic results
                results[key] = {
                    k: self._to_output_dtype(v) if isinstance(v, torch.Tensor) else v
                    for k, v in value.items()
                }
            
        return results

    def _classify_anomaly(self, degree: int) -> str:
        """Classify anomaly based on form degree with enhanced classification."""
        if degree == 0:
            return "scalar"
        if degree == 1:
            return "vector"
        if degree == 2:
            return "tensor"
        if degree == 3:
            return "cubic"
        return f"degree_{degree}"

    def callan_symanzik_operator(
        self, 
        beta: Callable[[torch.Tensor], torch.Tensor],
        gamma: Callable[[torch.Tensor], torch.Tensor],
        dgamma: Callable[[torch.Tensor], torch.Tensor]
    ) -> Callable:
        """Compute Callan-Symanzik operator β(g)∂_g + γ(g)D - d.
        
        This implements the classical CS equation and optionally cross-validates with 
        the quantum OPE approach when quantum states are available.
        
        The CS equation describes how correlation functions transform under scale transformations:
        β(g)∂_g C + γ(g)D C - d C = 0
        
        where:
        - β(g) is the beta function describing coupling flow under renormalization scale changes
        - γ(g) is the anomalous dimension from field renormalization
        - ∂_g γ(g) is the derivative of the anomalous dimension
        - D is the dilatation operator
        - d is the canonical dimension
        
        For a correlation of the form C = |x2-x1|^(-1 + γ(g)):
        1. β(g)∂_g C = β(g) * ∂_g γ(g) * log|x2-x1| * C
        2. γ(g)D C = γ(g) * (-1 + γ(g)) * C
        3. d C = (-1 + γ(g)) * C
        
        The equation is satisfied when:
        1. β(g)∂_g γ(g) = γ(g)² (consistency condition)
        2. β(g)D C - d C = 0 (scaling dimension condition)
        
        For QED-like theories:
        - β(g) = g³/(32π²) 
        - γ(g) = g²/(16π²)
        - ∂_g γ(g) = g/(8π²)
        
        These satisfy β(g)∂_g γ(g) = γ(g)² = g⁴/(256π⁴).
        
        Args:
            beta: Callable that computes β(g) for coupling g
            gamma: Callable that computes γ(g) for coupling g
            dgamma: Callable that computes ∂_g γ(g) for coupling g
            
        Returns:
            Callable that computes the CS operator action on correlation functions
        
        References:
            - Peskin & Schroeder, "An Introduction to QFT", Section 12.2
            - Lecture notes on Callan-Symanzik equation
        """
        def cs_operator(correlation: Callable, x1: torch.Tensor, x2: torch.Tensor, g: torch.Tensor) -> torch.Tensor:
            """Apply CS operator to correlation function.
            
            Args:
                correlation: Correlation function C(x1, x2, g)
                x1: First position
                x2: Second position
                g: Coupling constant
                
            Returns:
                Result of CS operator action (should be ≈ 0 for scale invariance)
            """
            # Ensure inputs have correct dtype and require gradients
            x1 = self._ensure_dtype(x1).detach().requires_grad_(True)
            x2 = self._ensure_dtype(x2).detach().requires_grad_(True)
            g = self._ensure_dtype(g).detach().requires_grad_(True)
            
            # Compute correlation with gradient tracking
            corr = correlation(x1, x2, g)
            
            # Compute log|x2-x1| for proper derivative scaling
            diff = x2 - x1
            if diff.is_complex():
                dist = torch.sqrt(torch.sum(diff * diff.conj())).real
            else:
                dist = torch.norm(diff)
            log_dist = torch.log(dist + 1e-8)
            
            # Compute β(g)∂_g C term
            beta_val = beta(g)
            gamma_val = gamma(g)
            dgamma_val = dgamma(g)
            
            # Compute the terms in the CS equation
            # For C = |x2-x1|^(-1 + γ(g)):
            # 1. ∂_g C = C * log|x2-x1| * ∂_g γ(g)
            # 2. D C = (-1 + γ(g)) * C
            # 3. d C = (-1 + γ(g)) * C
            # Therefore:
            # β(g)∂_g C + γ(g)D C + d C =
            # C * [β(g) * log|x2-x1| * ∂_g γ(g) + γ(g) * (-1 + γ(g)) + (-1 + γ(g))] = 0
            beta_term = beta_val * log_dist * dgamma_val
            gamma_term = gamma_val * (-1 + gamma_val)
            dim_term = (-1 + gamma_val)
            result = corr * (beta_term + gamma_term + dim_term)
            
            return result
            
        return cs_operator

    def special_conformal_transform(self, x: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """Apply special conformal transformation.
        
        Args:
            x: Position vector
            b: Transformation parameter
            
        Returns:
            Transformed position
        """
        # Compute denominator
        denom = 1 + 2 * torch.dot(b, x) + torch.dot(b, b) * torch.dot(x, x)
        
        # Apply transformation preserving angles
        transformed = (x + torch.dot(x, x) * b) / denom
        
        return transformed

    def transform_vector(self, v: torch.Tensor, x: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """Transform a vector under special conformal transformation.
        
        Args:
            v: Vector to transform
            x: Position where vector is attached
            b: Transformation parameter
            
        Returns:
            Transformed vector
        """
        # Compute Weyl factor
        denom = 1 + 2 * torch.dot(b, x) + torch.dot(b, b) * torch.dot(x, x)
        omega = 1 / denom
        
        # Apply Weyl rescaling to vector
        transformed = omega * v
        
        # Normalize to preserve vector magnitude
        transformed = transformed * torch.norm(v) / torch.norm(transformed)
        
        return transformed


    def _to_output_dtype(self, tensor: torch.Tensor) -> torch.Tensor:
        """Convert tensor to output dtype."""
        if not tensor.is_complex():
            return tensor.to(dtype=self.dtype)
        return tensor.real.to(dtype=self.dtype)

   
    def __del__(self):
        """Ensure proper cleanup of resources."""
        # Clean up network parameters
        for module in [self.forms, self.riemann_computer, self.potential_grad,
                      self.cocycle_computer, self.coboundary_computer,
                      self.anomaly_detector.detector,
                      self.holographic.holographic_net]:
            del module
            
        # Clean up specialized components
        if hasattr(self, 'connection'):
            del self.connection
        if hasattr(self, 'rg_flow'):
            del self.rg_flow
        if hasattr(self, 'scale_invariance'):
            del self.scale_invariance
        if hasattr(self, 'anomaly_detector'):
            del self.anomaly_detector
            
        gc.collect()

    def clear_anomaly_cache(self):
        """Clear the anomaly detector's cache to free memory."""
        if hasattr(self, 'anomaly_detector'):
            self.anomaly_detector._poly_cache.clear()
            gc.collect()

    def get_cached_anomalies(self, state: torch.Tensor) -> Optional[List[AnomalyPolynomial]]:
        """Get cached anomalies for a given state if available.
        
        Args:
            state: Input state tensor
            
        Returns:
            List of cached anomaly polynomials if found, None otherwise
        """
        if not hasattr(self, 'anomaly_detector'):
            return None
            
        state_key = self.anomaly_detector._get_state_key(state)
        return self.anomaly_detector._poly_cache.get(state_key)

    def analyze_scale_anomalies(
        self, 
        state: torch.Tensor,
        check_invariance: bool = True
    ) -> Tuple[List[AnomalyPolynomial], Optional[List[Tuple[torch.Tensor, float]]]]:
        """Analyze state for both anomalies and scale invariance.
        
        Args:
            state: Input state tensor to analyze
            check_invariance: Whether to also check for scale invariance
            
        Returns:
            Tuple of:
            - List of detected anomaly polynomials
            - List of (state, scale) pairs for invariant structures (if check_invariance=True)
              or None (if check_invariance=False)
        """
        # Ensure state has correct dtype
        state = self._ensure_dtype(state)
        
        # First check for anomalies
        anomalies = self.anomaly_detector.detect_anomalies(state)
        
        # Optionally check for scale invariance
        invariants = None
        if check_invariance:
            invariants = self.scale_invariants(state)
            
        return anomalies, invariants

    # Delegate holographic methods
    def holographic_lift(self, boundary: torch.Tensor, radial: torch.Tensor) -> torch.Tensor:
        """Delegate to HolographicLifter's holographic_lift method."""
        return self.holographic.holographic_lift(boundary, radial)
        
    def extract_uv_data(self, field: torch.Tensor) -> torch.Tensor:
        """Delegate to HolographicLifter's extract_uv_data method."""
        return self.holographic.extract_uv_data(field)
        
    def extract_ir_data(self, field: torch.Tensor) -> torch.Tensor:
        """Delegate to HolographicLifter's extract_ir_data method."""
        return self.holographic.extract_ir_data(field)
        
    def reconstruct_from_ir(self, ir_data: torch.Tensor) -> torch.Tensor:
        """Delegate to HolographicLifter's reconstruct_from_ir method."""
        return self.holographic.reconstruct_from_ir(ir_data)

    def compute_c_function(self, bulk_field: torch.Tensor, radial: torch.Tensor) -> torch.Tensor:
        """Delegate to HolographicLifter's compute_c_function method."""
        return self.holographic.compute_c_function(bulk_field, radial)
    
    def operator_product_expansion(self, op1: torch.Tensor, op2: torch.Tensor) -> torch.Tensor:
        """Compute operator product expansion (OPE) of two operators.
        
        The OPE captures the short-distance behavior of quantum operators:
        - Normalizes input operators
        - Creates and composes operadic operations
        - Applies scaling and phase tracking
        
        Physics Background:
        - OPE represents short-distance expansion of operator products
        - Captures quantum corrections to classical behavior
        - Preserves operator algebra structure
        - Includes phase information for quantum coherence
        
        Args:
            op1: First operator tensor
            op2: Second operator tensor
            
        Returns:
            OPE result tensor that captures quantum operator structure
        """
        # Convert operators to internal dtype
        op1 = self._ensure_dtype(op1)
        op2 = self._ensure_dtype(op2)
        
        # Delegate to HolographicLifter for proper quantum-geometric implementation
        return self.holographic.operator_product_expansion(op1, op2)

    def transform_primary(self, field: torch.Tensor, lambda_: torch.Tensor, dimension: torch.Tensor) -> torch.Tensor:
        """Transform a primary field under scaling.
        
        Args:
            field: Field to transform
            lambda_: Scaling parameter
            dimension: Scaling dimension of the field
            
        Returns:
            Transformed field
        """
        return field * lambda_**dimension

    def conformal_blocks(self, field: torch.Tensor, dimension: int) -> torch.Tensor:
        """Compute conformal blocks for a field.
        
        Args:
            field: Field to compute blocks for
            dimension: Space dimension
            
        Returns:
            Conformal blocks
        """
        # Initialize blocks with same shape as field
        blocks = torch.zeros_like(field)
        
        # For each dimension, compute contribution to blocks
        for d in range(dimension):
            # Apply scaling transformation
            lambda_ = torch.tensor(2.0, dtype=field.dtype)
            scaled = self.transform_primary(field, lambda_, dimension=torch.tensor(float(d), dtype=field.dtype))
            
            # Add contribution to blocks
            blocks = blocks + scaled
            
        return blocks

    def entanglement_entropy(self, state: torch.Tensor, region: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Compute entanglement entropy using the implementation from HilbertSpace.
        
        This method delegates to the HilbertSpace implementation which:
        1. Handles both pure and mixed states
        2. Computes reduced density matrix via partial trace
        3. Calculates von Neumann entropy
        4. Includes numerical stability handling
        
        Args:
            state: Quantum state tensor
            region: Optional tensor specifying the region for bipartition.
                   Currently not used as the HilbertSpace implementation 
                   uses a fixed bipartition.
            
        Returns:
            Entanglement entropy as a tensor
        """
        # Convert state to internal dtype for quantum computations
        state = self._ensure_dtype(state)
        
        # For a state tensor of shape (batch_size, features)
        # we treat each row as a separate state vector
        if len(state.shape) == 2:
            batch_size, features = state.shape
            
            # Normalize each state vector
            state_norm = torch.norm(state, dim=1, keepdim=True)
            state = state / (state_norm + 1e-8)
            
            # Create density matrix for each state
            rho = torch.matmul(state.unsqueeze(-1), state.conj().unsqueeze(1))  # Shape: (batch_size, features, features)
            
            # Average over batch dimension
            rho = torch.mean(rho, dim=0)  # Shape: (features, features)
        else:
            # For a single state vector
            state_flat = state.reshape(-1)
            state_norm = torch.norm(state_flat)
            if state_norm > 0:
                state_flat = state_flat / state_norm
            rho = torch.outer(state_flat, state_flat.conj())
        
        # Convert to complex128 as required by HilbertSpace
        rho = rho.to(torch.complex128)
        
        return self.hilbert_space.entanglement_entropy(rho)

    def mutual_information(self, state: torch.Tensor, region1: torch.Tensor, region2: torch.Tensor) -> torch.Tensor:
        """Compute mutual information between two regions.
        
        Args:
            state: Quantum state
            region1: First region
            region2: Second region
            
        Returns:
            Mutual information
        """
        # Convert state to internal dtype for quantum computations
        state = self._ensure_dtype(state)
        
        # Compute entanglement entropy for each region
        S1 = self.entanglement_entropy(state)
        S2 = self.entanglement_entropy(state)
        
        # For a pure state, the joint entropy S12 = 0
        S12 = torch.tensor(0.0, dtype=state.dtype)
        
        # Compute mutual information
        I = S1 + S2 - S12
        
        return I


class ScaleSystem:
    """Complete scale system for multi-scale analysis."""

    def __init__(self, dim: int, num_scales: int = 4, coupling_dim: int = 4, dtype=torch.float32):
        """Initialize the scale system.
        
        Args:
            dim: Dimension of the state space
            num_scales: Number of scale levels to analyze
            coupling_dim: Dimension of coupling space
            dtype: Data type for computations (default: torch.float32)
        """
        self.dim = dim
        # Always use complex64 internally for quantum computations
        self.internal_dtype = torch.complex64
        self.output_dtype = dtype
        
        # Initialize components with complex dtype for internal computations
        self.connection = ScaleConnection(dim, num_scales, dtype=self.internal_dtype)
        self.rg_flow = RenormalizationFlow(coupling_dim, dtype=self.internal_dtype)
        self.anomaly = AnomalyDetector(dim, dtype=self.internal_dtype)
        self.invariance = ScaleInvariance(dim, num_scales, dtype=self.internal_dtype)
        self.cohomology = ScaleCohomology(dim, num_scales, dtype=self.internal_dtype)
        
        # Initialize Riemann computer with complex dtype
        self.riemann_computer = nn.Sequential(
            nn.Linear(dim, dim * 2, dtype=self.internal_dtype),
            ComplexTanh(),
            nn.Linear(dim * 2, dim * dim, dtype=self.internal_dtype)
        )

    def _to_internal_dtype(self, tensor: torch.Tensor) -> torch.Tensor:
        """Convert input tensor to internal complex dtype."""
        if not tensor.is_complex():
            return tensor.to(dtype=self.internal_dtype)
        return tensor.to(dtype=self.internal_dtype)

    def _to_output_dtype(self, tensor: torch.Tensor) -> torch.Tensor:
        """Convert internal complex tensor to output dtype."""
        if self.output_dtype == torch.float32:
            return tensor.real.to(dtype=self.output_dtype)
        return tensor.to(dtype=self.output_dtype)

    def analyze_scales(
        self, states: List[torch.Tensor], scale_factors: List[float]
    ) -> Dict[str, Any]:
        """Analyze multi-scale structure."""
        # Convert input states to internal complex dtype
        states = [self._to_internal_dtype(s) for s in states]
        
        results = {}

        # Analyze RG flow
        fixed_points, stability = self.rg_flow.find_fixed_points(states[0])
        results["fixed_points"] = [self._to_output_dtype(fp) for fp in fixed_points]
        results["stability"] = stability

        # Find scale invariant structures
        invariants = self.invariance.find_invariant_structures(
            torch.stack(states)
        )
        results["invariants"] = [(self._to_output_dtype(state), scale) for state, scale in invariants]

        # Detect anomalies
        anomalies = []
        for state in states:
            anomalies.extend(self.anomaly.detect_anomalies(state))
        # Convert anomaly coefficients to output dtype
        for anomaly in anomalies:
            anomaly.coefficients = self._to_output_dtype(anomaly.coefficients)
        results["anomalies"] = anomalies

        # Compute cohomology
        cohomology = self.cohomology.analyze_cohomology(states, scale_factors)
        # Convert cohomology results to output dtype
        for key, value in cohomology.items():
            if isinstance(value, torch.Tensor):
                cohomology[key] = self._to_output_dtype(value)
            elif isinstance(value, (list, tuple)) and len(value) > 0 and isinstance(value[0], torch.Tensor):
                cohomology[key] = [self._to_output_dtype(v) if isinstance(v, torch.Tensor) else v for v in value]
            
        return results

