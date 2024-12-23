"""Scale System Implementation for Crystal Structures.

This module implements the scale system for multi-scale analysis:
- Scale connections between layers
- Renormalization group flows
- Fixed point detection
- Anomaly polynomials
- Scale invariant structures
"""

from dataclasses import dataclass
from typing import Callable, List, Tuple, Dict, Any

import numpy as np
import torch
from torch import nn


@dataclass
class ScaleConnectionData:
    """Represents a connection between different scales."""

    source_scale: float
    target_scale: float
    connection_map: torch.Tensor  # Linear map between scales
    holonomy: torch.Tensor  # Parallel transport around scale loop


@dataclass
class RGFlow:
    """Represents a renormalization group flow."""

    beta_function: Callable[[torch.Tensor], torch.Tensor]
    fixed_points: List[torch.Tensor]
    stability: List[torch.Tensor]  # Stability matrices at fixed points
    flow_lines: List[torch.Tensor]  # Trajectories in coupling space


@dataclass
class AnomalyPolynomial:
    """Represents an anomaly polynomial."""

    coefficients: torch.Tensor  # Polynomial coefficients
    variables: List[str]  # Variable names
    degree: int  # Polynomial degree
    type: str  # Type of anomaly


class ScaleConnection:
    """Implementation of scale connections."""

    def __init__(self, dim: int, num_scales: int = 4, dtype=torch.float32):
        """Initialize scale connection.
        
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
        self.connections = nn.ModuleList(
            [nn.Linear(dim, dim, dtype=dtype) for _ in range(num_scales - 1)]
        )

        # Holonomy computation with correct output dimension
        self.holonomy_computer = nn.Sequential(
            nn.Linear(dim * 2, dim * 4, dtype=dtype),
            nn.ReLU(),
            nn.Linear(dim * 4, dim * dim, dtype=dtype)  # Output will be reshaped to dim x dim
        )

    def connect_scales(
        self, source_state: torch.Tensor, source_scale: float, target_scale: float
    ) -> torch.Tensor:
        """Connect states at different scales."""
        scale_idx = int(np.log2(target_scale / source_scale))
        if scale_idx >= len(self.connections):
            raise ValueError("Scale difference too large")

        return self.connections[scale_idx](source_state)

    def compute_holonomy(self, states: List[torch.Tensor]) -> torch.Tensor:
        """Compute holonomy for a loop of scale transformations."""
        # Ensure states have batch dimension
        states_batch = [s.unsqueeze(0) if s.dim() == 2 else s for s in states]
        
        # Concatenate initial and final states along feature dimension
        loop_states = torch.cat([states_batch[0], states_batch[-1]], dim=-1)
        
        # Compute holonomy and reshape to dim x dim matrix
        holonomy = self.holonomy_computer(loop_states)
        return holonomy.view(-1, self.dim, self.dim).squeeze(0)  # Remove batch dimension


class RenormalizationFlow:
    """Implementation of renormalization group flows."""

    def __init__(self, coupling_dim: int, max_iter: int = 100, dtype=torch.float32):
        """Initialize RG flow.
        
        Args:
            coupling_dim: Dimension of coupling space (must be positive)
            max_iter: Maximum number of iterations
            dtype: Data type for tensors
            
        Raises:
            ValueError: If coupling_dim <= 0
        """
        if coupling_dim <= 0:
            raise ValueError(f"Coupling dimension must be positive, got {coupling_dim}")
            
        self.coupling_dim = coupling_dim
        self.max_iter = max_iter
        self.dtype = dtype

        # Beta function network
        hidden_dim = max(coupling_dim * 2, 1)  # Ensure hidden dimension is at least 1
        self.beta_network = nn.Sequential(
            nn.Linear(coupling_dim, hidden_dim, dtype=dtype),
            nn.Tanh(),
            nn.Linear(hidden_dim, coupling_dim, dtype=dtype),
        )

        # Fixed point detector
        self.fp_detector = nn.Sequential(
            nn.Linear(coupling_dim, hidden_dim, dtype=dtype),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1, dtype=dtype),
            nn.Sigmoid(),
        )

    def beta_function(self, couplings: torch.Tensor) -> torch.Tensor:
        """Compute beta function at given couplings."""
        return self.beta_network(couplings)

    def find_fixed_points(
        self, initial_points: torch.Tensor
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        """Find fixed points and compute their stability."""
        fixed_points = []
        stability_matrices = []

        for point in initial_points:
            # Flow to fixed point
            current = point.clone()
            for _ in range(self.max_iter):
                beta = self.beta_function(current)
                if torch.norm(beta) < 1e-6:
                    break
                current -= 0.1 * beta

            # Check if point is fixed
            if self.fp_detector(current) > 0.5:
                fixed_points.append(current)

                # Compute stability matrix
                current.requires_grad_(True)
                beta = self.beta_function(current)
                stability = torch.autograd.functional.jacobian(
                    self.beta_function, current
                )
                stability_matrices.append(stability)

        return fixed_points, stability_matrices

    def compute_flow_lines(
        self, start_points: torch.Tensor, num_steps: int = 50
    ) -> List[torch.Tensor]:
        """Compute RG flow lines from starting points."""
        flow_lines = []

        for point in start_points:
            line = [point.clone()]
            current = point.clone()

            for _ in range(num_steps):
                beta = self.beta_function(current)
                current -= 0.1 * beta
                line.append(current.clone())

            flow_lines.append(torch.stack(line))

        return flow_lines


class AnomalyDetector:
    """Detection and analysis of anomalies."""

    def __init__(self, dim: int, max_degree: int = 4, dtype=torch.float32):
        self.dim = dim
        self.max_degree = max_degree
        self.dtype = dtype

        # Anomaly detection network
        self.detector = nn.Sequential(
            nn.Linear(dim, dim * 2, dtype=dtype),
            nn.ReLU(),
            nn.Linear(dim * 2, max_degree + 1, dtype=dtype)
        )

        # Variable names for polynomials
        self.variables = [f"x_{i}" for i in range(dim)]

    def detect_anomalies(self, state: torch.Tensor) -> List[AnomalyPolynomial]:
        """Detect anomalies in quantum state."""
        anomalies = []

        # Analyze state for different polynomial degrees
        coefficients = self.detector(state)

        for degree in range(self.max_degree + 1):
            if torch.norm(coefficients[degree:]) > 1e-6:
                anomalies.append(
                    AnomalyPolynomial(
                        coefficients=coefficients[degree:],
                        variables=self.variables[: degree + 1],
                        degree=degree,
                        type=self._classify_anomaly(degree),
                    )
                )

        return anomalies

    def _classify_anomaly(self, degree: int) -> str:
        """Classify type of anomaly based on degree."""
        if degree == 1:
            return "linear"
        if degree == 2:
            return "quadratic"
        if degree == 3:
            return "cubic"
        return f"degree_{degree}"


class ScaleInvariance:
    """Analysis of scale invariant structures."""

    def __init__(self, dim: int, num_scales: int = 4, dtype=torch.float32):
        """Initialize scale invariance detector.
        
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

        # Scale transformation network
        self.scale_transform = nn.ModuleList(
            [nn.Linear(dim, dim, dtype=dtype) for _ in range(num_scales)]
        )

        # Invariant detector
        hidden_dim = max(dim * 2, 1)  # Ensure hidden dimension is at least 1
        self.invariant_detector = nn.Sequential(
            nn.Linear(dim * 2, hidden_dim * 2, dtype=dtype),
            nn.ReLU(),
            nn.Linear(hidden_dim * 2, 1, dtype=dtype),
            nn.Sigmoid()
        )

    def check_invariance(self, state: torch.Tensor, scale_factor: float) -> bool:
        """Check if state is invariant under scale transformation."""
        # Transform state
        scale_idx = min(int(np.log2(scale_factor)), self.num_scales - 1)
        transformed = self.scale_transform[scale_idx](state)

        # Check invariance
        combined = torch.cat([state, transformed], dim=-1)
        invariance = self.invariant_detector(combined)

        return invariance.item() > 0.5

    def find_invariant_structures(
        self, states: torch.Tensor
    ) -> List[Tuple[torch.Tensor, float]]:
        """Find scale invariant structures and their scale factors."""
        invariants = []

        for state in states:
            for scale in [2**i for i in range(self.num_scales)]:
                if self.check_invariance(state, scale):
                    invariants.append((state, scale))

        return invariants


class ScaleCohomology:
    """Multi-scale cohomological structure for crystal analysis.
    
    This class implements the scale cohomology system following the theoretical framework:
    - De Rham complex for differential forms
    - Geometric flow equations for scale evolution
    - Renormalization group analysis
    - Anomaly detection and classification
    - Scale invariance analysis
    """

    def __init__(self, dim: int, num_scales: int = 4, dtype=torch.float32):
        """Initialize the scale cohomology system.
        
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

        # De Rham complex components (Ω^k forms)
        self.forms = nn.ModuleList([
            nn.Sequential(
                nn.Linear(max(self._compute_form_dim(k, dim), 1), max(dim * 2, 1), dtype=dtype),
                nn.ReLU(),
                nn.Linear(max(dim * 2, 1), max(self._compute_form_dim(k + 1, dim), 1), dtype=dtype)
            ) for k in range(dim + 1)
        ])

        # Geometric flow components with optimized architecture
        hidden_dim = max(dim * dim * 2, 1)  # Ensure hidden dimension is at least 1
        self.riemann_computer = nn.Sequential(
            nn.Linear(max(dim * dim, 1), hidden_dim, dtype=dtype),
            nn.ReLU(),
            nn.Linear(hidden_dim, max(dim * dim, 1), dtype=dtype)
        )

        # Initialize potential gradient network with correct dimensions
        # and proper initialization for infinitesimal generator
        self.potential_grad = nn.Sequential(
            nn.Linear(max(dim * dim, 1), hidden_dim, dtype=dtype),
            nn.ReLU(),
            nn.Linear(hidden_dim, max(dim * dim, 1), dtype=dtype)
        )
        # Initialize the last layer to be close to zero
        # This ensures the infinitesimal generator is small
        with torch.no_grad():
            self.potential_grad[-1].weight.data *= 0.01
            self.potential_grad[-1].bias.data *= 0.01

        # Specialized networks for cohomology computation
        hidden_dim = max(dim * 4, 1)  # Ensure hidden dimension is at least 1
        self.cocycle_computer = nn.Sequential(
            nn.Linear(max(dim * 3, 1), hidden_dim, dtype=dtype),
            nn.ReLU(),
            nn.Linear(hidden_dim, max(dim, 1), dtype=dtype)
        )

        self.coboundary_computer = nn.Sequential(
            nn.Linear(max(dim * 2, 1), hidden_dim, dtype=dtype),
            nn.ReLU(),
            nn.Linear(hidden_dim, max(dim, 1), dtype=dtype)
        )

        # Initialize components with optimized parameters
        self.connection = ScaleConnection(dim, num_scales, dtype=dtype)
        self.rg_flow = RenormalizationFlow(dim, dtype=dtype)
        self.anomaly_detector = AnomalyDetector(dim, dtype=dtype)
        self.scale_invariance = ScaleInvariance(dim, num_scales, dtype=dtype)

        # Specialized networks for advanced computations
        hidden_dim = max(dim * 4, 1)  # Ensure hidden dimension is at least 1
        self.callan_symanzik_net = nn.Sequential(
            nn.Linear(max(dim * 2, 1), hidden_dim, dtype=dtype),
            nn.ReLU(),
            nn.Linear(hidden_dim, max(dim, 1), dtype=dtype)
        )
        
        self.ope_net = nn.Sequential(
            nn.Linear(max(dim * 2, 1), hidden_dim, dtype=dtype),
            nn.ReLU(),
            nn.Linear(hidden_dim, max(dim, 1), dtype=dtype)
        )

        self.conformal_net = nn.Sequential(
            nn.Linear(max(dim, 1), max(dim * 2, 1), dtype=dtype),
            nn.ReLU(),
            nn.Linear(max(dim * 2, 1), 1, dtype=dtype)
        )

    @staticmethod
    def _compute_form_dim(k: int, dim: int) -> int:
        """Compute dimension of k-form space using optimized binomial calculation."""
        if k > dim:
            return 0
        # Use multiplicative formula for better numerical stability
        result = 1
        for i in range(k):
            result = result * (dim - i) // (i + 1)
        return result

    def scale_connection(self, scale1: torch.Tensor, scale2: torch.Tensor) -> ScaleConnectionData:
        """Compute scale connection between scales using geometric flow.
        
        The connection satisfies the compatibility condition:
        c13 = c23 ∘ c12
        
        This is achieved by using the logarithmic scale ratio to compute
        the connection map, ensuring that composition of connections
        corresponds to addition of logarithms.
        """
        # Convert scales to float and compute ratio
        s1 = float(scale1.item())
        s2 = float(scale2.item())
        log_ratio = np.log(s2 / s1)

        # Initialize base metric
        g = torch.eye(self.dim, dtype=self.dtype)
        
        # Compute generator matrix
        g_flat = g.reshape(-1).unsqueeze(0)  # Add batch dimension
        F = self.potential_grad(g_flat).reshape(self.dim, self.dim)
        
        # Compute connection map using matrix exponential
        # This ensures compatibility since:
        # exp(log(s3/s1) F) = exp(log(s3/s2) F) @ exp(log(s2/s1) F)
        connection_map = torch.matrix_exp(log_ratio * F)

        # Compute holonomy
        holonomy = self.connection.compute_holonomy([g, connection_map])

        return ScaleConnectionData(
            source_scale=s1,
            target_scale=s2,
            connection_map=connection_map,
            holonomy=holonomy
        )

    def connection_generator(self, scale: torch.Tensor) -> torch.Tensor:
        """Compute infinitesimal generator of scale transformations.
        
        The generator represents the infinitesimal change in the connection
        under scale transformations. It is computed from the potential
        gradient to ensure compatibility with the scale connection.
        """
        # Initialize base metric
        g = torch.eye(self.dim, dtype=self.dtype)
        g_flat = g.reshape(-1)
        
        # Compute potential gradient - this is the generator
        generator = self.potential_grad(g_flat).reshape(self.dim, self.dim)
        
        return generator

    def renormalization_flow(self, observable: torch.Tensor) -> RGFlow:
        """Compute RG flow using geometric evolution equations and cohomology."""
        if callable(observable):
            # If observable is a function, create a tensor from its evaluation
            x = torch.zeros(self.dim, dtype=self.dtype)
            observable_tensor = observable(x).unsqueeze(0)
        else:
            observable_tensor = observable

        # Get initial points around observable with optimized sampling
        points = observable_tensor + 0.1 * torch.randn(10, self.dim, dtype=self.dtype)
        
        # Find fixed points and stability with improved convergence
        fixed_points, stability = self.rg_flow.find_fixed_points(points)
        
        # Compute flow lines with geometric guidance
        flow_lines = []
        for point in points:
            line = [point]
            current = point.clone()
            
            for _ in range(50):
                # Use geometric flow for updates
                g = self.scale_connection(current, current + 0.1).connection_map
                current = g @ current
                line.append(current.clone())
            
            flow_lines.append(torch.stack(line))
        
        return RGFlow(
            beta_function=self.rg_flow.beta_function,
            fixed_points=fixed_points,
            stability=stability,
            flow_lines=flow_lines
        )

    def fixed_points(self, observable: torch.Tensor) -> List[torch.Tensor]:
        """Find fixed points using cohomological methods and improved convergence."""
        if callable(observable):
            # If observable is a function, create a tensor from its evaluation
            x = torch.zeros(self.dim, dtype=self.dtype)
            observable_tensor = observable(x).unsqueeze(0)
        else:
            observable_tensor = observable

        points = observable_tensor + 0.1 * torch.randn(10, self.dim, dtype=self.dtype)
        fixed_points, _ = self.rg_flow.find_fixed_points(points)
        return fixed_points

    def fixed_point_stability(self, fixed_point: torch.Tensor, beta_function: Callable[[torch.Tensor], torch.Tensor]) -> str:
        """Analyze stability of a fixed point."""
        # Compute Jacobian at fixed point
        x = fixed_point.requires_grad_(True)
        beta = beta_function(x)
        grad = torch.autograd.grad(beta.sum(), x)[0]
        eigenvalues = torch.linalg.eigvals(grad)
        
        # Classify stability based on eigenvalue signs
        if torch.all(eigenvalues.real < 0):
            return "stable"
        elif torch.all(eigenvalues.real > 0):
            return "unstable"
        else:
            return "marginal"

    def critical_exponents(self, fixed_point: torch.Tensor, beta_function: Callable[[torch.Tensor], torch.Tensor]) -> List[float]:
        """Compute critical exponents at fixed point."""
        # Compute Jacobian at fixed point
        x = fixed_point.requires_grad_(True)
        beta = beta_function(x)
        grad = torch.autograd.grad(beta.sum(), x)[0]
        eigenvalues = torch.linalg.eigvals(grad)
        
        # Critical exponents are related to eigenvalues
        return [-ev.real.item() for ev in eigenvalues]

    def anomaly_polynomial(self, state: torch.Tensor) -> List[AnomalyPolynomial]:
        """Compute anomaly polynomials using differential forms and cohomology."""
        if callable(state):
            # If state is a function (symmetry action), evaluate it on test points
            test_points = torch.randn(10, self.dim, dtype=self.dtype)
            state_tensor = torch.stack([state(x) for x in test_points])
        else:
            state_tensor = state

        # Use forms for anomaly detection
        anomalies = []
        for k in range(self.dim + 1):
            omega = self.forms[k](state_tensor)
            if torch.norm(omega) > 1e-6:
                anomalies.append(
                    AnomalyPolynomial(
                        coefficients=omega,
                        variables=[f"x_{i}" for i in range(k + 1)],
                        degree=k,
                        type=self._classify_anomaly(k)
                    )
                )
        return anomalies

    def scale_invariants(self, structure: torch.Tensor) -> List[Tuple[torch.Tensor, float]]:
        """Find scale invariant structures using differential forms."""
        return self.scale_invariance.find_invariant_structures(structure.unsqueeze(0))

    def operator_product_expansion(self, op1: torch.Tensor, op2: torch.Tensor) -> torch.Tensor:
        """Compute operator product expansion with improved efficiency."""
        combined = torch.cat([op1, op2], dim=-1)
        return self.ope_net(combined)

    def conformal_symmetry(self, state: torch.Tensor) -> bool:
        """Check if state has conformal symmetry using optimized detection."""
        return bool(self.conformal_net(state).item() > 0.5)

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
            anomalies = self.anomaly_polynomial(state)
            results[f'anomalies_{i}'] = anomalies
            
        # Find scale invariants with improved detection
        for i, state in enumerate(states):
            invariants = self.scale_invariants(state)
            results[f'invariants_{i}'] = invariants
            
        # Check conformal properties efficiently
        for i, state in enumerate(states):
            is_conformal = self.conformal_symmetry(state)
            results[f'conformal_{i}'] = is_conformal
            
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

    def callan_symanzik_operator(self, beta: Callable[[torch.Tensor], torch.Tensor], 
                              gamma: Callable[[torch.Tensor], torch.Tensor]) -> Callable:
        """Compute Callan-Symanzik operator β(g)∂_g + γ(g)Δ - d.
        
        The Callan-Symanzik operator describes how correlation functions
        change under scale transformations.
        
        Args:
            beta: Beta function β(g) describing coupling evolution
            gamma: Anomalous dimension γ(g)
            
        Returns:
            Callable that applies the CS operator to correlation functions
        """
        def cs_operator(correlation: Callable, x1: torch.Tensor, x2: torch.Tensor, g: torch.Tensor) -> torch.Tensor:
            # Compute β(g)∂_g term
            g.requires_grad_(True)
            corr = correlation(x1, x2, g)
            grad_g = torch.autograd.grad(corr, g, create_graph=True)[0]
            beta_term = beta(g) * grad_g
            
            # Compute γ(g)Δ term using Laplacian
            x1.requires_grad_(True)
            x2.requires_grad_(True)
            grad_x1 = torch.autograd.grad(corr, x1, create_graph=True)[0]
            grad_x2 = torch.autograd.grad(corr, x2, create_graph=True)[0]
            laplacian = torch.sum(grad_x1**2 + grad_x2**2)
            gamma_term = gamma(g) * laplacian
            
            # Combine terms with dimension d
            d = float(self.dim)
            return beta_term + gamma_term - d * corr
            
        return cs_operator

    def special_conformal_transform(self, x: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """Apply special conformal transformation x' = (x + bx²)/(1 + 2bx + b²x²).
        
        Special conformal transformations preserve angles and can be used
        to study CFT correlation functions.
        
        Args:
            x: Position vector
            b: Transformation parameter vector
            
        Returns:
            Transformed position vector
        """
        # Convert to complex coordinates for conformal map
        if not x.is_complex():
            x = x.to(torch.complex64)
        if not b.is_complex():    
            b = b.to(torch.complex64)
            
        # Compute x² and b·x
        x_sq = torch.sum(x * x)
        b_dot_x = torch.sum(b * x)
        
        # Apply conformal transformation
        numerator = x + b * x_sq
        denominator = 1 + 2 * b_dot_x + torch.sum(b * b) * x_sq
        
        return numerator / denominator

    def transform_vector(self, v: torch.Tensor, x: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """Transform vector under conformal transformation.
        
        Vectors transform covariantly under conformal maps to preserve
        angles between vectors.
        
        Args:
            v: Vector to transform
            x: Position where vector is attached
            b: Conformal transformation parameter
            
        Returns:
            Transformed vector
        """
        # Convert to complex coordinates
        if not v.is_complex():
            v = v.to(torch.complex64)
        if not x.is_complex():
            x = x.to(torch.complex64)
        if not b.is_complex():
            b = b.to(torch.complex64)
            
        # Compute transformation Jacobian
        x_sq = torch.sum(x * x)
        b_dot_x = torch.sum(b * x)
        denom = 1 + 2 * b_dot_x + torch.sum(b * b) * x_sq
        
        # Transform vector
        jacobian = torch.eye(self.dim, dtype=torch.complex64) / denom
        jacobian = jacobian - 2 * torch.outer(b, x) / denom**2
        
        return jacobian @ v

    def holographic_lift(self, boundary: torch.Tensor, radial: torch.Tensor) -> torch.Tensor:
        """Lift boundary field to bulk using AdS/CFT correspondence.
        
        Implements holographic lifting of boundary data to bulk fields
        following the AdS/CFT dictionary. Uses the Fefferman-Graham
        expansion near the boundary.
        
        Args:
            boundary: Boundary field configuration
            radial: Radial coordinate points for bulk reconstruction
            
        Returns:
            Bulk field configuration
        """
        # Convert to complex if needed for holomorphic functions
        if not boundary.is_complex():
            boundary = boundary.to(torch.complex64)
            
        # Initialize bulk field
        bulk_shape = (len(radial), *boundary.shape)
        bulk = torch.zeros(bulk_shape, dtype=torch.complex64)
        
        # Compute bulk field using Fefferman-Graham expansion
        for i, z in enumerate(radial):
            # Leading term
            bulk[i] = boundary * z**(-self.dim)
            
            # Subleading corrections from conformal dimension
            for n in range(1, 4):  # Include first few corrections
                bulk[i] += (-1)**n * boundary * z**(-self.dim + 2*n) / (2*n)
                
            # Add quantum corrections using OPE
            if i > 0:  # Skip boundary point
                ope_corr = self.operator_product_expansion(
                    bulk[i-1],
                    boundary
                )
                bulk[i] += ope_corr * z**(-self.dim + 2)
                
        return bulk

    def entanglement_entropy(self, state: torch.Tensor, region: torch.Tensor) -> torch.Tensor:
        """Compute entanglement entropy using replica trick.
        
        Implements the replica trick to compute entanglement entropy:
        S = -Tr(ρ log ρ) = -lim_{n→1} ��_n Tr(ρ^n)
        
        Args:
            state: Quantum state tensor
            region: Binary mask defining the region
            
        Returns:
            Entanglement entropy value
        """
        # Convert state to density matrix if needed
        if state.dim() == 1:
            state = torch.outer(state, state.conj())
            
        # Ensure complex
        if not state.is_complex():
            state = state.to(torch.complex64)
            
        # Compute reduced density matrix
        # First reshape state into bipartite form using region mask
        region = region.bool()
        n_sites = state.shape[0]
        n_region = int(region.sum().item())  # Convert to Python int
        
        # Reshape into bipartite form using integer dimensions
        dim_a = 2**n_region
        dim_b = 2**(n_sites - n_region)
        state_bipartite = state.reshape(dim_a, dim_b)
        
        # Compute reduced density matrix by tracing out complement
        rho = state_bipartite @ state_bipartite.conj().t()
        
        # Compute eigenvalues
        eigenvals = torch.linalg.eigvals(rho)
        eigenvals = eigenvals.real  # Should be real for density matrix
        
        # Remove numerical noise
        eigenvals = eigenvals[eigenvals > 1e-10]
        
        # Compute von Neumann entropy
        entropy = -torch.sum(eigenvals * torch.log(eigenvals))
        
        return entropy


class ScaleSystem:
    """Complete scale system for multi-scale analysis."""

    def __init__(self, dim: int, num_scales: int = 4, coupling_dim: int = 4, dtype=torch.float32):
        self.connection = ScaleConnection(dim, num_scales, dtype=dtype)
        self.rg_flow = RenormalizationFlow(coupling_dim, dtype=dtype)
        self.anomaly = AnomalyDetector(dim, dtype=dtype)
        self.invariance = ScaleInvariance(dim, num_scales, dtype=dtype)
        self.cohomology = ScaleCohomology(dim, num_scales, dtype=dtype)

    def analyze_scales(
        self, states: List[torch.Tensor], scale_factors: List[float]
    ) -> Dict[str, Any]:
        """Analyze multi-scale structure."""
        results = {}

        # Analyze RG flow
        fixed_points, stability = self.rg_flow.find_fixed_points(states[0])
        results["fixed_points"] = fixed_points
        results["stability"] = stability

        # Find scale invariant structures
        invariants = self.invariance.find_invariant_structures(
            torch.stack(states)
        )
        results["invariants"] = invariants

        # Detect anomalies
        anomalies = []
        for state in states:
            anomalies.extend(self.anomaly.detect_anomalies(state))
        results["anomalies"] = anomalies

        # Compute cohomology
        cohomology = self.cohomology.analyze_cohomology(states, scale_factors)
        results["cohomology"] = cohomology

        return results
