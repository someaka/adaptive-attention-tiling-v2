"""Pattern Dynamics Implementation for Neural Attention.

This module implements pattern dynamics for attention mechanisms:
- Reaction-diffusion systems for pattern formation
- Stability analysis of attention patterns
- Bifurcation detection and analysis
- Pattern control mechanisms
- Evolution optimization
"""

from dataclasses import dataclass
from typing import List, Optional, Tuple, Callable, Union
import numpy as np

import torch
from torch import nn


@dataclass
class ReactionDiffusionState:
    """State of the reaction-diffusion system."""

    activator: torch.Tensor  # Activator concentration
    inhibitor: torch.Tensor  # Inhibitor concentration
    gradients: Optional[torch.Tensor] = None  # Spatial gradients (optional)
    time: float = 0.0  # Current time (default 0.0)

    def sum(self, dim=None) -> torch.Tensor:
        """Compute sum of activator and inhibitor concentrations.
        
        Args:
            dim: Dimension to sum over. If None, sum over all dimensions.
        """
        if dim is None:
            return self.activator.sum() + self.inhibitor.sum()
        else:
            return self.activator.sum(dim=dim) + self.inhibitor.sum(dim=dim)


@dataclass
class StabilityInfo:
    """Information about pattern stability."""

    eigenvalues: torch.Tensor  # Stability eigenvalues
    eigenvectors: torch.Tensor  # Corresponding modes
    growth_rates: torch.Tensor  # Mode growth rates
    stable: bool  # Overall stability flag


@dataclass
class BifurcationPoint:
    """Represents a bifurcation in pattern dynamics."""

    parameter: float  # Bifurcation parameter value
    type: str  # Type of bifurcation
    normal_form: torch.Tensor  # Normal form coefficients
    eigenvalues: torch.Tensor  # Critical eigenvalues


@dataclass
class StabilityMetrics:
    """Metrics for pattern stability analysis."""
    linear_stability: torch.Tensor
    nonlinear_stability: torch.Tensor
    lyapunov_spectrum: torch.Tensor
    structural_stability: float


@dataclass 
class ControlSignal:
    """Control signal for pattern formation."""
    magnitude: torch.Tensor
    direction: torch.Tensor
    constraints: List[Callable]


@dataclass
class BifurcationDiagram:
    """Bifurcation diagram for pattern dynamics."""
    parameter_range: torch.Tensor
    bifurcation_points: List[BifurcationPoint]
    solution_branches: torch.Tensor
    stability_regions: torch.Tensor


class ReactionDiffusionSystem:
    """Implementation of reaction-diffusion dynamics."""

    def __init__(self, grid_size: int, dt: float = 0.01):
        """Initialize reaction-diffusion system.
        
        Args:
            grid_size: Size of square grid
            dt: Time step
        """
        self.grid_size = grid_size
        self.dt = dt
        
        # Initialize neural networks for reaction terms
        input_size = grid_size * grid_size * 2  # Flattened size for both species
        hidden_size = 64
        
        self.activator_network = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, grid_size * grid_size)
        )
        
        self.inhibitor_network = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, grid_size * grid_size)
        )

        # Diffusion operators (3x3 convolution with periodic padding)
        self.diffusion_activator = nn.Conv2d(1, 1, 3, padding=1)
        self.diffusion_inhibitor = nn.Conv2d(1, 1, 3, padding=1)
        
        # Initialize diffusion kernels with normalized weights for mass conservation
        with torch.no_grad():
            kernel = torch.tensor([[0.05, 0.2, 0.05],
                               [0.2, 0.0, 0.2],
                               [0.05, 0.2, 0.05]])
            kernel = kernel / kernel.sum()  # Normalize for mass conservation
            self.diffusion_activator.weight.data[0,0] = kernel
            self.diffusion_inhibitor.weight.data[0,0] = kernel

    def step(
        self, state: ReactionDiffusionState, dt: float = 0.01
    ) -> ReactionDiffusionState:
        """Perform one step of reaction-diffusion."""
        batch_size = state.activator.shape[0]
        grid_size = int(np.sqrt(state.activator.shape[1]))  # Corrected to use shape[1]
        
        # Reshape for 2D convolution
        activator = state.activator.reshape(batch_size, 1, grid_size, grid_size)
        inhibitor = state.inhibitor.reshape(batch_size, 1, grid_size, grid_size)
        
        # Apply diffusion
        diff_a = self.diffusion_activator(activator)
        diff_i = self.diffusion_inhibitor(inhibitor)
        
        # Flatten back
        diff_a = diff_a.reshape(batch_size, -1)
        diff_i = diff_i.reshape(batch_size, -1)
        
        # Apply reaction - reshape inputs to match network dimensions
        combined = torch.cat([diff_a, diff_i], dim=-1)  # [batch_size, grid_size * grid_size * 2]
        reaction_a = self.activator_network(combined)  # [batch_size, grid_size * grid_size]
        reaction_i = self.inhibitor_network(combined)  # [batch_size, grid_size * grid_size]
        
        # Update state
        new_activator = state.activator + dt * (reaction_a + diff_a)
        new_inhibitor = state.inhibitor + dt * (reaction_i + diff_i)
        
        # Ensure concentrations remain bounded
        new_activator = torch.clamp(new_activator, -10.0, 10.0)
        new_inhibitor = torch.clamp(new_inhibitor, -10.0, 10.0)
        
        return ReactionDiffusionState(
            activator=new_activator,
            inhibitor=new_inhibitor,
            gradients=state.gradients,
            time=state.time + dt
        )

    def apply_diffusion(
        self,
        state: torch.Tensor,
        diffusion_coefficient: float
    ) -> torch.Tensor:
        """Apply diffusion operator to state.
        
        Args:
            state: Input state tensor [batch, height, width]
            diffusion_coefficient: Diffusion coefficient
        
        Returns:
            Diffused state tensor [batch, height, width]
        """
        # Add channel dimension if needed
        if state.dim() == 3:
            state = state.unsqueeze(1)  # [batch, 1, height, width]
        
        # Create normalized diffusion kernel
        kernel = torch.tensor([[0.05, 0.2, 0.05],
                           [0.2, 0.0, 0.2],
                           [0.05, 0.2, 0.05]], device=state.device)
        kernel = kernel / kernel.sum().abs()  # Normalize for mass conservation
        
        # Apply periodic padding
        padded = torch.nn.functional.pad(state, (1, 1, 1, 1), mode='circular')
        
        # Apply convolution manually for better control
        output = torch.zeros_like(state)
        for i in range(3):
            for j in range(3):
                output += kernel[i,j] * padded[:, :, i:i+state.shape[2], j:j+state.shape[3]]
        
        # Scale by diffusion coefficient
        output = diffusion_coefficient * output
        
        # Remove channel dimension if input was 3D
        if state.dim() == 4 and state.shape[1] == 1:
            output = output.squeeze(1)
            
        return output

    def reaction_diffusion(
        self,
        state: Optional[Union[ReactionDiffusionState, torch.Tensor]] = None,
        diffusion_tensor: Optional[torch.Tensor] = None,
        reaction_term: Optional[Callable] = None,
        *,
        batch_size: Optional[Union[int, torch.Tensor]] = None,
        grid_size: Optional[int] = None,
    ) -> torch.Tensor:
        """Evolve reaction-diffusion system.
        
        Args:
            state: Optional initial state. If None, random state is generated
            diffusion_tensor: Optional 2x2 diffusion tensor
            reaction_term: Optional reaction term function
            batch_size: Batch size for random initialization
            grid_size: Grid size for random initialization
        
        Returns:
            Evolved state tensor [batch, channels, height, width]
        """
        # Handle default parameters
        if grid_size is None:
            grid_size = self.size
        if batch_size is None:
            batch_size = 1
        if diffusion_tensor is None:
            diffusion_tensor = torch.tensor([[0.1, 0.0], [0.0, 0.05]])
        if reaction_term is None:
            reaction_term = self.reaction_term
            
        # Initialize state if not provided
        if state is None:
            state = torch.randn(batch_size, 2, grid_size, grid_size) * 0.1
        elif isinstance(state, ReactionDiffusionState):
            state = torch.stack([state.activator, state.inhibitor], dim=1)
            
        # Ensure proper shape
        if state.dim() == 3:
            state = state.unsqueeze(1)
            
        # Apply reaction step first
        reaction = reaction_term(state)
        state = state + self.dt * reaction
        
        # Apply diffusion step for each species
        diffused = torch.zeros_like(state)
        for i in range(2):
            for j in range(2):
                if diffusion_tensor[i,j] != 0:
                    diffused[:,i] += self.apply_diffusion(state[:,j], diffusion_tensor[i,j])
                    
        state = state + self.dt * diffused
        
        # Ensure concentrations remain bounded
        state = torch.clamp(state, -self.max_concentration, self.max_concentration)
        
        return state

class StabilityAnalyzer:
    """Analysis of pattern stability."""

    def __init__(self, input_dim: int, num_modes: int = 8, hidden_dim: int = 64):
        self.input_dim = input_dim
        self.num_modes = num_modes
        self.hidden_dim = hidden_dim

        # Stability analysis networks
        self.stability_network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim * 2),
            nn.ReLU(),
            nn.Linear(hidden_dim * 2, num_modes * 2),
        )

        # Mode decomposition
        self.mode_analyzer = nn.Sequential(
            nn.Linear(input_dim, num_modes), nn.Softmax(dim=-1)
        )

    def analyze_stability(self, state: ReactionDiffusionState) -> StabilityInfo:
        """Analyze stability of current pattern."""
        # Combine state information 
        state_vector = torch.cat([state.activator.mean(0), state.inhibitor.mean(0)])

        # Compute stability matrix
        stability = self.stability_network(state_vector)
        stability = stability.reshape(self.num_modes, 2, 2)

        # Compute eigendecomposition for each mode
        eigenvalues = []
        eigenvectors = []
        for mode_matrix in stability:
            evals, evecs = torch.linalg.eigh(mode_matrix)
            eigenvalues.append(evals)
            eigenvectors.append(evecs)

        eigenvalues = torch.stack(eigenvalues)
        eigenvectors = torch.stack(eigenvectors)

        # Analyze growth rates for different modes
        growth_rates = self.mode_analyzer(state_vector)

        # Check stability criteria
        stable = torch.all(eigenvalues.real < 0)

        return StabilityInfo(
            eigenvalues=eigenvalues,
            eigenvectors=eigenvectors,
            growth_rates=growth_rates,
            stable=stable
        )


class BifurcationDetector:
    """Detection and analysis of bifurcations."""

    def __init__(self, input_dim: int, param_range: Tuple[float, float], hidden_dim: int = 64):
        self.input_dim = input_dim
        self.param_range = param_range
        self.hidden_dim = hidden_dim

        # Bifurcation detection network
        self.detector = nn.Sequential(
            nn.Linear(input_dim + 1, hidden_dim * 2),
            nn.ReLU(),
            nn.Linear(hidden_dim * 2, 4),  # [type, strength, re(λ), im(λ)]
        )

        # Normal form computation
        self.normal_form = nn.Sequential(
            nn.Linear(input_dim, hidden_dim * 2),
            nn.Tanh(),
            nn.Linear(hidden_dim * 2, 3),  # Up to cubic terms
        )

    def detect_bifurcations(
        self, states: List[ReactionDiffusionState], parameters: torch.Tensor
    ) -> List[BifurcationPoint]:
        """Detect bifurcations in parameter range."""
        bifurcations = []

        for state, param in zip(states, parameters):
            # Combine state and parameter
            state_vector = torch.cat(
                [state.activator.mean(0), state.inhibitor.mean(0), param.unsqueeze(0)]
            )

            # Analyze bifurcation
            detection = self.detector(state_vector)
            
            # Check bifurcation strength threshold
            if detection[1] > 0.5:
                # Get bifurcation type
                bif_type = self._classify_bifurcation(detection)
                
                # Compute normal form coefficients
                normal_form = self.normal_form(state_vector)
                
                # Create bifurcation point
                bifurcations.append(
                    BifurcationPoint(
                        parameter=param.item(),
                        type=bif_type,
                        normal_form=normal_form,
                        eigenvalues=detection[2:],
                    )
                )

        return bifurcations

    def _classify_bifurcation(self, detection: torch.Tensor) -> str:
        """Classify type of bifurcation based on eigenvalues."""
        # Get real and imaginary parts
        re_lambda = detection[2]
        im_lambda = detection[3]
        
        # Classify based on eigenvalue structure
        if torch.abs(im_lambda) > 0.1:
            return "hopf"  # Complex conjugate pair crossing
        elif re_lambda > 0:
            if detection[0].argmax() == 2:
                return "pitchfork"  # Symmetry-breaking
            else:
                return "saddle-node"  # Tangent bifurcation
        else:
            return "transcritical"  # Exchange of stability


class PatternController:
    """Control of pattern formation and evolution."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 64,
        output_dim: int = 64,
        control_dim: int = 4
    ):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.control_dim = control_dim

        # Control policy network
        self.network = nn.Sequential(
            nn.Linear(input_dim * 2, hidden_dim * 2),  # *2 for both current and target states
            nn.ReLU(),
            nn.Linear(hidden_dim * 2, input_dim),  # Output control signal matching input dimension
        )

        # Value estimation
        self.value = nn.Sequential(
            nn.Linear(input_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def compute_control(self, state: ReactionDiffusionState, target: torch.Tensor) -> torch.Tensor:
        """Compute control signal to drive system toward target state.
        
        Args:
            state: Current state (ReactionDiffusionState)
            target: Target state tensor
        
        Returns:
            Control signal tensor
        """
        # Flatten and concatenate activator and inhibitor
        batch_size = state.activator.shape[0]
        flattened_state = torch.cat([
            state.activator.reshape(batch_size, -1),
            state.inhibitor.reshape(batch_size, -1)
        ], dim=1)
        
        # Concatenate with target state
        combined = torch.cat([flattened_state, target.reshape(batch_size, -1)], dim=1)
        
        # Compute control signal
        control = self.network(combined)
        
        # Reshape to match input state dimensions
        grid_size = int(np.sqrt(control.shape[1] // 2))
        control = control.reshape(batch_size, 2, grid_size, grid_size)
        
        return control


class PatternDynamics:
    """Complete pattern dynamics system."""

    def __init__(
        self, 
        dim: int,
        size: int,
        dt: float = 0.01,
        boundary: str = "periodic",
        hidden_dim: int = 64,
        num_modes: int = 8,
        param_range: Tuple[float, float] = (0.0, 1.0),
    ):
        self.dim = dim
        self.size = size
        self.dt = dt
        self.boundary = boundary
        self.max_concentration = 10.0
        
        # Initialize subsystems
        self.reaction_diffusion_system = ReactionDiffusionSystem(
            grid_size=size,
            dt=dt
        )
        self.stability = StabilityAnalyzer(
            input_dim=dim * size * 2,
            num_modes=num_modes,
            hidden_dim=hidden_dim
        )
        self.bifurcation = BifurcationDetector(
            input_dim=dim * size * 2,
            param_range=param_range,
            hidden_dim=hidden_dim
        )
        self.controller = PatternController(
            input_dim=dim * size * 2,
            hidden_dim=hidden_dim,
            output_dim=dim * size * 2
        )

    def apply_diffusion(
        self,
        state: torch.Tensor,
        dt: float
    ) -> torch.Tensor:
        """Apply diffusion operator.
        
        Args:
            state: Input tensor [batch, channels, height, width]
            dt: Time step
            
        Returns:
            Diffused tensor [batch, channels, height, width]
        """
        # Create 2D diffusion kernel that sums to 0 for mass conservation
        kernel = torch.tensor([[0.05, 0.2, 0.05],
                             [0.2, -1.0, 0.2],
                             [0.05, 0.2, 0.05]], dtype=state.dtype)
        
        # Reshape kernel for conv2d [out_channels, in_channels, height, width]
        kernel = kernel.view(1, 1, 3, 3).to(state.device)
        
        # Ensure input has 4 dimensions [batch, channels, height, width]
        if state.dim() == 3:  # [batch, height, width]
            state = state.unsqueeze(1)  # Add channel dimension
            
        # Apply periodic padding
        padded = torch.nn.functional.pad(state, (1, 1, 1, 1), mode='circular')
        
        # Process each channel separately
        channels = []
        for i in range(state.shape[1]):
            channel = padded[:, i:i+1]  # Keep channel dimension
            diffused = torch.nn.functional.conv2d(
                channel,
                kernel,
                padding=0
            )
            channels.append(diffused)
            
        # Combine channels
        diffused = torch.cat(channels, dim=1)
        
        return state + dt * diffused

    def reaction_term(self, state: torch.Tensor) -> torch.Tensor:
        """Default reaction term for pattern formation.
        
        This implements a simple activator-inhibitor system with:
            - Autocatalytic production of activator
            - Linear degradation of both species
            - Nonlinear inhibition
        
        Args:
            state: Input tensor [batch, channels, height, width]
            
        Returns:
            Reaction term tensor [batch, channels, height, width]
        """
        # Unpack activator and inhibitor
        u = state[:,0:1]  # Activator
        v = state[:,1:2]  # Inhibitor
        
        # Parameters for Turing pattern formation
        a = 2.0  # Production rate
        b = 1.0  # Degradation rate
        c = 3.0  # Cross-inhibition
        d = 5.0  # Saturation constant
        
        # Reaction terms (based on Gierer-Meinhardt model)
        du = a * u * u / (1 + d * u * u) - b * u + c  # Activator dynamics
        dv = a * u * u - b * v  # Inhibitor dynamics
        
        # Combine and return
        return torch.cat([du, dv], dim=1)

    def apply_reaction(
        self,
        state: torch.Tensor,
        reaction_term: Optional[Callable] = None
    ) -> torch.Tensor:
        """Apply reaction term to state.
        
        Args:
            state: Input tensor [batch, channels, height, width]
            reaction_term: Optional reaction term function
            
        Returns:
            Reacted tensor [batch, channels, height, width]
        """
        if reaction_term is None:
            reaction_term = self.reaction_term
            
        # Apply reaction term
        reaction = reaction_term(state)
        
        # Add small noise to break symmetry and promote pattern formation
        noise = 0.01 * torch.randn_like(state)
        
        return state + self.dt * (reaction + noise)

    def reaction_diffusion(
        self,
        state: Optional[Union[ReactionDiffusionState, torch.Tensor]] = None,
        diffusion_tensor: Optional[torch.Tensor] = None,
        reaction_term: Optional[Callable] = None,
        *,
        batch_size: Optional[Union[int, torch.Tensor]] = None,
        grid_size: Optional[int] = None,
    ) -> torch.Tensor:
        """Evolve reaction-diffusion system.
        
        Args:
            state: Optional initial state. If None, random state is generated
            diffusion_tensor: Optional 2x2 diffusion tensor
            reaction_term: Optional reaction term function
            batch_size: Batch size for random initialization
            grid_size: Grid size for random initialization
        
        Returns:
            Evolved state tensor [batch, channels, height, width]
        """
        # Handle default parameters
        if grid_size is None:
            grid_size = self.size
        if batch_size is None:
            batch_size = 1
        if diffusion_tensor is None:
            diffusion_tensor = torch.tensor([[0.1, 0.0], [0.0, 0.05]])
        if reaction_term is None:
            reaction_term = self.reaction_term
            
        # Initialize state if not provided
        if state is None:
            state = 0.5 + 0.1 * torch.randn(batch_size, 2, grid_size, grid_size)
        elif isinstance(state, ReactionDiffusionState):
            state = torch.stack([state.activator, state.inhibitor], dim=1)
            
        # Ensure proper shape
        if state.dim() == 3:
            state = state.unsqueeze(1)
            
        # Apply reaction step first
        state = self.apply_reaction(state, reaction_term)
        
        # Apply diffusion step for each species
        diffused = torch.zeros_like(state)
        for i in range(2):
            for j in range(2):
                if diffusion_tensor[i,j] != 0:
                    diffused[:,i] += self.apply_diffusion(state[:,j], diffusion_tensor[i,j])
                    
        state = state + diffused
        
        # Ensure concentrations remain bounded and positive
        state = torch.clamp(state, 0.0, self.max_concentration)
        
        return state

    def apply_reaction_diffusion(
        self, 
        state: torch.Tensor,
        diffusion_tensor: torch.Tensor,
        reaction_term: Optional[Callable] = None,
        grid_size: int = 32,
        batch_size: int = 1,
    ) -> torch.Tensor:
        """Apply reaction-diffusion dynamics to a state.
        
        Args:
            state: Input state tensor [batch, channels, height, width]
            diffusion_tensor: Diffusion coefficients
            reaction_term: Optional reaction term function
            grid_size: Size of spatial grid
            batch_size: Batch size
        
        Returns:
            Evolved state tensor [batch, channels, height, width]
        """
        # Handle scalar diffusion coefficient
        if isinstance(diffusion_tensor, (int, float)):
            diffusion_tensor = torch.eye(2) * diffusion_tensor
    
        # Ensure proper tensor dimensions
        if state.dim() == 3:  # [batch, height, width]
            state = state.unsqueeze(1)  # Add channel dimension [batch, 1, height, width]
    
        # Apply reaction-diffusion step
        return self.reaction_diffusion(
            state,
            diffusion_tensor,
            reaction_term,
            grid_size=grid_size,
            batch_size=batch_size
        )

    def evolve_pattern(
        self,
        state: torch.Tensor,
        diffusion_tensor: torch.Tensor,
        reaction_term: Optional[Callable] = None,
        steps: int = 100
    ) -> torch.Tensor:
        """Evolve pattern over time.
        
        Args:
            state: Initial state [batch_size, channels, height, width]
            diffusion_tensor: Diffusion coefficients [channels, channels]
            reaction_term: Optional reaction term function
            steps: Number of evolution steps
            
        Returns:
            Evolution tensor [steps, batch, channels, height, width]
        """
        # Ensure state has correct dimensions
        if state.dim() == 3:  # [batch, height, width]
            state = state.unsqueeze(1)  # [batch, channels, height, width]
        elif state.dim() == 2:  # [height, width]
            state = state.unsqueeze(0).unsqueeze(0)  # [1, channels, height, width]
        
        # Handle scalar diffusion coefficient
        if isinstance(diffusion_tensor, (int, float)):
            diffusion_tensor = torch.eye(2) * diffusion_tensor
    
        # Get dimensions
        batch_size = state.shape[0]
        channels = 2  # Fixed for reaction-diffusion
        grid_size = state.shape[-1]
    
        # Initialize evolution tensor
        evolution = torch.zeros(steps, batch_size, channels, grid_size, grid_size)
        evolution[0] = state
    
        # Evolve system
        current = state
        for t in range(1, steps):
            current = self.apply_reaction_diffusion(
                current,
                diffusion_tensor,
                reaction_term,
                grid_size=grid_size,
                batch_size=batch_size
            )
            evolution[t] = current
    
        return evolution

    def detect_pattern_formation(self, time_evolution: torch.Tensor) -> bool:
        """Detect if stable patterns have formed.
        
        Args:
            time_evolution: Evolution tensor [steps, batch, channels, height, width]
            
        Returns:
            True if stable patterns detected
        """
        if not isinstance(time_evolution, torch.Tensor):
            raise ValueError("time_evolution must be a tensor")
            
        if time_evolution.dim() != 5:
            raise ValueError(f"time_evolution must have 5 dimensions, got {time_evolution.dim()}")
            
        # Compute spatial variation over time
        spatial_var = time_evolution.var(dim=[3, 4])  # [steps, batch, channels]
        
        # Compute temporal variation of spatial patterns
        temporal_var = torch.diff(spatial_var, dim=0)  # [steps-1, batch, channels]
        
        # Patterns are stable if temporal variation decreases
        stability = torch.abs(temporal_var[-10:]).mean() < 0.01
        
        # And if spatial variation is significant
        pattern_formed = spatial_var[-1].mean() > 0.1
        
        return bool(stability and pattern_formed)

    def stability_analysis(
        self,
        fixed_point: Union[ReactionDiffusionState, torch.Tensor],
        batch_size: int,
        grid_size: int,
        epsilon: float = 1e-3
    ) -> torch.Tensor:
        """Analyze stability around fixed point.
        
        Args:
            fixed_point: Fixed point state
            batch_size: Batch size
            grid_size: Size of square grid
            epsilon: Perturbation size
        
        Returns:
            Stability eigenvalues
        """
        # Convert batch_size to int if needed
        if isinstance(batch_size, torch.Tensor):
            if batch_size.numel() == 1:
                batch_size = batch_size.item()
            else:
                batch_size = batch_size.shape[0]
                
        # Ensure fixed point has correct shape
        if isinstance(fixed_point, torch.Tensor):
            if fixed_point.dim() == 4:  # [batch, channels, height, width]
                fixed_point = ReactionDiffusionState(
                    activator=fixed_point[:, 0],
                    inhibitor=fixed_point[:, 1]
                )
            elif fixed_point.dim() == 3:  # [batch, height, width]
                fixed_point = ReactionDiffusionState(
                    activator=fixed_point,
                    inhibitor=torch.zeros_like(fixed_point)
                )
        
        # Compute Jacobian using finite differences
        perturbed = []
        for i in range(2):  # Activator and inhibitor
            for j in range(grid_size * grid_size):
                # Create perturbation
                perturbation = torch.zeros(batch_size, grid_size, grid_size)
                perturbation.view(batch_size, -1)[:, j] = epsilon
                
                # Perturb state
                if i == 0:
                    perturbed_state = ReactionDiffusionState(
                        activator=fixed_point.activator + perturbation,
                        inhibitor=fixed_point.inhibitor.clone()
                    )
                else:
                    perturbed_state = ReactionDiffusionState(
                        activator=fixed_point.activator.clone(),
                        inhibitor=fixed_point.inhibitor + perturbation
                    )
                
                # Compute perturbed evolution
                evolved = self.apply_reaction_diffusion(
                    perturbed_state,
                    torch.eye(2),  # Unit diffusion tensor for stability analysis
                    None,  # No reaction term for linear stability
                    grid_size=grid_size,
                    batch_size=batch_size
                )
                
                perturbed.append(evolved)
        
        # Compute eigenvalues
        jacobian = torch.stack(perturbed).reshape(2 * grid_size * grid_size, -1)
        eigenvalues = torch.linalg.eigvals(jacobian)
        
        return eigenvalues

    def compute_lyapunov_spectrum(self, pattern: torch.Tensor) -> torch.Tensor:
        """Compute Lyapunov spectrum."""
        # Use stability eigenvalues as approximation
        state = ReactionDiffusionState(
            activator=pattern[:, 0],
            inhibitor=pattern[:, 1],
            gradients=torch.zeros_like(pattern),
            time=0.0
        )
        info = self.stability.analyze_stability(state)
        return info.eigenvalues.real

    def test_structural_stability(
        self,
        pattern: torch.Tensor,
        perturbed_reaction: Callable
    ) -> float:
        """Test structural stability under perturbation."""
        # Compare original and perturbed trajectories
        original = self.evolve_pattern(pattern, torch.eye(2), lambda x: x)
        perturbed = self.evolve_pattern(pattern, torch.eye(2), perturbed_reaction)
        
        difference = torch.norm(original - perturbed)
        return 1.0 / (1.0 + difference)

    def bifurcation_analysis(
        self,
        pattern: torch.Tensor,
        parameterized_reaction: Callable,
        parameter_range: torch.Tensor
    ) -> BifurcationDiagram:
        """Analyze bifurcations."""
        # Evolve system for each parameter
        states = []
        for param in parameter_range:
            evolved = self.evolve_pattern(
                pattern,
                torch.eye(2),
                lambda x: parameterized_reaction(x, param)
            )
            states.append(evolved[-1])

        # Detect bifurcations
        bifurcations = self.bifurcation.detect_bifurcations(states, parameter_range)

        return BifurcationDiagram(
            parameter_range=parameter_range,
            bifurcation_points=bifurcations,
            solution_branches=torch.stack(states),
            stability_regions=torch.ones_like(parameter_range)
        )

    def compute_normal_form(self, bifurcation: BifurcationPoint) -> torch.Tensor:
        """Compute normal form coefficients."""
        return bifurcation.normal_form

    def pattern_control(
        self,
        current: Union[ReactionDiffusionState, torch.Tensor],
        target: Union[ReactionDiffusionState, torch.Tensor],
        constraints: Optional[List[Callable]] = None
    ) -> torch.Tensor:
        """Compute control signal to drive system toward target pattern.
        
        Args:
            current: Current state
            target: Target state
            constraints: Optional list of constraint functions
        
        Returns:
            Control signal tensor
        """
        # Convert tensor inputs to ReactionDiffusionState if needed
        if isinstance(current, torch.Tensor):
            if current.dim() == 4:  # [batch, channels, height, width]
                current = ReactionDiffusionState(
                    activator=current[:, 0],
                    inhibitor=current[:, 1]
                )
            elif current.dim() == 3:  # [batch, height, width]
                current = ReactionDiffusionState(
                    activator=current,
                    inhibitor=torch.zeros_like(current)
                )
    
        if isinstance(target, torch.Tensor):
            if target.dim() == 4:  # [batch, channels, height, width]
                target = ReactionDiffusionState(
                    activator=target[:, 0],
                    inhibitor=target[:, 1]
                )
            elif target.dim() == 3:  # [batch, height, width]
                target = ReactionDiffusionState(
                    activator=target,
                    inhibitor=torch.zeros_like(target)
                )
    
        # Flatten target state for controller
        batch_size = current.activator.shape[0]
        target_flat = torch.cat([
            target.activator.reshape(batch_size, -1),
            target.inhibitor.reshape(batch_size, -1)
        ], dim=1)
    
        # Compute control signal
        control = self.controller.compute_control(current, target_flat)
    
        # Apply constraints if provided
        if constraints:
            for constraint in constraints:
                control = constraint(control)
            
        return control

    def apply_control(
        self,
        pattern: torch.Tensor,
        control: ControlSignal
    ) -> torch.Tensor:
        """Apply control signal to pattern."""
        # Apply control while respecting constraints
        controlled = pattern + control.magnitude * control.direction.reshape(pattern.shape)
        
        # Project onto constraint manifold
        for constraint in control.constraints:
            violation = constraint(controlled)
            controlled = controlled - violation * control.direction.reshape(pattern.shape)
            
        return controlled

    def _compute_nonlinear_stability(
        self,
        pattern: torch.Tensor,
        perturbation: torch.Tensor
    ) -> torch.Tensor:
        """Compute nonlinear stability metric."""
        # Evolve both original and perturbed patterns
        original = self.evolve_pattern(pattern, torch.eye(2), lambda x: x)
        perturbed = self.evolve_pattern(pattern + perturbation, torch.eye(2), lambda x: x)
        
        # Compute maximal deviation
        difference = torch.norm(original - perturbed)
        return -torch.log(difference.max())

    def find_reaction_fixed_points(
        self,
        state: torch.Tensor
    ) -> List[torch.Tensor]:
        """Find fixed points of the reaction term."""
        # Sample points in phase space
        u_range = torch.linspace(-2, 2, 100)
        v_range = torch.linspace(-2, 2, 100)
        U, V = torch.meshgrid(u_range, v_range)
        points = torch.stack([U, V], dim=0)
        
        # Compute reaction term
        du = points[0]**2 * points[1] - points[0]
        dv = points[0]**2 - points[1]
        reaction = torch.stack([du, dv], dim=0)
        
        # Find zeros
        zeros = torch.where(torch.norm(reaction, dim=0) < 0.1)
        fixed_points = [points[:, i, j] for i, j in zip(*zeros)]
        
        return fixed_points

    def evolve_spatiotemporal(
        self,
        initial: torch.Tensor,
        coupling: Callable,
        t_span: List[float],
        steps: int = 100
    ) -> List[torch.Tensor]:
        """Evolve pattern with space-time coupling."""
        evolution = [initial]
        current = initial
        
        # Time points
        times = torch.linspace(t_span[0], t_span[1], steps)
        dt = times[1] - times[0]
        
        for t in times:
            # Apply diffusion
            diffused = self.apply_diffusion(current, dt)
            
            # Apply reaction with coupling
            coupled = diffused + coupling(diffused, t)
            reacted = self.apply_reaction(coupled)
            
            # Update state
            current = reacted
            evolution.append(current)
            
        return evolution

    def find_spatiotemporal_symmetries(
        self,
        evolution: Union[torch.Tensor, List[torch.Tensor]]
    ) -> torch.Tensor:
        """Find symmetries in space-time evolution.
        
        Args:
            evolution: Evolution tensor [steps, batch, channels, height, width]
            or list of tensors
        
        Returns:
            Symmetry tensor [batch_size, num_symmetries]
        """
        # Convert list to tensor if needed
        if isinstance(evolution, list):
            evolution = torch.stack(evolution)
    
        # Ensure correct dimensions
        if evolution.dim() == 4:  # [steps, batch, height, width]
            evolution = evolution.unsqueeze(2)  # Add channel dimension
    
        # Get dimensions
        steps, batch_size = evolution.shape[:2]
    
        # Compute temporal correlations
        temporal_corr = torch.zeros(batch_size, steps)
        for t in range(steps):
            temporal_corr[:, t] = torch.mean(
                (evolution[t] * evolution[0]).reshape(batch_size, -1),
                dim=1
            )
    
        # Find peaks in correlations
        peaks = torch.zeros(batch_size, steps, dtype=bool)
        for b in range(batch_size):
            for t in range(1, steps-1):
                if (temporal_corr[b,t] > temporal_corr[b,t-1] and 
                    temporal_corr[b,t] > temporal_corr[b,t+1]):
                    peaks[b,t] = True
    
        # Count symmetries
        num_symmetries = peaks.sum(dim=1)
    
        return num_symmetries.float()

    def classify_pattern(
        self,
        evolution: List[torch.Tensor]
    ) -> str:
        """Classify spatiotemporal pattern type.
        
        Args:
            evolution: List of state tensors over time
            
        Returns:
            Pattern type as string: "stationary", "periodic", "quasi-periodic", or "chaotic"
        """
        # Convert list to tensor [time, batch, channels, height, width]
        trajectory = torch.stack(evolution)
        
        # Compute time differences and norms
        diff = torch.diff(trajectory, dim=0)
        norm = torch.norm(diff.reshape(len(evolution)-1, -1), dim=1)
        
        # Compute mean and std of differences
        mean_diff = torch.mean(norm)
        std_diff = torch.std(norm)
        
        # Compute temporal autocorrelation
        n = len(evolution)
        corr = torch.zeros(n//2)
        for t in range(n//2):
            corr[t] = torch.mean(
                (trajectory[t:] * trajectory[:-t if t > 0 else None]).reshape(n-t, -1),
                dim=1
            ).mean()
        
        # Normalize correlation
        corr = corr / corr[0]
        
        # Find peaks in correlation
        peaks = []
        for t in range(1, len(corr)-1):
            if corr[t] > corr[t-1] and corr[t] > corr[t+1]:
                peaks.append(t)
                
        # Classify based on peaks and variability
        if mean_diff < 0.01:  # Very small changes
            return "stationary"
        elif len(peaks) == 0:  # No periodicity
            if std_diff / mean_diff > 0.5:  # High variability
                return "chaotic"
            else:
                return "stationary"
        elif len(peaks) == 1:  # Single period
            return "periodic"
        else:  # Multiple periods
            return "quasi-periodic"

    def estimate_embedding_dimension(
        self,
        evolution: List[torch.Tensor]
    ) -> int:
        """Estimate embedding dimension of attractor."""
        # Convert to tensor and flatten spatial dimensions
        trajectory = torch.stack(evolution)
        flat = trajectory.reshape(len(evolution), -1)
        
        # Perform SVD
        _, s, _ = torch.linalg.svd(flat)
        
        # Count significant singular values
        return torch.sum(s > 0.1 * s[0]).item()

    def control_energy(
        self,
        control: ControlSignal
    ) -> float:
        """Compute control energy."""
        return torch.norm(control.magnitude * control.direction).item()

    def test_reachability(
        self,
        current: torch.Tensor,
        target: torch.Tensor
    ) -> bool:
        """Test if target is reachable from current state."""
        # Create state
        state = ReactionDiffusionState(
            activator=current[:, 0],
            inhibitor=current[:, 1],
            gradients=torch.zeros_like(current),
            time=0.0
        )
        
        # Get control
        control = self.controller.compute_control(state, target.reshape(-1))
        
        # Check if control can reach target
        controlled = current + control.reshape(current.shape)
        return torch.allclose(controlled, target, rtol=0.1)

    def find_homogeneous_state(
        self,
        initial_guess: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Find homogeneous steady state using Newton's method."""
        if initial_guess is None:
            initial_guess = torch.zeros(self.dim)
            
        def reaction_term(x):
            # Default reaction term (activator-inhibitor)
            u, v = x[0], x[1]
            du = u**2 * v - u
            dv = u**2 - v
            return torch.stack([du, dv])
            
        # Newton iteration
        x = initial_guess
        for _ in range(100):
            fx = reaction_term(x)
            if torch.norm(fx) < 1e-6:
                break
                
            # Compute Jacobian
            h = 1e-6
            jac = torch.zeros(self.dim, self.dim)
            for i in range(self.dim):
                x_plus = x.clone()
                x_plus[i] += h
                jac[:, i] = (reaction_term(x_plus) - fx) / h
                
            # Update
            dx = torch.linalg.solve(jac, -fx)
            x = x + dx
            
        return x.reshape(1, self.dim, 1, 1).repeat(1, 1, self.size, self.size)

    def generate_target_pattern(
        self,
        batch_size: int,
        grid_size: int,
        num_species: int = 2
    ) -> torch.Tensor:
        """Generate a target pattern for testing control systems.
        
        Args:
            batch_size: Number of patterns to generate
            grid_size: Size of each spatial dimension
            num_species: Number of chemical species
            
        Returns:
            Target pattern tensor of shape [batch_size, num_species, grid_size, grid_size]
        """
        # Generate random pattern
        pattern = torch.randn(batch_size, num_species, grid_size, grid_size)
        
        # Apply smoothing to make it more realistic
        kernel = torch.ones(1, 1, 3, 3) / 9.0
        smoothed = []
        for i in range(num_species):
            species = pattern[:, i:i+1]  # Keep batch and channel dims
            smoothed.append(nn.functional.conv2d(species, kernel, padding=1))
        pattern = torch.cat(smoothed, dim=1)
        
        # Ensure concentrations are bounded
        pattern = torch.clamp(pattern, -self.max_concentration, self.max_concentration)
        
        return pattern

    def spatiotemporal_evolution(
        self,
        initial_state: Union[ReactionDiffusionState, torch.Tensor],
        steps: int = 100,
        *,
        diffusion_tensor: Optional[torch.Tensor] = None,
        reaction_term: Optional[Callable] = None,
    ) -> torch.Tensor:
        """Compute spatiotemporal evolution of pattern.
        
        Args:
            initial_state: Initial state
            steps: Number of time steps
            diffusion_tensor: Optional diffusion tensor
            reaction_term: Optional reaction term
        
        Returns:
            Evolution tensor [steps, batch, channels, height, width]
        """
        # Convert initial state to tensor if needed
        if isinstance(initial_state, ReactionDiffusionState):
            state = torch.stack([initial_state.activator, initial_state.inhibitor], dim=1)
        else:
            state = initial_state
        
        # Default diffusion tensor if not provided
        if diffusion_tensor is None:
            diffusion_tensor = torch.eye(2)
        
        # Evolve system
        evolution = []
        current = state
        
        for _ in range(steps):
            evolved = self.apply_reaction_diffusion(
                current,
                diffusion_tensor,
                reaction_term,
                grid_size=state.shape[-1],
                batch_size=state.shape[0]
            )
            evolution.append(evolved)
            current = evolved
        
        # Stack along time dimension
        return torch.stack(evolution)
