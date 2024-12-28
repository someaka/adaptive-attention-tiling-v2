import torch
from typing import List, Tuple
from torch import nn
from src.core.crystal.scale_classes.complextanh import ComplexTanh
from src.core.crystal.scale_classes.memory_utils import memory_manager, memory_efficient_computation


class RenormalizationFlow:
    """Implementation of renormalization group flows."""

    def __init__(self, dim: int, max_iter: int = 100, dtype: torch.dtype = torch.float32):
        """Initialize RG flow."""
        self.dim = dim
        self.dtype = dtype
        self.max_iter = max_iter
        
        # Initialize networks with minimal dimensions
        hidden_dim = dim  # Use same dimension as input
        
        self.beta_network = nn.Sequential(
            nn.Linear(dim, hidden_dim, dtype=dtype),
            ComplexTanh(),
            nn.Linear(hidden_dim, dim, dtype=dtype)
        )
        
        self.metric_network = nn.Sequential(
            nn.Linear(dim, hidden_dim, dtype=dtype),
            ComplexTanh(),
            nn.Linear(hidden_dim, dim * dim, dtype=dtype)
        )
        
        # Initialize fixed point detector with minimal dimensions
        self.fp_detector = nn.Sequential(
            nn.Linear(dim, hidden_dim, dtype=dtype),
            ComplexTanh(),
            nn.Linear(hidden_dim, 1, dtype=dtype),
            ComplexTanh()
        )
        
        # Initialize memory manager for tensor operations
        self._memory_manager = memory_manager
        
        # Initialize default apply function
        self.apply = lambda x: x  # Identity function by default
        
        # Initialize flow lines storage
        self.flow_lines = []  # Store RG flow trajectories

    def compute_flow_lines(
        self, start_points: torch.Tensor, num_steps: int = 50
    ) -> List[torch.Tensor]:
        """Compute RG flow lines from starting points."""
        with memory_efficient_computation("compute_flow_lines"):
            flow_lines = []

        # Ensure start_points has correct shape
        if start_points.dim() == 1:
            start_points = start_points.unsqueeze(0)

        for point in start_points:
            line = [point.clone()]
            current = point.clone()

            for _ in range(num_steps):
                # Check if current point is near fixed point
                detector_output = self.fp_detector(current.reshape(1, -1))
                if detector_output.item() > 0.9:  # High confidence of fixed point
                    break
                    
                beta = self.beta_function(current)
                current = current - 0.1 * beta  # Use subtraction for gradient descent
                line.append(current.clone())
                del beta  # Clean up intermediate tensor

            flow_lines.append(torch.stack(line))

        self.flow_lines = flow_lines  # Store flow lines
        return flow_lines

    def scale_points(self) -> List[float]:
        """Get the scale points sampled in the flow."""
        # Return exponentially spaced points from flow lines
        if not self.flow_lines:
            return []
        num_points = self.flow_lines[0].shape[0]
        return [2.0 ** i for i in range(num_points)]

    def _ensure_dtype(self, tensor: torch.Tensor) -> torch.Tensor:
        """Ensure tensor has correct dtype."""
        if tensor.dtype != self.dtype:
            tensor = tensor.to(dtype=self.dtype)
        return tensor

    def beta_function(self, state: torch.Tensor) -> torch.Tensor:
        """Compute beta function at given state."""
        with memory_efficient_computation("beta_function"):
        # Ensure state has correct dtype and shape
            state = self._ensure_dtype(state)
        
        # Normalize state for network input
        state_norm = torch.norm(state)
        if state_norm > 0:
            normalized_state = state / state_norm
        else:
            normalized_state = state
            
        # Reshape for network input (batch_size x input_dim)
        if normalized_state.dim() == 1:
            normalized_state = normalized_state.reshape(1, -1)
        elif normalized_state.dim() == 2:
            pass  # Already in correct shape
        else:
            raise ValueError(f"Input tensor must be 1D or 2D, got shape {normalized_state.shape}")
            
        # Handle dimension mismatch by padding or truncating
        if normalized_state.shape[1] != self.dim:
            if normalized_state.shape[1] > self.dim:
                # Take first dim components
                normalized_state = normalized_state[:, :self.dim]
            else:
                # Pad with zeros
                padding = torch.zeros(normalized_state.shape[0], self.dim - normalized_state.shape[1], dtype=self.dtype)
                normalized_state = torch.cat([normalized_state, padding], dim=1)
        
        # Compute beta function
        beta = self.beta_network(normalized_state)
        
        # Reshape output to match input shape
        if state.dim() == 1:
            beta = beta.squeeze(0)
            # Pad or truncate output to match input size
            if beta.shape != state.shape:
                if len(beta) < len(state):
                    padding = torch.zeros(len(state) - len(beta), dtype=self.dtype)
                    beta = torch.cat([beta, padding])
                else:
                    beta = beta[:len(state)]
        
            # Clean up intermediate tensors
            del normalized_state
            if 'padding' in locals():
                del padding
        
        return beta
            
    def evolve(self, t: float) -> 'RenormalizationFlow':
        """Evolve RG flow to time t."""
        with memory_efficient_computation("evolve"):
            # Create new RG flow with same parameters
            flow = RenormalizationFlow(
                dim=self.dim,
                dtype=self.dtype
            )
            
            # Define evolution operator
            def evolve_operator(state: torch.Tensor) -> torch.Tensor:
                with memory_efficient_computation("evolve_operator"):
                    beta = self.beta_function(state)
                    evolved = state + t * beta
                    del beta
                    return evolved
            
            # Set the apply method
            flow.apply = evolve_operator
            return flow

    def find_fixed_points(
        self, initial_points: torch.Tensor
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        """Find fixed points and compute their stability."""
        # Ensure points have correct dtype
        initial_points = initial_points.to(dtype=self.dtype)
        
        fixed_points = []
        stability_matrices = []

        # Ensure initial points are properly shaped
        if initial_points.dim() > 2:
            initial_points = initial_points.reshape(-1, initial_points.shape[-1])
        elif initial_points.dim() == 1:
            initial_points = initial_points.unsqueeze(0)

        for point in initial_points:
            # Flow to fixed point
            current = point.clone()
            for _ in range(self.max_iter):
                beta = self.beta_function(current)
                if torch.norm(beta) < 1e-6:
                    break
                current -= 0.1 * beta

            # Check if point is fixed using mean of detector output
            detector_output = self.fp_detector(current)
            # For complex output, use magnitude
            if detector_output.is_complex():
                detector_value = torch.abs(detector_output).mean().item()
            else:
                detector_value = detector_output.mean().item()

            if detector_value > 0.5:
                fixed_points.append(current)

                # Compute stability matrix using autograd
                current.requires_grad_(True)
                beta = self.beta_function(current)
                
                # Initialize stability matrix with proper shape and dtype
                stability_matrix = torch.zeros(
                    (beta.numel(), current.numel()),
                    dtype=self.dtype
                )
                
                # Compute Jacobian for each component
                for i in range(beta.numel()):
                    if beta[i].is_complex():
                        # For complex components, compute gradient of real and imaginary parts separately
                        grad_real = torch.autograd.grad(
                            beta[i].real, current, 
                            grad_outputs=torch.ones_like(beta[i].real),
                            retain_graph=True
                        )[0].flatten()
                        grad_imag = torch.autograd.grad(
                            beta[i].imag, current,
                            grad_outputs=torch.ones_like(beta[i].imag),
                            retain_graph=True
                        )[0].flatten()
                        stability_matrix[i] = grad_real + 1j * grad_imag
                    else:
                        grad = torch.autograd.grad(
                            beta[i], current,
                            grad_outputs=torch.ones_like(beta[i]),
                            retain_graph=True
                        )[0].flatten()
                        stability_matrix[i] = grad

                stability_matrices.append(stability_matrix)

        return fixed_points, stability_matrices

