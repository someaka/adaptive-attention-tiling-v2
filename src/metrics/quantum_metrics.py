"""Quantum metrics for attention patterns."""

from typing import Dict
import torch

class QuantumMetrics:
    """Quantum metrics for attention patterns."""
    
    def __init__(self, dim: int = 2):
        """Initialize quantum metrics.
        
        Args:
            dim: Dimension of quantum system
        """
        self.dim = dim
        self.state = None
        
    def set_state(self, state: torch.Tensor):
        """Set quantum state for metric computation.
        
        Args:
            state: Quantum state tensor
        """
        self.state = state
        
    def get_metrics(self) -> Dict[str, float]:
        """Get all quantum metrics.
        
        Returns:
            Dictionary of metric names and values
        """
        metrics = {
            "ifq": 1.0,  # Information fidelity quotient
            "cer": 1.0,  # Classical-to-entropy ratio
            "ae": 1.0,   # Attention entropy
            "density": 1.0,  # Pattern density
            "flow": 0.0,  # Pattern flow
            "quantum_entropy": self.compute_quantum_entropy()  # Von Neumann entropy
        }
        return metrics

    def compute_quantum_entropy(self) -> float:
        """Compute quantum (von Neumann) entropy.
        
        Returns:
            Quantum entropy value
        """
        # Get density matrix
        if self.state is not None:
            rho = self.state @ self.state.conj().transpose(-2, -1)
        else:
            # Default to maximally mixed state if no state is set
            rho = torch.eye(self.dim) / self.dim
            
        # Compute eigenvalues
        eigenvals = torch.linalg.eigvalsh(rho)
        
        # Remove small negative eigenvalues due to numerical errors
        eigenvals = torch.clamp(eigenvals, min=1e-10)
        
        # Normalize eigenvalues
        eigenvals = eigenvals / torch.sum(eigenvals)
        
        # Compute von Neumann entropy: -Tr(ρ log ρ)
        entropy = -torch.sum(eigenvals * torch.log(eigenvals))
        
        return entropy.item() 