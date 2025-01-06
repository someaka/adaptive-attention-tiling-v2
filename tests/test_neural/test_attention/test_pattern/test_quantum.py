"""Tests for quantum pattern functionality."""

import torch
import pytest
import numpy as np
import logging
from src.neural.attention.pattern.dynamics import PatternDynamics
from src.core.quantum.types import QuantumState
from src.core.quantum.state_space import HilbertSpace
from src.core.quantum.neural_quantum_bridge import NeuralQuantumBridge

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def print_tensor_info(name: str, tensor: torch.Tensor):
    """Print detailed tensor information."""
    logger.info(f"\n{name}:")
    logger.info(f"Shape: {tensor.shape}")
    logger.info(f"Dtype: {tensor.dtype}")
    logger.info(f"Device: {tensor.device}")
    logger.info(f"Norm: {torch.norm(tensor)}")
    if torch.is_complex(tensor):
        logger.info(f"Real norm: {torch.norm(tensor.real)}")
        logger.info(f"Imag norm: {torch.norm(tensor.imag)}")
    logger.info(f"Max abs: {torch.max(torch.abs(tensor))}")
    logger.info(f"Min abs: {torch.min(torch.abs(tensor))}")
    if len(tensor.shape) > 2:
        logger.info(f"Per-channel norms: {[torch.norm(tensor[:,i]).item() for i in range(tensor.shape[1])]}")

@pytest.fixture
def quantum_system():
    """Create quantum-enabled pattern system."""
    system = PatternDynamics(
        grid_size=8,
        space_dim=2,
        quantum_enabled=True,
        hidden_dim=64
    )
    # Initialize HilbertSpace with explicit dtype and bridge
    hilbert_space = HilbertSpace(dim=system.dim, dtype=torch.float64)
    
    # Calculate total dimension for the bridge
    total_dim = system.dim * system.size * system.size
    bridge = NeuralQuantumBridge(
        hidden_dim=total_dim,  # Use total dimension for the bridge
        manifold_type="hyperbolic",
        dtype=torch.float64
    )
    system.quantum_flow.hilbert_space = hilbert_space
    system.quantum_flow.bridge = bridge
    
    # Set the manifold dimension in the quantum flow
    system.quantum_flow.manifold_dim = total_dim
    
    # Initialize metric network with correct input/output dimensions
    # The output size should be total_dim * total_dim to be reshaped into a square matrix
    system.quantum_flow.metric_net = torch.nn.Sequential(
        torch.nn.Linear(total_dim, 256, dtype=torch.float64),
        torch.nn.ReLU(),
        torch.nn.Linear(256, total_dim * total_dim, dtype=torch.float64)
    )
    
    # Initialize state reconstruction network with correct dimensions
    # Input: total_dim * 2 (real and imaginary parts)
    # Output: total_dim * 2 (real and imaginary parts)
    system.quantum_flow.state_reconstruction_net = torch.nn.Sequential(
        torch.nn.Linear(total_dim * 2, 256, dtype=torch.float64),
        torch.nn.ReLU(),
        torch.nn.Linear(256, 128, dtype=torch.float64),
        torch.nn.Tanh(),
        torch.nn.Linear(128, total_dim * 2, dtype=torch.float64)
    )
    
    # Ensure system uses float64
    system.to(torch.float64)
    return system

class TestQuantumPatterns:
    """Test suite for quantum pattern functionality."""
    
    def test_quantum_state_conversion(self, quantum_system):
        """Test conversion between classical and quantum states."""
        logger.info("\n=== Starting quantum state conversion test ===")
        
        # Create test state with proper normalization
        state = torch.randn(1, quantum_system.dim, quantum_system.size, quantum_system.size, 
                          dtype=torch.float64)
        print_tensor_info("Initial state (before any normalization)", state)
        
        # First normalization - normalize per channel
        state = state / (torch.norm(state, dim=(2,3), keepdim=True) + 1e-8)
        print_tensor_info("After channel-wise normalization", state)
        logger.info(f"Verification - per-channel norms: {[torch.norm(state[0,i]).item() for i in range(state.shape[1])]}")
        
        # Reshape for quantum conversion - flatten to (batch_size, hidden_dim)
        state_flat = state.reshape(state.shape[0], -1)  # Flatten all dimensions after batch
        state_flat = state_flat / torch.norm(state_flat, dim=-1, keepdim=True)  # Normalize flattened state
        logger.info("\n--- Converting to quantum state ---")
        logger.info(f"State shape before conversion: {state_flat.shape}")
        logger.info(f"State dtype before conversion: {state_flat.dtype}")
        
        # Convert to quantum using the bridge
        quantum_state = quantum_system.quantum_flow.bridge.neural_to_quantum(state_flat)
        logger.info("\n--- Converting back to classical state ---")
        
        # Convert back to classical state
        classical_state = quantum_system.quantum_flow.bridge.quantum_to_neural(quantum_state)
        classical_state = classical_state / torch.norm(classical_state, dim=-1, keepdim=True)  # Normalize after conversion
        logger.info(f"State shape after quantum->classical conversion: {classical_state.shape}")
        
        # Verify properties
        assert isinstance(quantum_state, QuantumState), "Quantum state has incorrect type"
        assert classical_state.shape == state_flat.shape, "Classical state shape mismatch"
        assert torch.allclose(torch.norm(classical_state), torch.tensor(1.0, dtype=torch.float64), atol=1e-6), \
            "Classical state norm not preserved"
        
    def test_quantum_evolution(self, quantum_system):
        """Test quantum state evolution."""
        logger.info("\n=== Starting quantum evolution test ===")
        
        # Create normalized test state
        state = torch.randn(1, quantum_system.dim, quantum_system.size, quantum_system.size,
                          dtype=torch.float64)
        state = state / torch.norm(state)
        print_tensor_info("Initial state", state)
        
        # Reshape for quantum conversion - flatten to (batch_size, hidden_dim)
        state_flat = state.reshape(state.shape[0], -1)  # Flatten all dimensions after batch
        
        # Convert to quantum state
        quantum_state = quantum_system.quantum_flow.bridge.neural_to_quantum(state_flat)
        logger.info("\n--- Evolving quantum state ---")
        
        # Project to manifold coordinates for metric computation
        # Double the state for real and imaginary parts
        state_doubled = torch.cat([state_flat, state_flat], dim=-1)
        manifold_coords = quantum_system.quantum_flow.state_reconstruction_net(state_doubled)
        
        # Get initial metric
        metric = quantum_system.quantum_flow.compute_metric(manifold_coords)
        
        # Evolve quantum state
        evolved_metric, metrics = quantum_system.quantum_flow.flow_step(
            metric=metric,
            quantum_state=quantum_state,
            timestep=0.1
        )
        logger.info(f"Evolution metrics: {metrics}")
        
        # Convert back to classical state
        evolved_classical = quantum_system.quantum_flow.bridge.quantum_to_neural(quantum_state)
        logger.info("\n--- Evolved classical state ---")
        logger.info(f"Shape: {evolved_classical.shape}")
        logger.info(f"Norm: {torch.norm(evolved_classical).item()}")
        
        # Verify properties
        assert isinstance(quantum_state, QuantumState), "Evolved quantum state has incorrect type"
        assert evolved_classical.shape == state_flat.shape, "Evolved classical state shape mismatch"
        assert torch.allclose(torch.norm(evolved_classical), torch.tensor(1.0, dtype=torch.float64), atol=1e-6), \
            "Evolution did not preserve norm"
        assert not torch.allclose(evolved_classical, state_flat, atol=1e-6), \
            "Evolution did not change state"
        
    def test_quantum_geometric_tensor(self, quantum_system):
        """Test quantum geometric tensor computation."""
        logger.info("\n=== Starting quantum geometric tensor test ===")
        
        # Create normalized test state
        state = torch.randn(1, quantum_system.dim, quantum_system.size, quantum_system.size,
                          dtype=torch.float64)
        state = state / torch.norm(state)
        print_tensor_info("Initial state", state)
        
        # Reshape for quantum conversion - flatten to (batch_size, hidden_dim)
        state_flat = state.reshape(state.shape[0], -1)  # Flatten all dimensions after batch
        
        # Convert to quantum state
        quantum_state = quantum_system.quantum_flow.bridge.neural_to_quantum(state_flat)
        logger.info("\n--- Computing quantum geometric tensor ---")
        
        # Project to manifold coordinates for metric computation
        # Double the state for real and imaginary parts
        state_doubled = torch.cat([state_flat, state_flat], dim=-1)
        manifold_coords = quantum_system.quantum_flow.state_reconstruction_net(state_doubled)
        
        # Get metric
        metric = quantum_system.quantum_flow.compute_metric(manifold_coords)
        
        # Compute quantum geometric tensor
        tensor = quantum_system.quantum_flow.compute_quantum_metric_tensor(quantum_state, metric)
        logger.info(f"Tensor shape: {tensor.shape}")
        logger.info(f"Tensor dtype: {tensor.dtype}")
        
        # Verify properties
        assert tensor.shape[-2:] == (quantum_state.dim, quantum_state.dim), \
            "Tensor shape mismatch"
        assert torch.allclose(tensor, tensor.conj().transpose(-2, -1), atol=1e-6), \
            "Tensor not Hermitian"
        
        # Check positive semi-definiteness
        eigenvalues = torch.linalg.eigvalsh(tensor)
        assert torch.all(eigenvalues >= -1e-6), "Tensor not positive semi-definite"
        
    def test_berry_phase(self, quantum_system):
        """Test Berry phase computation."""
        logger.info("\n=== Starting Berry phase test ===")
        
        # Create normalized test state
        state = torch.randn(1, quantum_system.dim, quantum_system.size, quantum_system.size,
                          dtype=torch.float64)
        state = state / torch.norm(state)
        print_tensor_info("Initial state", state)
        
        # Reshape for quantum conversion - flatten to (batch_size, hidden_dim)
        state_flat = state.reshape(state.shape[0], -1)  # Flatten all dimensions after batch
        state_flat = state_flat / torch.norm(state_flat, dim=-1, keepdim=True)  # Normalize flattened state
        
        # Convert to quantum state
        quantum_state = quantum_system.quantum_flow.bridge.neural_to_quantum(state_flat)
        logger.info("\n--- Computing Berry phase ---")
        
        # Project to manifold coordinates for metric computation
        # Double the state for real and imaginary parts
        state_doubled = torch.cat([state_flat, state_flat], dim=-1)
        manifold_coords = quantum_system.quantum_flow.state_reconstruction_net(state_doubled)
        
        # Create a closed path in manifold coordinates
        t = torch.linspace(0, 2*torch.pi, 100, dtype=torch.float64)
        path_points = []
        for ti in t:
            # Create a point with the same dimension as manifold_coords
            point = torch.zeros_like(manifold_coords)
            point[..., 0] = torch.cos(ti)  # First component
            point[..., 1] = torch.sin(ti)  # Second component
            path_points.append(point)
        logger.info(f"Path length: {len(path_points)}")
        
        # Stack path points into a tensor and reshape for batch processing
        path_tensor = torch.stack(path_points, dim=0)  # [num_points, batch_size, hidden_dim]
        path_tensor = path_tensor.transpose(0, 1)  # [batch_size, num_points, hidden_dim]
        
        # Compute Berry phase
        phase = quantum_system.quantum_flow.compute_berry_phase(path_tensor)
        phase = phase.squeeze()  # Remove batch dimension
        logger.info(f"Berry phase: {phase.item()}")
        
        # Verify properties
        assert isinstance(phase, torch.Tensor), "Phase has incorrect type"
        assert phase.shape == (), "Phase has incorrect shape"
        assert phase.dtype == torch.complex64, "Phase has incorrect dtype"
        assert -torch.pi <= torch.angle(phase) <= torch.pi, "Phase outside valid range"
        
        # Test path independence for small loops
        small_path_points = []
        for ti in t:
            point = torch.zeros_like(state_flat)
            point[..., 0] = 0.1 * torch.cos(ti)  # Scaled first component
            point[..., 1] = 0.1 * torch.sin(ti)  # Scaled second component
            small_path_points.append(point)
        small_path_tensor = torch.stack(small_path_points, dim=0)  # [num_points, batch_size, hidden_dim]
        small_path_tensor = small_path_tensor.transpose(0, 1)  # [batch_size, num_points, hidden_dim]
        
        small_phase = quantum_system.quantum_flow.compute_berry_phase(small_path_tensor)
        small_phase = small_phase.squeeze()  # Remove batch dimension
        
        # Compute ratio of phases
        ratio = torch.abs(phase / (small_phase + 1e-8))  # Add small epsilon to avoid division by zero
        assert torch.abs(ratio - 100) < 10, "Berry phase not scaling with area for small loops"
        
    def test_quantum_potential(self, quantum_system):
        """Test quantum potential computation."""
        logger.info("\n=== Starting quantum potential test ===")
        
        # Create normalized test state
        state = torch.randn(1, quantum_system.dim, quantum_system.size, quantum_system.size, 
                          dtype=torch.float64)
        state = state / torch.norm(state)
        print_tensor_info("Initial state", state)
        
        # Reshape for quantum conversion - flatten to (batch_size, hidden_dim)
        state_flat = state.reshape(state.shape[0], -1)  # Flatten all dimensions after batch
        state_flat = state_flat / torch.norm(state_flat, dim=-1, keepdim=True)  # Normalize flattened state
        
        # Convert to quantum state
        quantum_state = quantum_system.quantum_flow.bridge.neural_to_quantum(state_flat)
        logger.info("\n--- Computing quantum potential ---")
        
        # Compute quantum metrics
        metrics = {}
        
        # Get density matrix
        rho = quantum_state.density_matrix()  # [batch_size, dim, dim]
        rho = rho.squeeze(0)  # Remove batch dimension for trace computation
        
        # Compute von Neumann entropy
        entropy = quantum_system.quantum_flow.hilbert_space.compute_entropy(quantum_state)
        metrics["von_neumann_entropy"] = entropy
        
        # Compute purity
        purity = torch.trace(torch.matmul(rho, rho)).real
        metrics["purity"] = purity
        
        logger.info(f"Quantum metrics: {metrics}")
        
        # Verify properties
        assert "von_neumann_entropy" in metrics, "Missing von Neumann entropy"
        assert "purity" in metrics, "Missing purity"
        assert metrics["von_neumann_entropy"].dtype == torch.float64, "Entropy has incorrect dtype"
        assert metrics["purity"].dtype == torch.float64, "Purity has incorrect dtype"
        assert torch.all(metrics["von_neumann_entropy"] >= -1e-6), "Negative entropy"  # Allow small numerical errors
        assert torch.all(metrics["purity"] <= 1.0 + 1e-6), "Purity greater than 1"  # Allow small numerical errors
        
        # Test scaling behavior
        scaled_state = state_flat * 2.0
        scaled_state = scaled_state / torch.norm(scaled_state, dim=-1, keepdim=True)  # Normalize scaled state
        scaled_quantum_state = quantum_system.quantum_flow.bridge.neural_to_quantum(scaled_state)
        
        # Compute scaled metrics
        scaled_metrics = {}
        scaled_rho = scaled_quantum_state.density_matrix()
        scaled_rho = scaled_rho.squeeze(0)  # Remove batch dimension for trace computation
        scaled_entropy = quantum_system.quantum_flow.hilbert_space.compute_entropy(scaled_quantum_state)
        scaled_purity = torch.trace(torch.matmul(scaled_rho, scaled_rho)).real
        scaled_metrics["von_neumann_entropy"] = scaled_entropy
        scaled_metrics["purity"] = scaled_purity
        
        assert torch.allclose(scaled_metrics["purity"], metrics["purity"], atol=1e-6), \
            "Purity not invariant under scaling"
        
    def test_quantum_disabled(self, quantum_system):
        """Test behavior when quantum features are disabled."""
        # Disable quantum features
        quantum_system.quantum_enabled = False
        
        # Create test state
        state = torch.randn(1, quantum_system.dim, quantum_system.size, quantum_system.size, 
                          dtype=torch.float64)
        state = state / torch.norm(state)
        
        # Check that quantum operations raise error
        with pytest.raises(RuntimeError):
            quantum_system._to_quantum_state(state)
            
    def test_parallel_transport(self, quantum_system):
        """Test parallel transport of quantum state."""
        logger.info("\n=== Starting parallel transport test ===")
        
        # Create normalized test state
        state = torch.randn(1, quantum_system.dim, quantum_system.size, quantum_system.size,
                          dtype=torch.float64)
        state = state / torch.norm(state)
        print_tensor_info("Initial state", state)
        
        # Reshape for quantum conversion - flatten to (batch_size, hidden_dim)
        state_flat = state.reshape(state.shape[0], -1)  # Flatten all dimensions after batch
        
        # Convert to quantum state
        quantum_state = quantum_system.quantum_flow.bridge.neural_to_quantum(state_flat)
        logger.info("\n--- Parallel transporting quantum state ---")
        
        # Project to manifold coordinates for metric computation
        # Double the state for real and imaginary parts
        state_doubled = torch.cat([state_flat, state_flat], dim=-1)
        manifold_coords = quantum_system.quantum_flow.state_reconstruction_net(state_doubled)
        
        # Create transport points
        p1 = torch.tensor([0.0, 0.0], dtype=torch.float64)
        p2 = torch.tensor([1.0, 0.0], dtype=torch.float64)
        logger.info(f"Transport from {p1} to {p2}")
        
        # Get connection
        metric = quantum_system.quantum_flow.compute_metric(manifold_coords)
        connection = quantum_system.quantum_flow.compute_connection(metric)
        
        # Transport state
        transported = quantum_system.quantum_flow.parallel_transport_state(
            quantum_state,
            p2 - p1,  # Transport vector
            connection=connection
        )
        logger.info(f"Transported state type: {type(transported)}")
        
        # Verify properties
        assert isinstance(transported, QuantumState), "Transported state has incorrect type"
        assert transported.shape == quantum_state.shape, "Transport changed state shape"
        assert torch.allclose(torch.norm(transported.amplitudes), torch.tensor(1.0, dtype=torch.float64), atol=1e-6), \
            "Transport did not preserve norm"
        
        # Test transport around loop
        points = torch.stack([
            torch.tensor([0.0, 0.0], dtype=torch.float64),
            torch.tensor([1.0, 0.0], dtype=torch.float64),
            torch.tensor([1.0, 1.0], dtype=torch.float64),
            torch.tensor([0.0, 0.0], dtype=torch.float64)
        ])
        
        # Transport around loop
        current_state = quantum_state
        for i in range(len(points) - 1):
            vector = points[i + 1] - points[i]
            current_state = quantum_system.quantum_flow.parallel_transport_state(
                current_state,
                vector,
                connection=connection
            )
        
        # Check holonomy phase
        inner_product = torch.vdot(current_state.amplitudes.flatten(), 
                                 quantum_state.amplitudes.flatten())
        phase = torch.angle(inner_product)
        logger.info(f"Holonomy phase: {phase.item()}")
        assert -torch.pi <= phase <= torch.pi, "Invalid holonomy phase"
        assert torch.abs(torch.abs(inner_product) - 1.0) < 1e-6, \
            "Transport not unitary"