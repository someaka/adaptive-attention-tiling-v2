"""Tests for attention state validation."""
import pytest
import torch

from src.core.tiling.attention_state import AttentionState
from src.core.quantum.state_space import QuantumState
from src.core.tiling.state_manager import StateManager, StateConfig, StateType

@pytest.fixture
def hidden_dim() -> int:
    """Return hidden dimension for tests."""
    return 64

@pytest.fixture
def num_heads() -> int:
    """Return number of attention heads for tests."""
    return 8

@pytest.fixture
def batch_size() -> int:
    """Return batch size for tests."""
    return 16

@pytest.fixture
def seq_length() -> int:
    """Return sequence length for tests."""
    return 32

@pytest.fixture
def device() -> torch.device:
    """Return device for tests."""
    return torch.device('cpu')

@pytest.fixture
def dtype() -> torch.dtype:
    """Return dtype for tests."""
    return torch.float32

@pytest.fixture
def attention_state():
    """Create attention state fixture for testing."""
    batch_size = 16
    num_heads = 8
    seq_length = 32
    hidden_dim = 64
    
    attention_state = AttentionState.initialize(
        hidden_dim=hidden_dim,
        num_heads=num_heads,
        batch_size=batch_size,
        seq_length=seq_length
    )
    
    return attention_state

def test_initialization(attention_state: AttentionState):
    """Test successful initialization of AttentionState."""
    assert attention_state.state_manager is not None
    assert attention_state.geometric_state is not None
    assert attention_state.state_manager.states["input"] is not None
    assert attention_state.state_manager.states["manifold"] is not None

def test_state_validation(attention_state: AttentionState):
    """Test state validation."""
    # Test valid state
    valid_state = attention_state.geometric_state.clone()
    assert attention_state.validate_state(valid_state)

    # Test invalid states
    invalid_states = [
        torch.randn(10),  # Wrong dimensions
        torch.ones(2, 2, 2, 2, dtype=torch.int),  # Wrong dtype
        torch.full((2, 2, 2, 2), float('inf')),  # Contains inf
        torch.full((2, 2, 2, 2), float('nan'))  # Contains nan
    ]
    
    for state in invalid_states:
        assert not attention_state.validate_state(state)

def test_complex_state_handling(
    attention_state: AttentionState,
    hidden_dim: int,
    num_heads: int,
    batch_size: int,
    seq_length: int,
    dtype: torch.dtype
):
    """Test complex state operations."""
    # Create complex state with phase
    real = torch.randn(batch_size, num_heads, seq_length, hidden_dim)
    imag = torch.randn(batch_size, num_heads, seq_length, hidden_dim)
    state = torch.complex(real, imag)
    
    # Test state validation
    assert attention_state.validate_state(state)

    # Test phase invariance
    phase = torch.exp(1j * torch.rand(1))
    phase_shifted = state * phase
    assert attention_state.validate_state(phase_shifted)

    # Test state update with complex values
    key = "complex_state"
    updated = attention_state.update_quantum_state(key, state)
    assert isinstance(updated, QuantumState)
    
    # Verify quantum state properties
    assert updated.amplitudes is not None
    assert updated.amplitudes.dtype == torch.complex128
    assert updated.amplitudes.shape == (batch_size, num_heads, seq_length, hidden_dim)
    assert torch.is_complex(updated.amplitudes)
    
    # Verify phase tracking
    assert updated.phase is not None
    assert updated.phase.dtype == torch.complex128
    
    # Verify per-head normalization
    norms = torch.sqrt(torch.sum(torch.abs(updated.amplitudes) ** 2, 
                                dim=(-2, -1)))  # Sum over sequence and hidden dims
    assert torch.allclose(norms, torch.ones_like(norms), rtol=1e-5)
    
    # Verify original norm is stored and has correct shape for multi-head
    assert updated.original_norm is not None
    assert updated.original_norm.shape == (batch_size, num_heads, 1, 1)
    
    # Test phase invariance of the quantum state
    phase_new = torch.exp(1j * torch.rand(1))
    phase_shifted_amplitudes = updated.amplitudes * phase_new
    
    # The new state should still be normalized per head
    phase_shifted_norms = torch.sqrt(torch.sum(torch.abs(phase_shifted_amplitudes) ** 2, 
                                             dim=(-2, -1)))  # Sum over sequence and hidden dims
    assert torch.allclose(phase_shifted_norms, torch.ones_like(phase_shifted_norms), rtol=1e-5)
    
    # Test density matrix computation
    rho = updated.density_matrix()
    assert rho.shape == (batch_size, num_heads, seq_length, hidden_dim, hidden_dim)
    assert torch.is_complex(rho)
    
    # Verify density matrix properties
    # 1. Hermiticity
    assert torch.allclose(rho, rho.transpose(-1, -2).conj(), rtol=1e-5)
    
    # 2. Trace = 1 for each state in batch and head
    traces = torch.diagonal(rho, dim1=-2, dim2=-1).sum(-1)
    assert torch.allclose(traces.real, torch.ones_like(traces.real), rtol=1e-5)
    assert torch.allclose(traces.imag, torch.zeros_like(traces.imag), rtol=1e-5)

def test_batch_processing(
    attention_state: AttentionState,
    hidden_dim: int,
    batch_size: int,
    num_heads: int,
    seq_length: int,
    dtype: torch.dtype
):
    """Test batch processing capabilities."""
    # Create batch of states
    states = torch.randn(batch_size, num_heads, seq_length, hidden_dim, dtype=dtype)
    states = states / torch.norm(states, dim=-1, keepdim=True)

    # Test batch update
    key = "batch_states"
    updated = attention_state.update_quantum_state(key, states)
    assert isinstance(updated, QuantumState)
    assert updated.amplitudes.shape == (batch_size, num_heads, seq_length, hidden_dim)

def test_entanglement_tracking(attention_state: AttentionState):
    """Test entanglement tracking."""
    source_scale = 1.0
    target_scale = 2.0
    entropy = torch.tensor(0.5)

    # Track entanglement
    attention_state.track_entanglement(source_scale, target_scale, entropy)
    key = f"{source_scale:.1f}->{target_scale:.1f}"
    assert key in attention_state.entanglement_history
    assert len(attention_state.entanglement_history[key]) == 1
    assert attention_state.entanglement_history[key][0] == 0.5

def test_quantum_state_updates(
    attention_state: AttentionState,
    hidden_dim: int,
    batch_size: int,
    num_heads: int,
    seq_length: int,
    dtype: torch.dtype
):
    """Test quantum state updates through state manager."""
    # Create test quantum state
    state = torch.randn(batch_size, num_heads, seq_length, hidden_dim, dtype=dtype)

    # Update through state manager
    key = "test_state"
    updated_state = attention_state.update_quantum_state(key, state)

    # Verify quantum state properties
    assert isinstance(updated_state, QuantumState)
    assert updated_state.amplitudes is not None
    assert updated_state.amplitudes.shape == (batch_size, num_heads, seq_length, hidden_dim)
    assert updated_state.amplitudes.dtype == torch.complex128
    
    # Verify original norm is stored (one per batch and head)
    assert hasattr(updated_state, 'original_norm')
    assert updated_state.original_norm is not None
    assert updated_state.original_norm.shape == (batch_size, num_heads, 1, 1)

def test_metric_updates(attention_state: AttentionState):
    """Test metric updates."""
    key = "test_metric"
    value = torch.tensor(0.5)
    attention_state.update_metrics(key, value)
    assert key in attention_state.metrics
    assert torch.equal(attention_state.metrics[key], value)

def test_attention_pattern_updates(attention_state: AttentionState):
    """Test attention pattern updates."""
    key = "test_pattern"
    pattern = torch.randn(4, 4)
    attention_state.update_attention_pattern(key, pattern)
    assert key in attention_state.attention_patterns
    assert torch.equal(attention_state.attention_patterns[key], pattern)

def test_state_manager_integration(attention_state: AttentionState):
    """Test integration with state manager for updates."""
    # Get initial state shape
    batch_size, num_heads, seq_length, hidden_dim = attention_state.geometric_state.shape

    # Create new state with same dimensions
    new_state = torch.randn(batch_size, num_heads, seq_length, hidden_dim)
    
    # Update quantum state with new state
    quantum_state = attention_state.update_quantum_state("test", new_state)

    # Verify state was stored in manager
    assert "test" in attention_state.state_manager.states
    stored_state = attention_state.state_manager.states["test"]

    # Verify quantum state properties
    assert isinstance(quantum_state, QuantumState)
    assert quantum_state.amplitudes is not None
    assert quantum_state.amplitudes.shape == (batch_size, num_heads, seq_length, hidden_dim)
    assert quantum_state.amplitudes.dtype == torch.complex128
    assert quantum_state.phase is not None
    assert quantum_state.phase.dtype == torch.complex128

    # Verify per-head normalization
    norms = torch.sqrt(torch.sum(torch.abs(quantum_state.amplitudes) ** 2, 
                                dim=(-2, -1)))  # Sum over sequence and hidden dims
    assert torch.allclose(norms, torch.ones_like(norms), rtol=1e-5)

    # Test complex state update
    update_state = torch.complex(
        torch.randn(batch_size, num_heads, seq_length, hidden_dim),
        torch.randn(batch_size, num_heads, seq_length, hidden_dim)
    )
    
    # Get the current state before update
    current_state = attention_state.state_manager.states["test"]

    # Update quantum state again
    quantum_state = attention_state.update_quantum_state("test", update_state)

    # Verify the update resulted in a valid quantum state
    stored_state = attention_state.state_manager.states["test"]
    
    # Verify per-head normalization is maintained
    norms = torch.sqrt(torch.sum(torch.abs(stored_state) ** 2, 
                                dim=(-2, -1)))  # Sum over sequence and hidden dims
    assert torch.allclose(norms, torch.ones_like(norms), rtol=1e-5)

    # Verify quantum state is complex
    assert torch.is_complex(quantum_state.amplitudes)
    assert quantum_state.amplitudes.dtype == torch.complex128

    # Verify phase invariance
    phase = torch.exp(1j * torch.rand(1))
    phase_shifted = quantum_state.amplitudes * phase
    phase_shifted_norms = torch.sqrt(torch.sum(torch.abs(phase_shifted) ** 2, 
                                             dim=(-2, -1)))  # Sum over sequence and hidden dims
    assert torch.allclose(phase_shifted_norms, torch.ones_like(phase_shifted_norms), rtol=1e-5)

def test_error_handling(
    attention_state: AttentionState,
    hidden_dim: int,
    batch_size: int,
    num_heads: int,
    seq_length: int,
    dtype: torch.dtype
):
    """Test error handling in state operations."""
    # Test invalid state update
    with pytest.raises(ValueError, match="Invalid state tensor"):
        attention_state.update_quantum_state(
            "quantum",  # This key should exist
            torch.randn(hidden_dim)  # Wrong shape
        )

    # Test invalid entanglement tracking
    with pytest.raises(ValueError, match="Entropy must be a scalar tensor"):
        attention_state.track_entanglement(1.0, 2.0, torch.randn(2))  # Wrong shape 

def test_initialize_classmethod(
    hidden_dim: int,
    num_heads: int,
    batch_size: int,
    seq_length: int,
    dtype: torch.dtype,
    device: torch.device
):
    """Test the initialize classmethod."""
    # Test successful initialization
    state = AttentionState.initialize(
        hidden_dim=hidden_dim,
        num_heads=num_heads,
        batch_size=batch_size,
        seq_length=seq_length,
        dtype=dtype,
        device=device
    )
    assert isinstance(state, AttentionState)
    assert state.geometric_state.shape == (batch_size, num_heads, seq_length, hidden_dim)
    
    # Test invalid dimensions
    with pytest.raises(ValueError, match="All dimensions must be positive"):
        AttentionState.initialize(hidden_dim=-1, num_heads=num_heads)
    with pytest.raises(ValueError, match="All dimensions must be positive"):
        AttentionState.initialize(hidden_dim=hidden_dim, num_heads=0)

def test_invalid_state_manager_config():
    """Test handling of invalid state manager configurations."""
    # Create state with mismatched dimensions
    batch_size = 16
    num_heads = 8
    seq_length = 32
    hidden_dim = 64
    
    attention_state = AttentionState.initialize(
        hidden_dim=hidden_dim,
        num_heads=num_heads,
        batch_size=batch_size,
        seq_length=seq_length
    )
    
    # Try to update with invalid state
    invalid_state = torch.randn(batch_size, num_heads, seq_length, hidden_dim // 2)
    with pytest.raises(ValueError, match="Invalid state tensor"):
        attention_state.update_quantum_state("test", invalid_state)
        
    # Try to update with invalid dimensions
    invalid_state = torch.randn(batch_size, num_heads + 1, seq_length, hidden_dim)
    with pytest.raises(ValueError, match="Invalid state tensor"):
        attention_state.update_quantum_state("test", invalid_state)

def test_quantum_properties(attention_state: AttentionState):
    """Test quantum properties of states."""
    # Test that quantum state updates preserve quantum properties
    batch_size, num_heads, seq_length, hidden_dim = attention_state.geometric_state.shape
    state = torch.randn_like(attention_state.geometric_state)
    quantum_state = attention_state.update_quantum_state("test", state)

    # 1. Check per-head normalization
    norms = torch.sqrt(torch.sum(torch.abs(quantum_state.amplitudes) ** 2, 
                               dim=(-2, -1)))  # Sum over sequence and hidden dims
    assert torch.allclose(norms, torch.ones_like(norms), rtol=1e-5)

    # 2. Check density matrix properties for each head
    # Take first sequence element of each batch and head for memory efficiency
    sample_states = quantum_state.amplitudes[:, :, 0]  # Shape: [batch_size, num_heads, hidden_dim]
    
    # Normalize sample states per head
    sample_norms = torch.sqrt(torch.sum(torch.abs(sample_states) ** 2, dim=-1, keepdim=True))
    sample_states = sample_states / sample_norms.clamp(min=1e-8)
    
    # Compute density matrices for samples
    density_matrices = torch.einsum('bhd,bhe->bhde', 
                                  sample_states, 
                                  sample_states.conj())

    # Check hermiticity
    assert torch.allclose(density_matrices, 
                         density_matrices.transpose(-2, -1).conj(), 
                         rtol=1e-5)

    # Check trace = 1 for each batch and head
    traces = torch.diagonal(density_matrices, dim1=-2, dim2=-1).sum(-1)
    assert torch.allclose(traces, torch.ones_like(traces), rtol=1e-5)

    # 3. Check phase invariance per head
    phase = torch.exp(1j * torch.rand(1))
    phase_shifted = quantum_state.amplitudes * phase
    phase_shifted_norms = torch.sqrt(torch.sum(torch.abs(phase_shifted) ** 2,
                                             dim=(-2, -1)))  # Sum over sequence and hidden dims
    assert torch.allclose(phase_shifted_norms, torch.ones_like(phase_shifted_norms), rtol=1e-5)

def test_mask_handling(
    attention_state: AttentionState,
    batch_size: int,
    num_heads: int,
    seq_length: int
):
    """Test attention mask handling."""
    # Test default masks
    assert attention_state.key_padding_mask is not None
    assert attention_state.key_padding_mask.shape == (batch_size, seq_length)
    assert attention_state.key_padding_mask.dtype == torch.bool
    assert attention_state.key_padding_mask.all()  # All True by default

    assert attention_state.attention_mask is not None
    assert attention_state.attention_mask.shape == (batch_size, num_heads, seq_length, seq_length)
    assert attention_state.attention_mask.dtype == torch.bool
    assert attention_state.attention_mask.all()  # All True by default

    # Test setting key padding mask
    key_padding_mask = torch.ones(batch_size, seq_length, dtype=torch.bool)
    key_padding_mask[:, -1] = False  # Mask out last token
    attention_state.set_key_padding_mask(key_padding_mask)
    assert torch.equal(attention_state.key_padding_mask, key_padding_mask)

    # Test setting attention mask
    attention_mask = torch.ones(seq_length, seq_length, dtype=torch.bool)
    attention_mask.triu_(1).logical_not_()  # Causal mask
    attention_state.set_attention_mask(attention_mask)
    assert torch.equal(attention_state.attention_mask, attention_mask)

    # Test head-specific attention mask
    head_mask = torch.ones(batch_size, num_heads, seq_length, seq_length, dtype=torch.bool)
    head_mask[:, :, :, -1] = False  # Can't attend to last token
    attention_state.set_attention_mask(head_mask)
    assert torch.equal(attention_state.attention_mask, head_mask)

    # Test applying masks to attention scores
    scores = torch.randn(batch_size, num_heads, seq_length, seq_length)
    masked_scores = attention_state.apply_masks(scores)
    
    # Check that masked positions are -inf
    assert torch.isinf(masked_scores[:, :, :, -1]).all()
    assert (masked_scores[:, :, :, -1] < 0).all()

def test_causal_initialization(
    hidden_dim: int,
    num_heads: int,
    batch_size: int,
    seq_length: int,
    device: torch.device
):
    """Test initialization with causal masking."""
    state = AttentionState.initialize(
        hidden_dim=hidden_dim,
        num_heads=num_heads,
        batch_size=batch_size,
        seq_length=seq_length,
        device=device,
        causal=True
    )

    # Verify causal mask was created
    assert state.attention_mask is not None
    assert state.attention_mask.shape == (batch_size, num_heads, seq_length, seq_length)
    
    # Verify it's a proper causal mask
    causal = torch.triu(torch.ones(seq_length, seq_length, dtype=torch.bool), diagonal=1).logical_not()
    causal = causal.expand(batch_size, num_heads, seq_length, seq_length)
    assert torch.equal(state.attention_mask, causal)

def test_invalid_masks():
    """Test handling of invalid masks."""
    state = AttentionState.initialize(
        hidden_dim=64,
        num_heads=8,
        batch_size=16,
        seq_length=32
    )

    # Test invalid key padding mask shapes
    with pytest.raises(ValueError, match="Key padding mask must be 2-dimensional"):
        state.set_key_padding_mask(torch.ones(16, 32, 32, dtype=torch.bool))  # Wrong dims

    with pytest.raises(ValueError, match="Key padding mask sequence length must match geometric state"):
        state.set_key_padding_mask(torch.ones(16, 64, dtype=torch.bool))  # Wrong seq length

    # Test invalid attention mask shapes
    with pytest.raises(ValueError, match="Attention mask must be 2D or 4D"):
        state.set_attention_mask(torch.ones(32, 32, 32, dtype=torch.bool))  # Wrong dims

    with pytest.raises(ValueError, match="2D attention mask must have shape"):
        state.set_attention_mask(torch.ones(64, 64, dtype=torch.bool))  # Wrong size

    # Test invalid mask dtypes
    with pytest.raises(ValueError, match="Key padding mask must be a boolean tensor"):
        state.set_key_padding_mask(torch.ones(16, 32))  # Wrong dtype

    with pytest.raises(ValueError, match="Attention mask must be a boolean tensor"):
        state.set_attention_mask(torch.ones(32, 32))  # Wrong dtype 