import pytest
import torch
import numpy as np
from typing import List, Optional, Tuple

# Only allow CPU devices
ALLOWED_DEVICES = ["cpu"]

@pytest.fixture
def device() -> str:
    """Get test device."""
    return "cpu"

@pytest.fixture
def test_inputs(batch_size: int = 32, seq_len: int = 128, hidden_dim: int = 64) -> torch.Tensor:
    """Generate test inputs."""
    return torch.randn(batch_size, seq_len, hidden_dim)

class AttentionTileFixture:
    """Fixture for testing attention tile implementation."""
    
    def __init__(self, input_dim: int, hidden_dim: int, device: str = "cpu"):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.device = device
        self._state: Optional[torch.Tensor] = None

    @property
    def state(self) -> Optional[torch.Tensor]:
        """Get the current state."""
        return self._state

    @state.setter
    def state(self, value: Optional[torch.Tensor]) -> None:
        """Set the current state."""
        self._state = value.to(device=self.device) if value is not None else None

    def process(self, inputs: torch.Tensor) -> torch.Tensor:
        """Process inputs through attention tile."""
        # Implementation here
        return inputs.to(device=self.device)

@pytest.fixture
def attention_tile(device: str) -> AttentionTileFixture:
    """Create an attention tile fixture."""
    return AttentionTileFixture(input_dim=128, hidden_dim=128, device=device)

@pytest.mark.dependency(name="test_tile_dimensions")
@pytest.mark.order(1)
@pytest.mark.tiling
@pytest.mark.level0
def test_tile_dimensions(attention_tile):
    """Test tile dimension properties. Level 0: Only depends on basic attribute access."""
    # Test input and hidden dimensions
    assert attention_tile.input_dim == 128
    assert attention_tile.hidden_dim == 128
    assert attention_tile.input_dim > 0
    assert attention_tile.hidden_dim > 0

@pytest.mark.dependency(depends=["test_tile_dimensions"])
@pytest.mark.order(2)
@pytest.mark.tiling
@pytest.mark.level1
def test_tile_output_shape(attention_tile):
    """Test tile output shapes. Level 1: Depends on tile dimensions."""
    # Test with different batch sizes
    batch_sizes = [1, 2, 4, 8]
    seq_len = 16
    
    for batch_size in batch_sizes:
        inputs = torch.randn(batch_size, seq_len, attention_tile.input_dim)
        output = attention_tile.process(inputs)
        
        # Check output shape
        assert output.shape == (batch_size, seq_len, attention_tile.hidden_dim)
        assert output.device.type == attention_tile.device

@pytest.mark.dependency(depends=["test_tile_dimensions", "test_tile_output_shape"])
@pytest.mark.order(3)
@pytest.mark.tiling
@pytest.mark.level2
def test_attention_tile_cpu(attention_tile, test_inputs):
    """Test attention tile on CPU. Level 2: Depends on tile dimensions and shape validation."""
    output = attention_tile.process(test_inputs)
    assert output.shape == test_inputs.shape

@pytest.mark.dependency(depends=["test_tile_dimensions", "test_tile_output_shape"])
@pytest.mark.order(4)
@pytest.mark.tiling
@pytest.mark.level2
def test_state_management(attention_tile, test_inputs):
    """Test state management. Level 2: Depends on tile dimensions and shape validation."""
    state = torch.randn(1, 128, 128)
    attention_tile.state = state
    assert attention_tile.state is not None
    assert torch.allclose(attention_tile.state, state.to(device=attention_tile.device))
