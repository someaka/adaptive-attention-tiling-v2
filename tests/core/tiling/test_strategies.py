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

def test_attention_tile_cpu(attention_tile, test_inputs):
    """Test attention tile on CPU."""
    output = attention_tile.process(test_inputs)
    assert output.shape == test_inputs.shape

def test_state_management(attention_tile, test_inputs):
    """Test state management across operations."""
    state = torch.randn(1, 128, 128)
    attention_tile.state = state
    assert attention_tile.state is not None
    assert torch.allclose(attention_tile.state, state.to(device=attention_tile.device))
