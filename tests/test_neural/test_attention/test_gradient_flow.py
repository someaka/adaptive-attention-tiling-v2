import torch
import pytest
from typing import Tuple, Dict, Any
from pytest import approx

from src.core.tiling.quantum_geometric_attention import QuantumGeometricAttention

@pytest.fixture
def setup_attention() -> Tuple[QuantumGeometricAttention, Dict[str, Any]]:
    """Setup attention layer and parameters for testing."""
    params = {
        "batch_size": 2,  # Reduced batch size
        "hidden_dim": 16,  # Reduced hidden dim
        "seq_length": 4,
        "num_heads": 2,  # Reduced heads
        "manifold_dim": 8  # Reduced manifold dim
    }
    
    layer = QuantumGeometricAttention(
        hidden_dim=params["hidden_dim"],
        num_heads=params["num_heads"],
        manifold_dim=params["manifold_dim"]
    )
    
    return layer, params

class TestGradientFlow:
    """Test gradient flow through quantum geometric attention."""
    
    @pytest.mark.timeout(30)  # 30 second timeout
    def test_connection_gradient_flow(self, setup_attention):
        """Test gradient flow through pattern_bundle.connection."""
        layer, params = setup_attention
        x = torch.randn(params["batch_size"], params["seq_length"], params["hidden_dim"], requires_grad=True)
        
        # Get initial connection parameter
        connection = layer.quantum_bridge.pattern_bundle.connection
        assert connection.requires_grad, "Connection should require gradients"
        
        # Forward pass
        output = layer(x)
        
        # Compute loss and backward
        loss = output.abs().mean()
        loss.backward()
        
        # Check gradients
        assert connection.grad is not None, "Connection should have gradients"
        assert connection.grad.abs().mean() > 0, "Connection gradients should be non-zero"
    
    @pytest.mark.timeout(30)  # 30 second timeout
    def test_metric_view_gradient_flow(self, setup_attention):
        """Test gradient flow through metric_view."""
        layer, params = setup_attention
        x = torch.randn(params["batch_size"], params["seq_length"], params["hidden_dim"], requires_grad=True)
        
        # Get metric view directly from the layer
        metric = layer.metric
        metric.requires_grad_(True)
        
        # Add gradient hook
        gradients = []
        def hook(grad):
            gradients.append(grad)
            return grad
        metric.register_hook(hook)
        
        # Forward pass
        output = layer(x)
        
        # Compute loss and backward
        loss = output.abs().mean()
        loss.backward()
        
        # Check gradients
        assert len(gradients) > 0, "Metric view should have received gradients"
        assert gradients[0].abs().mean() > 0, "Metric view gradients should be non-zero"
    
    @pytest.mark.timeout(30)  # 30 second timeout
    def test_connection_view_gradient_flow(self, setup_attention):
        """Test gradient flow through connection_view."""
        layer, params = setup_attention
        x = torch.randn(params["batch_size"], params["seq_length"], params["hidden_dim"], requires_grad=True)
        print(f"\nInput x dtype: {x.dtype}")

        # Get connection view directly from the layer
        connection = layer.quantum_bridge.pattern_bundle.connection
        print(f"Connection dtype: {connection.dtype}")
        connection.requires_grad_(True)
        connection.retain_grad()  # Ensure gradients are retained

        # Add gradient hook
        gradients = []
        def hook(grad):
            if grad is not None:  # Only append non-None gradients
                gradients.append(grad.detach().clone())
                print(f"Gradient dtype in hook: {grad.dtype}")
                print(f"Gradient shape in hook: {grad.shape}")
                print(f"Gradient norm in hook: {grad.norm().item()}")
            return grad
        connection.register_hook(hook)

        # Forward pass
        output = layer(x)

        # For complex tensors, compute loss on both real and imaginary parts
        if torch.is_complex(output):
            loss = output.real.abs().mean() + output.imag.abs().mean()
        else:
            loss = output.abs().mean()
        loss.backward()

        # Check gradients
        assert len(gradients) > 0, "Connection should have received gradients"

    @pytest.mark.timeout(30)  # 30 second timeout
    def test_flattened_input_gradient_flow(self, setup_attention):
        """Test gradient flow through flattened input tensor."""
        layer, params = setup_attention
        x = torch.randn(params["batch_size"], params["seq_length"], params["hidden_dim"], requires_grad=True)

        # Forward pass using original input (x)
        output = layer(x)  # This will create and store x_flat inside the layer

        # Get x_flat from the layer
        x_flat = layer.x_flat

        # Add gradient hook
        gradients = []
        def hook(grad):
            if grad is not None:  # Only append non-None gradients
                gradients.append(grad.detach().clone())
                print(f"x_flat gradient shape in hook: {grad.shape}")
                print(f"x_flat gradient norm in hook: {grad.norm().item()}")
            return grad
        x_flat.register_hook(hook)

        # For complex tensors, compute loss on both real and imaginary parts
        if torch.is_complex(output):
            loss = output.real.abs().mean() + output.imag.abs().mean()
        else:
            loss = output.abs().mean()
        loss.backward()

        # Check gradients
        assert len(gradients) > 0, "Flattened input should have received gradients"

    @pytest.mark.timeout(30)  # 30 second timeout
    def test_end_to_end_gradient_flow(self, setup_attention):
        """Test end-to-end gradient flow through all components."""
        layer, params = setup_attention
        x = torch.randn(params["batch_size"], params["seq_length"], params["hidden_dim"], requires_grad=True)

        # Forward pass first to create tensors
        output = layer(x)

        # Track all components using the tensors from the layer
        components = {
            "connection": layer.quantum_bridge.pattern_bundle.connection,
            "metric": layer.metric,
            "x_flat": layer.x_flat  # Use stored x_flat
        }

        # Add gradient hooks to all components
        gradients = {name: [] for name in components}
        hooks = {}  # Store hooks to prevent garbage collection
        for name, comp in components.items():
            def make_hook(name):
                def hook(grad):
                    if grad is not None:  # Only append non-None gradients
                        gradients[name].append(grad.detach().clone())
                        print(f"{name} gradient shape in hook: {grad.shape}")
                        print(f"{name} gradient norm in hook: {grad.norm().item()}")
                    return grad
                return hook
            hooks[name] = make_hook(name)
            comp.register_hook(hooks[name])

        # For complex tensors, compute loss on both real and imaginary parts
        if torch.is_complex(output):
            loss = output.real.abs().mean() + output.imag.abs().mean()
        else:
            loss = output.abs().mean()
        loss.backward()

        # Check gradients
        for name, grads in gradients.items():
            assert len(grads) > 0, f"{name} should have received gradients"