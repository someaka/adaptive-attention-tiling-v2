import torch
import pytest
import logging
from typing import Tuple, Dict, Any
from pytest import approx
from torch.autograd.profiler import profile as Profile
import torch.nn as nn

from src.core.tiling.quantum_geometric_attention import QuantumGeometricAttention

logger = logging.getLogger(__name__)

def complex_randn(*size, device=None, dtype=torch.complex64):
    """Create random complex tensor with proper initialization."""
    real_dtype = torch.float32 if dtype == torch.complex64 else torch.float64
    real = torch.randn(*size, device=device, dtype=real_dtype)
    imag = torch.randn(*size, device=device, dtype=real_dtype)
    return torch.complex(real, imag).to(dtype=dtype)

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
        manifold_dim=params["manifold_dim"],
        dtype=torch.complex64  # Use complex dtype
    )
    
    return layer, params

class TestGradientFlow:
    """Test gradient flow through quantum geometric attention."""
    
    @pytest.mark.timeout(30)  # 30 second timeout
    def test_connection_gradient_flow(self, setup_attention):
        """Test gradient flow through pattern_bundle.connection."""
        layer, params = setup_attention
        x = complex_randn(params["batch_size"], params["seq_length"], params["hidden_dim"])
        x.requires_grad_(True)
        
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
        x = complex_randn(params["batch_size"], params["seq_length"], params["hidden_dim"])
        x.requires_grad_(True)
        
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
        x = complex_randn(params["batch_size"], params["seq_length"], params["hidden_dim"])
        x.requires_grad_(True)
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
        x = complex_randn(params["batch_size"], params["seq_length"], params["hidden_dim"])
        x.requires_grad_(True)

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
        x = complex_randn(params["batch_size"], params["seq_length"], params["hidden_dim"])
        x.requires_grad_(True)

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

    @pytest.mark.timeout(30)  # 30 second timeout
    def test_pattern_metric_gradient_flow(self, setup_attention):
        """Test gradient flow through pattern_metric."""
        layer, params = setup_attention
        x = complex_randn(params["batch_size"], params["seq_length"], params["hidden_dim"])
        x.requires_grad_(True)
        
        # Get pattern_metric directly from the layer
        pattern_metric = layer.pattern_metric
        pattern_metric.requires_grad_(True)
        pattern_metric.retain_grad()  # Ensure gradients are retained
        
        # Add gradient hook
        gradients = []
        def hook(grad):
            if grad is not None:  # Only append non-None gradients
                gradients.append(grad.detach().clone())
                print(f"Pattern metric gradient shape in hook: {grad.shape}")
                print(f"Pattern metric gradient norm in hook: {grad.norm().item()}")
            return grad
        pattern_metric.register_hook(hook)
        
        # Forward pass
        output = layer(x)
        
        # For complex tensors, compute loss on both real and imaginary parts
        if torch.is_complex(output):
            loss = output.real.abs().mean() + output.imag.abs().mean()
        else:
            loss = output.abs().mean()
        loss.backward()
        
        # Check gradients
        assert len(gradients) > 0, "Pattern metric should have received gradients"
        assert gradients[0].abs().mean() > 0, "Pattern metric gradients should be non-zero"

    @pytest.mark.timeout(30)  # 30 second timeout
    def test_base_metric_gradient_flow(self, setup_attention):
        """Test gradient flow through base_metric."""
        layer, params = setup_attention
        x = complex_randn(params["batch_size"], params["seq_length"], params["hidden_dim"])
        x.requires_grad_(True)
        
        # Get base_metric directly from the layer
        base_metric = layer.base_metric
        base_metric.requires_grad_(True)
        base_metric.retain_grad()  # Ensure gradients are retained
        
        # Add gradient hook
        gradients = []
        def hook(grad):
            if grad is not None:  # Only append non-None gradients
                gradients.append(grad.detach().clone())
                print(f"Base metric gradient shape in hook: {grad.shape}")
                print(f"Base metric gradient norm in hook: {grad.norm().item()}")
            return grad
        base_metric.register_hook(hook)
        
        # Forward pass
        output = layer(x)
        
        # For complex tensors, compute loss on both real and imaginary parts
        if torch.is_complex(output):
            loss = output.real.abs().mean() + output.imag.abs().mean()
        else:
            loss = output.abs().mean()
        loss.backward()
        
        # Check gradients
        assert len(gradients) > 0, "Base metric should have received gradients"
        assert gradients[0].abs().mean() > 0, "Base metric gradients should be non-zero"

    @pytest.mark.timeout(30)  # 30 second timeout
    def test_combined_metric_gradient_flow(self, setup_attention):
        """Test gradient flow through combined_metric."""
        layer, params = setup_attention
        x = complex_randn(params["batch_size"], params["seq_length"], params["hidden_dim"])
        x.requires_grad_(True)
        
        # Get combined_metric directly from the layer
        combined_metric = layer.combined_metric
        combined_metric.requires_grad_(True)
        combined_metric.retain_grad()  # Ensure gradients are retained
        
        # Add gradient hook
        gradients = []
        def hook(grad):
            if grad is not None:  # Only append non-None gradients
                gradients.append(grad.detach().clone())
                print(f"Combined metric gradient shape in hook: {grad.shape}")
                print(f"Combined metric gradient norm in hook: {grad.norm().item()}")
            return grad
        combined_metric.register_hook(hook)
        
        # Forward pass
        output = layer(x)
        
        # For complex tensors, compute loss on both real and imaginary parts
        if torch.is_complex(output):
            loss = output.real.abs().mean() + output.imag.abs().mean()
        else:
            loss = output.abs().mean()
        loss.backward()
        
        # Check gradients
        assert len(gradients) > 0, "Combined metric should have received gradients"
        assert gradients[0].abs().mean() > 0, "Combined metric gradients should be non-zero"

    @pytest.mark.timeout(30)  # 30 second timeout
    def test_quantum_bridge_gradient_diagnostic(self, setup_attention):
        """Detailed diagnostic test for gradient flow through quantum bridge."""
        layer, params = setup_attention
        print("\n=== Starting Quantum Bridge Gradient Diagnostic ===")
        
        # Create test input
        x = complex_randn(params["batch_size"], params["seq_length"], params["hidden_dim"])
        x.requires_grad_(True)
        print(f"\nInput shape: {x.shape}")
        
        # Track intermediate tensors
        intermediate_tensors = {}
        computation_steps = []
        
        def save_tensor(name: str, tensor: torch.Tensor, step_info: str = ""):
            """Enhanced tensor tracking with computation step info."""
            if tensor.requires_grad:
                tensor.retain_grad()
                intermediate_tensors[name] = tensor
                computation_steps.append(f"Step: {step_info}")
                
                print(f"\nTracking tensor: {name}")
                print(f"Step info: {step_info}")
                print(f"Shape: {tensor.shape}")
                print(f"Requires grad: {tensor.requires_grad}")
                print(f"Is complex: {tensor.is_complex()}")
                if tensor.is_complex():
                    print(f"Complex stats:")
                    print(f"  Magnitude mean: {tensor.abs().mean().item():.6f}")
                    print(f"  Real mean: {tensor.real.mean().item():.6f}")
                    print(f"  Imag mean: {tensor.imag.mean().item():.6f}")
                
                def hook(grad):
                    if grad is not None:
                        print(f"\nGradient for {name} (Step: {step_info}):")
                        print(f"  Shape: {grad.shape}")
                        if grad.is_complex():
                            grad_abs = grad.abs()
                            print(f"  Complex Gradient stats:")
                            print(f"    Magnitude norm: {torch.norm(grad_abs).item():.6f}")
                            print(f"    Real mean: {grad.real.mean().item():.6f}")
                            print(f"    Imag mean: {grad.imag.mean().item():.6f}")
                            print(f"    Max magnitude: {grad_abs.max().item():.6f}")
                            print(f"    Min magnitude: {grad_abs.min().item():.6f}")
                        else:
                            print(f"  Gradient stats:")
                            print(f"    Norm: {torch.norm(grad).item():.6f}")
                            print(f"    Mean: {grad.mean().item():.6f}")
                            print(f"    Max: {grad.max().item():.6f}")
                            print(f"    Min: {grad.min().item():.6f}")
                        return grad
                    return grad
                
                tensor.register_hook(hook)
        
        # Forward pass with enhanced tracking
        quantum_bridge = layer.quantum_bridge
        
        # Track initial tensors
        save_tensor("input", x, "Initial input tensor")
        save_tensor("pattern_bundle.metric", quantum_bridge.pattern_bundle.metric, "Pattern bundle metric parameter")
        save_tensor("pattern_bundle.connection", quantum_bridge.pattern_bundle.connection, "Pattern bundle connection parameter")
        
        # Track metric and connection views
        metric_view = quantum_bridge.pattern_bundle.metric.clone()
        connection_view = quantum_bridge.pattern_bundle.connection.clone()
        save_tensor("metric_view", metric_view, "Cloned metric view")
        save_tensor("connection_view", connection_view, "Cloned connection view")
        
        # Track connection usage in forward pass
        x_flat = x.reshape(-1, params["hidden_dim"])
        save_tensor("x_flat", x_flat, "Flattened input")
        
        # Track intermediate quantum states
        print("\n=== Starting Forward Pass ===")
        output = layer(x)
        save_tensor("output", output, "Final output")
        
        # Compute loss and backward
        print("\n=== Starting Backward Pass ===")
        loss = output.abs().sum()
        print(f"Loss value: {loss.item():.6f}")
        loss.backward()
        
        # Log gradient flow analysis
        print("\n=== Gradient Flow Analysis ===")
        print("=" * 50)
        
        # Check each tracked tensor
        for name, tensor in intermediate_tensors.items():
            print(f"\nAnalyzing tensor: {name}")
            print(f"  Shape: {tensor.shape}")
            print(f"  Requires grad: {tensor.requires_grad}")
            if hasattr(tensor, 'grad') and tensor.grad is not None:
                grad = tensor.grad
                if grad.is_complex():
                    grad_abs = grad.abs()
                    print(f"  Complex Gradient stats:")
                    print(f"    Magnitude norm: {torch.norm(grad_abs).item():.6f}")
                    print(f"    Real mean: {grad.real.mean().item():.6f}")
                    print(f"    Imag mean: {grad.imag.mean().item():.6f}")
                    print(f"    Max magnitude: {grad_abs.max().item():.6f}")
                    print(f"    Min magnitude: {grad_abs.min().item():.6f}")
                else:
                    print(f"  Gradient stats:")
                    print(f"    Norm: {torch.norm(grad).item():.6f}")
                    print(f"    Mean: {grad.mean().item():.6f}")
                    print(f"    Max: {grad.max().item():.6f}")
                    print(f"    Min: {grad.min().item():.6f}")
            else:
                print("  No gradients")
        
        # Log computation steps
        print("\n=== Computation Steps ===")
        for i, step in enumerate(computation_steps):
            print(f"{i+1}. {step}")
        
        # Final assertions with detailed error messages
        connection_grad = quantum_bridge.pattern_bundle.connection.grad
        assert connection_grad is not None, \
            "No gradients in pattern_bundle.connection - gradient flow is blocked"
        
        # Additional assertions to verify gradient quality
        if connection_grad is not None:
            grad_abs = connection_grad.abs()
            assert torch.isfinite(grad_abs).all(), \
                "Connection gradients contain inf/nan values"
            assert grad_abs.mean() > 0, \
                f"Connection gradients are zero (mean magnitude: {grad_abs.mean().item():.6f})"

    @pytest.mark.timeout(30)  # 30 second timeout
    def test_multi_head_gradient_flow(self, setup_attention):
        """Test gradient flow through multi-head attention mechanism."""
        layer, params = setup_attention
        
        # Create complex input tensor
        x = complex_randn(params["batch_size"], params["seq_length"], params["hidden_dim"])
        x.requires_grad_(True)
        
        # Forward pass
        output = layer(x)
        
        # Track gradients for query, key, value projections in each tile
        gradients = []
        for i, tile in enumerate(layer.tiles):
            def make_hook(tile_idx):
                def hook(grad):
                    if grad is not None:
                        grad_stats = {
                            "tile": tile_idx,
                            "shape": grad.shape,
                            "norm": grad.abs().mean().item(),
                            "real_std": grad.real.std().item(),
                            "imag_std": grad.imag.std().item()
                        }
                        logger.info(f"Tile {tile_idx} gradient stats: {grad_stats}")
                        gradients.append((tile_idx, grad.detach().clone()))
                    return grad
                return hook
            
            # Register hooks for query, key, value weights
            tile.query.weight.register_hook(make_hook(i))
            tile.key.weight.register_hook(make_hook(i))
            tile.value.weight.register_hook(make_hook(i))
        
        # Compute loss using complex tensor operations
        loss = output.abs().pow(2).sum()  # Use sum() instead of mean() for stronger gradients
        loss.backward()
        
        # Check gradients
        assert len(gradients) > 0, "Should have received gradients for tile projections"
        for tile_idx, grad in gradients:
            assert grad.abs().mean() > 0, f"Tile {tile_idx} gradients should be non-zero"
            
            # Log detailed gradient statistics
            logger.info(f"\nGradient statistics for tile {tile_idx}:")
            logger.info(f"Shape: {grad.shape}")
            logger.info(f"Requires grad: {grad.requires_grad}")
            logger.info(f"Is complex: {torch.is_complex(grad)}")
            logger.info("Complex stats:")
            logger.info(f"  Magnitude mean: {grad.abs().mean().item():.6f}")
            logger.info(f"  Real mean: {grad.real.mean().item():.6f}")
            logger.info(f"  Imag mean: {grad.imag.mean().item():.6f}")
            logger.info(f"  Real std: {grad.real.std().item():.6f}")
            logger.info(f"  Imag std: {grad.imag.std().item():.6f}")

    @pytest.mark.timeout(30)  # 30 second timeout
    def test_quantum_bridge_pattern_bundle_metric_flow(self, setup_attention):
        """Test gradient flow through quantum_bridge.pattern_bundle.metric."""
        layer, params = setup_attention
        x = complex_randn(params["batch_size"], params["seq_length"], params["hidden_dim"])
        x.requires_grad_(True)
        
        # Get pattern bundle metric directly from the registered parameter
        pattern_bundle_metric = layer.quantum_bridge.pattern_bundle_metric  # Access registered parameter
        assert pattern_bundle_metric is not None, "Pattern bundle metric parameter not found"
        pattern_bundle_metric.requires_grad_(True)
        pattern_bundle_metric.retain_grad()  # Ensure gradients are retained
        
        # Track intermediate tensors
        intermediate_tensors = {}
        computation_steps = []
        
        def save_tensor(name, tensor, step_info=""):
            """Enhanced tensor tracking with computation step info."""
            if tensor.requires_grad:
                tensor.retain_grad()
                intermediate_tensors[name] = tensor
                computation_steps.append(f"Step: {step_info}")
                
                print(f"\nTracking tensor: {name}")
                print(f"Step info: {step_info}")
                print(f"Shape: {tensor.shape}")
                print(f"Requires grad: {tensor.requires_grad}")
                print(f"Is complex: {tensor.is_complex()}")
                if tensor.is_complex():
                    print(f"Complex stats:")
                    print(f"  Magnitude mean: {tensor.abs().mean().item():.6f}")
                    print(f"  Real mean: {tensor.real.mean().item():.6f}")
                    print(f"  Imag mean: {tensor.imag.mean().item():.6f}")
                
                def hook(grad):
                    if grad is not None:
                        print(f"\nGradient for {name} (Step: {step_info}):")
                        print(f"  Shape: {grad.shape}")
                        if grad.is_complex():
                            grad_abs = grad.abs()
                            print(f"  Complex Gradient stats:")
                            print(f"    Magnitude norm: {torch.norm(grad_abs).item():.6f}")
                            print(f"    Real mean: {grad.real.mean().item():.6f}")
                            print(f"    Imag mean: {grad.imag.mean().item():.6f}")
                            print(f"    Max magnitude: {grad_abs.max().item():.6f}")
                            print(f"    Min magnitude: {grad_abs.min().item():.6f}")
                        else:
                            print(f"  Gradient stats:")
                            print(f"    Norm: {torch.norm(grad).item():.6f}")
                            print(f"    Mean: {grad.mean().item():.6f}")
                            print(f"    Max: {grad.max().item():.6f}")
                            print(f"    Min: {grad.min().item():.6f}")
                        return grad
                    return grad
                
                tensor.register_hook(hook)
        
        # Track initial tensors
        save_tensor("input", x, "Initial input tensor")
        save_tensor("pattern_bundle.metric", pattern_bundle_metric, "Pattern bundle metric parameter")
        
        # Forward pass
        output = layer(x)
        save_tensor("output", output, "Final output")
        
        # Use the same loss computation as the integration test
        loss = output.abs().pow(2).sum()
        print(f"\nLoss value: {loss.item():.6f}")
        loss.backward()
        
        # Log gradient flow analysis
        print("\n=== Gradient Flow Analysis ===")
        print("=" * 50)
        
        # Check each tracked tensor
        for name, tensor in intermediate_tensors.items():
            print(f"\nAnalyzing tensor: {name}")
            print(f"  Shape: {tensor.shape}")
            print(f"  Requires grad: {tensor.requires_grad}")
            if hasattr(tensor, 'grad') and tensor.grad is not None:
                grad = tensor.grad
                if grad.is_complex():
                    grad_abs = grad.abs()
                    print(f"  Complex Gradient stats:")
                    print(f"    Magnitude norm: {torch.norm(grad_abs).item():.6f}")
                    print(f"    Real mean: {grad.real.mean().item():.6f}")
                    print(f"    Imag mean: {grad.imag.mean().item():.6f}")
                    print(f"    Max magnitude: {grad_abs.max().item():.6f}")
                    print(f"    Min magnitude: {grad_abs.min().item():.6f}")
                else:
                    print(f"  Gradient stats:")
                    print(f"    Norm: {torch.norm(grad).item():.6f}")
                    print(f"    Mean: {grad.mean().item():.6f}")
                    print(f"    Max: {grad.max().item():.6f}")
                    print(f"    Min: {grad.min().item():.6f}")
            else:
                print("  No gradients")
        
        # Log computation steps
        print("\n=== Computation Steps ===")
        for i, step in enumerate(computation_steps):
            print(f"{i+1}. {step}")
        
        # Final assertions with detailed error messages
        metric_grad = pattern_bundle_metric.grad
        assert metric_grad is not None, \
            "No gradients in pattern_bundle.metric - gradient flow is blocked"
        
        # Additional assertions to verify gradient quality
        if metric_grad is not None:
            grad_abs = metric_grad.abs()
            assert torch.isfinite(grad_abs).all(), \
                "Metric gradients contain inf/nan values"
            assert grad_abs.mean() > 0, \
                f"Metric gradients are zero (mean magnitude: {grad_abs.mean().item():.6f})"

    @pytest.mark.timeout(30)  # 30 second timeout
    def test_metric_factors_gradient_flow(self, setup_attention):
        """Test gradient flow through quantum_bridge.pattern_bundle.riemannian_framework.metric_factors."""
        layer, params = setup_attention
        x = complex_randn(params["batch_size"], params["seq_length"], params["hidden_dim"])
        x.requires_grad_(True)
        
        # Get metric_factors directly from the riemannian framework
        metric_factors = layer.quantum_bridge.pattern_bundle.riemannian_framework.metric_factors
        assert metric_factors is not None, "Metric factors parameter not found"
        metric_factors.requires_grad_(True)
        metric_factors.retain_grad()  # Ensure gradients are retained
        
        # Add gradient hook
        gradients = []
        def hook(grad):
            if grad is not None:  # Only append non-None gradients
                gradients.append(grad.detach().clone())
                print(f"Metric factors gradient shape in hook: {grad.shape}")
                print(f"Metric factors gradient norm in hook: {grad.norm().item()}")
            return grad
        metric_factors.register_hook(hook)
        
        # Forward pass
        output = layer(x)
        
        # For complex tensors, compute loss on both real and imaginary parts
        if torch.is_complex(output):
            loss = output.real.abs().mean() + output.imag.abs().mean()
        else:
            loss = output.abs().mean()
        loss.backward()
        
        # Check gradients
        assert len(gradients) > 0, "Metric factors should have received gradients"
        assert gradients[0].abs().mean() > 0, "Metric factors gradients should be non-zero"

    @pytest.mark.timeout(30)  # 30 second timeout
    def test_connection_coeffs_gradient_flow(self, setup_attention):
        """Test gradient flow through connection coefficients."""
        layer, params = setup_attention
        x = complex_randn(params["batch_size"], params["seq_length"], params["hidden_dim"])
        x.requires_grad_(True)
        
        # Get the original connection parameter
        connection = layer.quantum_bridge.pattern_bundle.connection
        connection.requires_grad_(True)
        connection.retain_grad()
        
        # Track gradients for the connection parameter
        connection_grads = []
        def connection_hook(grad):
            if grad is not None:
                connection_grads.append(grad.detach().clone())
                print(f"\nConnection gradient stats:")
                print(f"Shape: {grad.shape}")
                print(f"Norm: {grad.norm().item():.6f}")
                print(f"Mean: {grad.mean().item():.6f}")
            return grad
        connection.register_hook(connection_hook)
        
        # Forward pass
        output = layer(x)
        
        # Compute loss that ensures connection is used
        loss = output.abs().pow(2).sum()  # Use squared loss for stronger gradients
        print(f"\nLoss value: {loss.item():.6f}")
        loss.backward()
        
        # Check connection gradients
        assert len(connection_grads) > 0, "Connection should have received gradients"
        grad = connection_grads[0]
        grad_norm = grad.abs().mean().item()
        print(f"\nConnection gradient norm: {grad_norm:.6f}")
        assert grad_norm > 0, f"Connection gradients are zero (norm: {grad_norm:.6f})"
        
        # Verify gradient properties
        assert torch.isfinite(grad).all(), "Connection gradients contain inf/nan values"
        assert grad.shape == connection.shape, f"Gradient shape {grad.shape} doesn't match parameter shape {connection.shape}"

    @pytest.mark.timeout(30)  # 30 second timeout
    def test_energy_conservation(self, setup_attention):
        """Test energy conservation during gradient flow."""
        layer, params = setup_attention
        x = complex_randn(params["batch_size"], params["seq_length"], params["hidden_dim"])
        x.requires_grad_(True)
        
        # Forward pass
        output = layer(x)
        
        # Compute initial energy
        def compute_energy(tensor):
            """Compute energy of a complex tensor."""
            return torch.sum(tensor.real ** 2 + tensor.imag ** 2)
        
        initial_energy = compute_energy(x)
        final_energy = compute_energy(output)
        
        # Check energy conservation with tolerance
        assert torch.allclose(
            initial_energy, final_energy,
            rtol=1e-2,  # 1% relative tolerance
            atol=1e-2   # Small absolute tolerance
        ), "Energy should be approximately conserved"
        
        # Backward pass should also conserve energy
        loss = output.abs().mean()
        loss.backward()
        
        # Check gradient energy conservation
        if x.grad is not None:
            grad_energy = compute_energy(x.grad)
            assert torch.isfinite(grad_energy), "Gradient energy should be finite"
            assert grad_energy > 0, "Gradient energy should be positive"

    @pytest.mark.timeout(30)  # 30 second timeout
    def test_dtype_consistency(self, setup_attention):
        """Test dtype consistency throughout the network."""
        layer, params = setup_attention
        
        # Test with both real and complex inputs
        real_input = torch.randn(params["batch_size"], params["seq_length"], params["hidden_dim"])
        complex_input = complex_randn(params["batch_size"], params["seq_length"], params["hidden_dim"])
        
        # Check real input
        real_input.requires_grad_(True)
        try:
            output_real = layer(real_input)
            assert output_real.dtype == torch.complex64, "Output should be complex64 regardless of input type"
        except RuntimeError as e:
            if "must have the same dtype" in str(e):
                print("Layer requires complex input")
            else:
                raise e
        
        # Check complex input
        complex_input.requires_grad_(True)
        output_complex = layer(complex_input)
        assert output_complex.dtype == torch.complex64, "Output should be complex64"
        
        # Check parameter dtypes
        for name, param in layer.named_parameters():
            assert param.dtype in [torch.complex64, torch.float32], \
                f"Parameter {name} has unexpected dtype {param.dtype}"
            
        # Test gradient dtypes
        loss = output_complex.abs().mean()
        loss.backward()
        
        for name, param in layer.named_parameters():
            if param.grad is not None:
                assert param.grad.dtype == param.dtype, \
                    f"Gradient dtype mismatch for {name}: param {param.dtype} vs grad {param.grad.dtype}"