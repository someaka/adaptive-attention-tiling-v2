import torch
import pytest
import logging
from typing import Tuple, Dict, Any
from pytest import approx
from torch.autograd.profiler import profile as Profile
import torch.nn as nn

from src.core.tiling.quantum_geometric_attention import QuantumGeometricAttention
from src.core.patterns.riemannian_flow import RiemannianFlow

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
    
    def create_input_tensor(self, params: Dict[str, Any], requires_grad: bool = True) -> torch.Tensor:
        """Helper method to create properly shaped input tensor.
        
        Args:
            params: Dictionary containing batch_size, num_heads, seq_length, and hidden_dim
            requires_grad: Whether the tensor should require gradients
            
        Returns:
            Tensor of shape [batch_size, num_heads, seq_length, head_dim]
        """
        head_dim = params["hidden_dim"] // params["num_heads"]
        x = complex_randn(
            params["batch_size"], 
            params["num_heads"], 
            params["seq_length"], 
            head_dim
        )
        x.requires_grad_(requires_grad)
        return x
    
    def reshape_for_layer(self, x: torch.Tensor, params: Dict[str, Any]) -> torch.Tensor:
        """Helper method to reshape input tensor for layer input.
        
        Args:
            x: Input tensor of shape [batch_size, num_heads, seq_length, head_dim]
            params: Dictionary containing hidden_dim
            
        Returns:
            Tensor of shape [batch_size, seq_length, hidden_dim]
        """
        return x.reshape(params["batch_size"], params["seq_length"], params["hidden_dim"])

    @pytest.fixture(autouse=True)
    def enable_anomaly_detection(self):
        """Enable gradient anomaly detection for all tests."""
        torch.autograd.set_detect_anomaly(True, check_nan=True)
        yield
        torch.autograd.set_detect_anomaly(False)

    def teardown_method(self):
        """Clean up after each test."""
        import gc
        gc.collect()
    
    @pytest.mark.timeout(30)  # 30 second timeout
    def test_connection_gradient_flow(self, setup_attention):
        """Test gradient flow through pattern_bundle.connection."""
        layer, params = setup_attention
        
        # Create input tensor with proper dimensions
        x = self.create_input_tensor(params)
        
        # Log initial shapes
        logger.info("\nInitial shapes:")
        logger.info(f"Input x shape: {x.shape}")
        for name, param in layer.named_parameters():
            logger.info(f"Parameter {name} shape: {param.shape}")
        
        # Forward pass
        output = layer(self.reshape_for_layer(x, params))
        logger.info(f"\nOutput shape: {output.shape}")
        
        # Compute loss and backward
        loss = output.abs().mean()
        logger.info(f"Initial loss: {loss.item()}")
        loss.backward()
        
        # Log gradient information
        logger.info("\nGradient information after backward:")
        for name, param in layer.named_parameters():
            if param.grad is not None:
                logger.info(f"\nParameter: {name}")
                logger.info(f"Parameter shape: {param.shape}")
                logger.info(f"Gradient shape: {param.grad.shape}")
                logger.info(f"Gradient norm: {param.grad.norm().item()}")
                logger.info(f"Gradient mean: {param.grad.abs().mean().item()}")
                logger.info(f"Contains NaN: {torch.isnan(param.grad).any().item()}")
                logger.info(f"Contains Inf: {torch.isinf(param.grad).any().item()}")
        
        # Get connection parameter and check its gradients
        connection = layer.quantum_bridge.pattern_bundle.connection
        assert connection.requires_grad, "Connection should require gradients"
        assert connection.grad is not None, "Connection should have gradients"

    @pytest.mark.timeout(30)  # 30 second timeout
    def test_metric_view_gradient_flow(self, setup_attention):
        """Test gradient flow through metric_view."""
        layer, params = setup_attention
        
        # Create input tensor with proper dimensions
        x = self.create_input_tensor(params)
        
        # Get metric view directly from the layer
        metric = layer.metric
        metric.requires_grad_(True)
        
        # Add gradient hook
        gradients = []
        def hook(grad):
            if grad is not None:  # Only append non-None gradients
                gradients.append(grad.detach().clone())
                logger.info(f"\nMetric gradient information:")
                logger.info(f"  Shape: {grad.shape}")
                logger.info(f"  Norm: {grad.norm().item():.6f}")
                logger.info(f"  Mean magnitude: {grad.abs().mean().item():.6f}")
                if grad.is_complex():
                    logger.info(f"  Real mean: {grad.real.mean().item():.6f}")
                    logger.info(f"  Imag mean: {grad.imag.mean().item():.6f}")
            return grad
        metric.register_hook(hook)
        
        # Forward pass
        output = layer(self.reshape_for_layer(x, params))
        
        # For complex tensors, compute loss on both real and imaginary parts
        if torch.is_complex(output):
            loss = output.real.abs().mean() + output.imag.abs().mean()
        else:
            loss = output.abs().mean()
            
        logger.info(f"\nLoss value: {loss.item():.6f}")
        loss.backward()
        
        # Check gradients
        assert len(gradients) > 0, "Metric view should have received gradients"
        
        # Log final gradient analysis
        logger.info("\nFinal gradient analysis:")
        for i, grad in enumerate(gradients):
            logger.info(f"\nGradient {i+1}:")
            logger.info(f"  Shape: {grad.shape}")
            logger.info(f"  Norm: {grad.norm().item():.6f}")
            logger.info(f"  Contains NaN: {torch.isnan(grad).any().item()}")
            logger.info(f"  Contains Inf: {torch.isinf(grad).any().item()}")
            
            # Additional assertions
            assert not torch.isnan(grad).any(), "Gradients contain NaN values"
            assert not torch.isinf(grad).any(), "Gradients contain Inf values"
            assert grad.norm().item() > 0, f"Gradients are zero (norm: {grad.norm().item():.6f})"

    @pytest.mark.timeout(30)  # 30 second timeout
    def test_connection_view_gradient_flow(self, setup_attention):
        """Test gradient flow through connection_view."""
        layer, params = setup_attention
        
        # Create input tensor with proper dimensions
        x = self.create_input_tensor(params)
        logger.info(f"\nInput x dtype: {x.dtype}")

        # Get connection view directly from the layer
        connection = layer.quantum_bridge.pattern_bundle.connection
        logger.info(f"Connection dtype: {connection.dtype}")
        connection.requires_grad_(True)
        connection.retain_grad()  # Ensure gradients are retained

        # Add gradient hook
        gradients = []
        def hook(grad):
            if grad is not None:  # Only append non-None gradients
                gradients.append(grad.detach().clone())
                logger.info(f"\nGradient information in hook:")
                logger.info(f"  Dtype: {grad.dtype}")
                logger.info(f"  Shape: {grad.shape}")
                logger.info(f"  Norm: {grad.norm().item():.6f}")
                logger.info(f"  Mean magnitude: {grad.abs().mean().item():.6f}")
                if grad.is_complex():
                    logger.info(f"  Real mean: {grad.real.mean().item():.6f}")
                    logger.info(f"  Imag mean: {grad.imag.mean().item():.6f}")
            return grad
        connection.register_hook(hook)

        # Forward pass
        output = layer(self.reshape_for_layer(x, params))

        # For complex tensors, compute loss on both real and imaginary parts
        if torch.is_complex(output):
            loss = output.real.abs().mean() + output.imag.abs().mean()
        else:
            loss = output.abs().mean()
            
        logger.info(f"\nLoss value: {loss.item():.6f}")
        loss.backward()

        # Check gradients
        assert len(gradients) > 0, "Connection should have received gradients"
        
        # Log final gradient analysis
        logger.info("\nFinal gradient analysis:")
        for i, grad in enumerate(gradients):
            logger.info(f"\nGradient {i+1}:")
            logger.info(f"  Shape: {grad.shape}")
            logger.info(f"  Norm: {grad.norm().item():.6f}")
            logger.info(f"  Contains NaN: {torch.isnan(grad).any().item()}")
            logger.info(f"  Contains Inf: {torch.isinf(grad).any().item()}")
            
            # Additional assertions
            assert not torch.isnan(grad).any(), "Gradients contain NaN values"
            assert not torch.isinf(grad).any(), "Gradients contain Inf values"
            assert grad.norm().item() > 0, f"Gradients are zero (norm: {grad.norm().item():.6f})"

    @pytest.mark.timeout(30)  # 30 second timeout
    def test_flattened_input_gradient_flow(self, setup_attention):
        """Test gradient flow through flattened input tensor."""
        layer, params = setup_attention
        
        # Create input tensor with proper dimensions
        x = self.create_input_tensor(params)
        logger.info(f"\nInput tensor shape: {x.shape}")

        # Forward pass using original input
        output = layer(self.reshape_for_layer(x, params))  # This will create and store x_flat inside the layer
        logger.info(f"Output tensor shape: {output.shape}")

        # Get x_flat from the layer
        x_flat = layer.x_flat
        logger.info(f"Flattened input shape: {x_flat.shape}")

        # Add gradient hook
        gradients = []
        def hook(grad):
            if grad is not None:  # Only append non-None gradients
                gradients.append(grad.detach().clone())
                logger.info(f"\nGradient information in hook:")
                logger.info(f"  Shape: {grad.shape}")
                logger.info(f"  Norm: {grad.norm().item():.6f}")
                logger.info(f"  Mean magnitude: {grad.abs().mean().item():.6f}")
                if grad.is_complex():
                    logger.info(f"  Real mean: {grad.real.mean().item():.6f}")
                    logger.info(f"  Imag mean: {grad.imag.mean().item():.6f}")
            return grad
        x_flat.register_hook(hook)

        # For complex tensors, compute loss on both real and imaginary parts
        if torch.is_complex(output):
            loss = output.real.abs().mean() + output.imag.abs().mean()
        else:
            loss = output.abs().mean()
            
        logger.info(f"\nLoss value: {loss.item():.6f}")
        loss.backward()

        # Check gradients
        assert len(gradients) > 0, "Flattened input should have received gradients"
        
        # Log final gradient analysis
        logger.info("\nFinal gradient analysis:")
        for i, grad in enumerate(gradients):
            logger.info(f"\nGradient {i+1}:")
            logger.info(f"  Shape: {grad.shape}")
            logger.info(f"  Norm: {grad.norm().item():.6f}")
            logger.info(f"  Contains NaN: {torch.isnan(grad).any().item()}")
            logger.info(f"  Contains Inf: {torch.isinf(grad).any().item()}")
            
            # Additional assertions
            assert not torch.isnan(grad).any(), "Gradients contain NaN values"
            assert not torch.isinf(grad).any(), "Gradients contain Inf values"
            assert grad.norm().item() > 0, f"Gradients are zero (norm: {grad.norm().item():.6f})"

    @pytest.mark.timeout(30)  # 30 second timeout
    def test_end_to_end_gradient_flow(self, setup_attention):
        """Test end-to-end gradient flow through all components."""
        layer, params = setup_attention
        
        # Create input tensor with proper dimensions
        x = self.create_input_tensor(params)

        # Forward pass first to create tensors
        output = layer(self.reshape_for_layer(x, params))

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
                        logger.info(f"\n{name} gradient information:")
                        logger.info(f"  Shape: {grad.shape}")
                        logger.info(f"  Norm: {grad.norm().item():.6f}")
                        logger.info(f"  Mean magnitude: {grad.abs().mean().item():.6f}")
                        if grad.is_complex():
                            logger.info(f"  Real mean: {grad.real.mean().item():.6f}")
                            logger.info(f"  Imag mean: {grad.imag.mean().item():.6f}")
                    return grad
                return hook
            hooks[name] = make_hook(name)
            comp.register_hook(hooks[name])

        # For complex tensors, compute loss on both real and imaginary parts
        if torch.is_complex(output):
            loss = output.real.abs().mean() + output.imag.abs().mean()
        else:
            loss = output.abs().mean()
            
        logger.info(f"\nLoss value: {loss.item():.6f}")
        loss.backward()

        # Check gradients
        logger.info("\nGradient analysis:")
        for name, grads in gradients.items():
            assert len(grads) > 0, f"{name} should have received gradients"
            grad = grads[0]
            logger.info(f"\n{name} gradient analysis:")
            logger.info(f"  Shape: {grad.shape}")
            logger.info(f"  Norm: {grad.norm().item():.6f}")
            logger.info(f"  Contains NaN: {torch.isnan(grad).any().item()}")
            logger.info(f"  Contains Inf: {torch.isinf(grad).any().item()}")
            
            # Additional assertions
            assert not torch.isnan(grad).any(), f"{name} gradients contain NaN values"
            assert not torch.isinf(grad).any(), f"{name} gradients contain Inf values"
            assert grad.norm().item() > 0, f"{name} gradients are zero (norm: {grad.norm().item():.6f})"

    @pytest.mark.timeout(30)  # 30 second timeout
    def test_pattern_metric_gradient_flow(self, setup_attention):
        """Test gradient flow through pattern_metric."""
        layer, params = setup_attention
        
        # Create input tensor with proper dimensions
        x = self.create_input_tensor(params)
        
        # Get pattern_metric directly from the layer
        pattern_metric = layer.pattern_metric
        pattern_metric.requires_grad_(True)
        pattern_metric.retain_grad()  # Ensure gradients are retained
        
        # Add gradient hook
        gradients = []
        def hook(grad):
            if grad is not None:  # Only append non-None gradients
                gradients.append(grad.detach().clone())
                logger.info(f"\nPattern metric gradient information:")
                logger.info(f"  Shape: {grad.shape}")
                logger.info(f"  Norm: {grad.norm().item():.6f}")
                logger.info(f"  Mean magnitude: {grad.abs().mean().item():.6f}")
                if grad.is_complex():
                    logger.info(f"  Real mean: {grad.real.mean().item():.6f}")
                    logger.info(f"  Imag mean: {grad.imag.mean().item():.6f}")
            return grad
        pattern_metric.register_hook(hook)
        
        # Forward pass
        output = layer(self.reshape_for_layer(x, params))
        
        # For complex tensors, compute loss on both real and imaginary parts
        if torch.is_complex(output):
            loss = output.real.abs().mean() + output.imag.abs().mean()
        else:
            loss = output.abs().mean()
            
        logger.info(f"\nLoss value: {loss.item():.6f}")
        loss.backward()
        
        # Check gradients
        assert len(gradients) > 0, "Pattern metric should have received gradients"
        
        # Log final gradient analysis
        logger.info("\nFinal gradient analysis:")
        for i, grad in enumerate(gradients):
            logger.info(f"\nGradient {i+1}:")
            logger.info(f"  Shape: {grad.shape}")
            logger.info(f"  Norm: {grad.norm().item():.6f}")
            logger.info(f"  Contains NaN: {torch.isnan(grad).any().item()}")
            logger.info(f"  Contains Inf: {torch.isinf(grad).any().item()}")
            
            # Additional assertions
            assert not torch.isnan(grad).any(), "Gradients contain NaN values"
            assert not torch.isinf(grad).any(), "Gradients contain Inf values"
            assert grad.norm().item() > 0, f"Gradients are zero (norm: {grad.norm().item():.6f})"

    @pytest.mark.timeout(30)  # 30 second timeout
    def test_base_metric_gradient_flow(self, setup_attention):
        """Test gradient flow through base_metric."""
        layer, params = setup_attention
        
        # Create input tensor with proper dimensions
        x = self.create_input_tensor(params)
        
        # Get base_metric directly from the layer
        base_metric = layer.base_metric
        base_metric.requires_grad_(True)
        base_metric.retain_grad()  # Ensure gradients are retained
        
        # Add gradient hook
        gradients = []
        def hook(grad):
            if grad is not None:  # Only append non-None gradients
                gradients.append(grad.detach().clone())
                logger.info(f"\nBase metric gradient information:")
                logger.info(f"  Shape: {grad.shape}")
                logger.info(f"  Norm: {grad.norm().item():.6f}")
                logger.info(f"  Mean magnitude: {grad.abs().mean().item():.6f}")
                if grad.is_complex():
                    logger.info(f"  Real mean: {grad.real.mean().item():.6f}")
                    logger.info(f"  Imag mean: {grad.imag.mean().item():.6f}")
            return grad
        base_metric.register_hook(hook)
        
        # Forward pass
        output = layer(self.reshape_for_layer(x, params))
        
        # For complex tensors, compute loss on both real and imaginary parts
        if torch.is_complex(output):
            loss = output.real.abs().mean() + output.imag.abs().mean()
        else:
            loss = output.abs().mean()
            
        logger.info(f"\nLoss value: {loss.item():.6f}")
        loss.backward()
        
        # Check gradients
        assert len(gradients) > 0, "Base metric should have received gradients"
        
        # Log final gradient analysis
        logger.info("\nFinal gradient analysis:")
        for i, grad in enumerate(gradients):
            logger.info(f"\nGradient {i+1}:")
            logger.info(f"  Shape: {grad.shape}")
            logger.info(f"  Norm: {grad.norm().item():.6f}")
            logger.info(f"  Contains NaN: {torch.isnan(grad).any().item()}")
            logger.info(f"  Contains Inf: {torch.isinf(grad).any().item()}")
            
            # Additional assertions
            assert not torch.isnan(grad).any(), "Gradients contain NaN values"
            assert not torch.isinf(grad).any(), "Gradients contain Inf values"
            assert grad.norm().item() > 0, f"Gradients are zero (norm: {grad.norm().item():.6f})"

    @pytest.mark.timeout(30)  # 30 second timeout
    def test_combined_metric_gradient_flow(self, setup_attention):
        """Test gradient flow through combined_metric."""
        layer, params = setup_attention
        
        # Create input tensor with proper dimensions
        x = self.create_input_tensor(params)
        
        # Get combined_metric directly from the layer
        combined_metric = layer.combined_metric
        combined_metric.requires_grad_(True)
        combined_metric.retain_grad()  # Ensure gradients are retained
        
        # Add gradient hook
        gradients = []
        def hook(grad):
            if grad is not None:  # Only append non-None gradients
                gradients.append(grad.detach().clone())
                logger.info(f"\nCombined metric gradient information:")
                logger.info(f"  Shape: {grad.shape}")
                logger.info(f"  Norm: {grad.norm().item():.6f}")
                logger.info(f"  Mean magnitude: {grad.abs().mean().item():.6f}")
                if grad.is_complex():
                    logger.info(f"  Real mean: {grad.real.mean().item():.6f}")
                    logger.info(f"  Imag mean: {grad.imag.mean().item():.6f}")
            return grad
        combined_metric.register_hook(hook)
        
        # Forward pass
        output = layer(self.reshape_for_layer(x, params))
        
        # For complex tensors, compute loss on both real and imaginary parts
        if torch.is_complex(output):
            loss = output.real.abs().mean() + output.imag.abs().mean()
        else:
            loss = output.abs().mean()
            
        logger.info(f"\nLoss value: {loss.item():.6f}")
        loss.backward()
        
        # Check gradients
        assert len(gradients) > 0, "Combined metric should have received gradients"
        
        # Log final gradient analysis
        logger.info("\nFinal gradient analysis:")
        for i, grad in enumerate(gradients):
            logger.info(f"\nGradient {i+1}:")
            logger.info(f"  Shape: {grad.shape}")
            logger.info(f"  Norm: {grad.norm().item():.6f}")
            logger.info(f"  Contains NaN: {torch.isnan(grad).any().item()}")
            logger.info(f"  Contains Inf: {torch.isinf(grad).any().item()}")
            
            # Additional assertions
            assert not torch.isnan(grad).any(), "Gradients contain NaN values"
            assert not torch.isinf(grad).any(), "Gradients contain Inf values"
            assert grad.norm().item() > 0, f"Gradients are zero (norm: {grad.norm().item():.6f})"

    @pytest.mark.timeout(30)  # 30 second timeout
    def test_quantum_bridge_gradient_diagnostic(self, setup_attention):
        """Detailed diagnostic test for gradient flow through quantum bridge."""
        logger.info("\n=== Starting Quantum Bridge Gradient Diagnostic ===")
        
        # Create test input
        layer, params = setup_attention
        x = self.create_input_tensor(params)
        logger.info(f"\nInput shape: {x.shape}")
        
        # Track intermediate tensors
        intermediate_tensors = {}
        computation_steps = []
        
        def save_tensor(name: str, tensor: torch.Tensor, step_info: str = ""):
            """Enhanced tensor tracking with computation step info."""
            if tensor.requires_grad:
                tensor.retain_grad()
                intermediate_tensors[name] = tensor
                computation_steps.append(f"Step: {step_info}")
                
                logger.info(f"\nTracking tensor: {name}")
                logger.info(f"Step info: {step_info}")
                logger.info(f"Shape: {tensor.shape}")
                logger.info(f"Requires grad: {tensor.requires_grad}")
                logger.info(f"Is complex: {tensor.is_complex()}")
                if tensor.is_complex():
                    logger.info(f"Complex stats:")
                    logger.info(f"  Magnitude mean: {tensor.abs().mean().item():.6f}")
                    logger.info(f"  Real mean: {tensor.real.mean().item():.6f}")
                    logger.info(f"  Imag mean: {tensor.imag.mean().item():.6f}")
                
                def hook(grad):
                    if grad is not None:
                        logger.info(f"\nGradient for {name} (Step: {step_info}):")
                        logger.info(f"  Shape: {grad.shape}")
                        if grad.is_complex():
                            grad_abs = grad.abs()
                            logger.info(f"  Complex Gradient stats:")
                            logger.info(f"    Magnitude norm: {torch.norm(grad_abs).item():.6f}")
                            logger.info(f"    Real mean: {grad.real.mean().item():.6f}")
                            logger.info(f"    Imag mean: {grad.imag.mean().item():.6f}")
                            logger.info(f"    Max magnitude: {grad_abs.max().item():.6f}")
                            logger.info(f"    Min magnitude: {grad_abs.min().item():.6f}")
                        else:
                            logger.info(f"  Gradient stats:")
                            logger.info(f"    Norm: {torch.norm(grad).item():.6f}")
                            logger.info(f"    Mean: {grad.mean().item():.6f}")
                            logger.info(f"    Max: {grad.max().item():.6f}")
                            logger.info(f"    Min: {grad.min().item():.6f}")
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
        x_flat = self.reshape_for_layer(x, params).reshape(-1, params["hidden_dim"])
        save_tensor("x_flat", x_flat, "Flattened input")
        
        # Track intermediate quantum states
        logger.info("\n=== Starting Forward Pass ===")
        output = layer(self.reshape_for_layer(x, params))
        save_tensor("output", output, "Final output")
        
        # Compute loss and backward
        logger.info("\n=== Starting Backward Pass ===")
        loss = output.abs().sum()
        logger.info(f"Loss value: {loss.item():.6f}")
        loss.backward()
        
        # Log gradient flow analysis
        logger.info("\n=== Gradient Flow Analysis ===")
        logger.info("=" * 50)
        
        # Check each tracked tensor
        for name, tensor in intermediate_tensors.items():
            logger.info(f"\nAnalyzing tensor: {name}")
            logger.info(f"  Shape: {tensor.shape}")
            logger.info(f"  Requires grad: {tensor.requires_grad}")
            if hasattr(tensor, 'grad') and tensor.grad is not None:
                grad = tensor.grad
                if grad.is_complex():
                    grad_abs = grad.abs()
                    logger.info(f"  Complex Gradient stats:")
                    logger.info(f"    Magnitude norm: {torch.norm(grad_abs).item():.6f}")
                    logger.info(f"    Real mean: {grad.real.mean().item():.6f}")
                    logger.info(f"    Imag mean: {grad.imag.mean().item():.6f}")
                    logger.info(f"    Max magnitude: {grad_abs.max().item():.6f}")
                    logger.info(f"    Min magnitude: {grad_abs.min().item():.6f}")
                else:
                    logger.info(f"  Gradient stats:")
                    logger.info(f"    Norm: {torch.norm(grad).item():.6f}")
                    logger.info(f"    Mean: {grad.mean().item():.6f}")
                    logger.info(f"    Max: {grad.max().item():.6f}")
                    logger.info(f"    Min: {grad.min().item():.6f}")
            else:
                logger.info("  No gradients")
        
        # Log computation steps
        logger.info("\n=== Computation Steps ===")
        for i, step in enumerate(computation_steps):
            logger.info(f"{i+1}. {step}")
        
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
        
        # Create input tensor with proper dimensions
        x = self.create_input_tensor(params)
        
        # Forward pass
        output = layer(self.reshape_for_layer(x, params))
        
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
                        logger.info(f"\nTile {tile_idx} gradient stats:")
                        logger.info(f"  Shape: {grad_stats['shape']}")
                        logger.info(f"  Norm: {grad_stats['norm']:.6f}")
                        logger.info(f"  Real std: {grad_stats['real_std']:.6f}")
                        logger.info(f"  Imag std: {grad_stats['imag_std']:.6f}")
                        gradients.append((tile_idx, grad.detach().clone()))
                    return grad
                return hook
            
            # Register hooks for query, key, value weights
            tile.query.weight.register_hook(make_hook(i))
            tile.key.weight.register_hook(make_hook(i))
            tile.value.weight.register_hook(make_hook(i))
        
        # Log initial tensor shapes
        logger.info("\nInitial tensor shapes:")
        logger.info(f"Input shape: {x.shape}")
        logger.info(f"Output shape: {output.shape}")
        
        # Log tile parameters
        for i, tile in enumerate(layer.tiles):
            logger.info(f"\nTile {i} parameters:")
            logger.info(f"  Query weight shape: {tile.query.weight.shape}")
            logger.info(f"  Key weight shape: {tile.key.weight.shape}")
            logger.info(f"  Value weight shape: {tile.value.weight.shape}")
        
        # Compute loss using complex tensor operations
        loss = output.abs().pow(2).sum()  # Use sum() instead of mean() for stronger gradients
        logger.info(f"\nInitial loss: {loss.item():.6f}")
        loss.backward()
        
        # Check gradients
        assert len(gradients) > 0, "Should have received gradients for tile projections"
        
        # Log final gradient analysis
        logger.info("\nGradient analysis summary:")
        for tile_idx, grad in gradients:
            grad_norm = grad.abs().mean().item()
            logger.info(f"\nTile {tile_idx}:")
            logger.info(f"  Gradient norm: {grad_norm:.6f}")
            logger.info(f"  Contains NaN: {torch.isnan(grad).any().item()}")
            logger.info(f"  Contains Inf: {torch.isinf(grad).any().item()}")
            
            # Additional assertions
            assert not torch.isnan(grad).any(), f"Tile {tile_idx} gradients contain NaN values"
            assert not torch.isinf(grad).any(), f"Tile {tile_idx} gradients contain Inf values"
            assert grad_norm > 0, f"Tile {tile_idx} gradients are zero (norm: {grad_norm:.6f})"

    @pytest.mark.timeout(30)  # 30 second timeout
    def test_quantum_bridge_pattern_bundle_metric_flow(self, setup_attention):
        """Test gradient flow through quantum_bridge.pattern_bundle.metric."""
        layer, params = setup_attention
        
        # Create input tensor with proper dimensions
        x = self.create_input_tensor(params)
        
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
                
                logger.info(f"\nTracking tensor: {name}")
                logger.info(f"Step info: {step_info}")
                logger.info(f"Shape: {tensor.shape}")
                logger.info(f"Requires grad: {tensor.requires_grad}")
                logger.info(f"Is complex: {tensor.is_complex()}")
                if tensor.is_complex():
                    logger.info(f"Complex stats:")
                    logger.info(f"  Magnitude mean: {tensor.abs().mean().item():.6f}")
                    logger.info(f"  Real mean: {tensor.real.mean().item():.6f}")
                    logger.info(f"  Imag mean: {tensor.imag.mean().item():.6f}")
                
                def hook(grad):
                    if grad is not None:
                        logger.info(f"\nGradient for {name} (Step: {step_info}):")
                        logger.info(f"  Shape: {grad.shape}")
                        if grad.is_complex():
                            grad_abs = grad.abs()
                            logger.info(f"  Complex Gradient stats:")
                            logger.info(f"    Magnitude norm: {torch.norm(grad_abs).item():.6f}")
                            logger.info(f"    Real mean: {grad.real.mean().item():.6f}")
                            logger.info(f"    Imag mean: {grad.imag.mean().item():.6f}")
                            logger.info(f"    Max magnitude: {grad_abs.max().item():.6f}")
                            logger.info(f"    Min magnitude: {grad_abs.min().item():.6f}")
                        else:
                            logger.info(f"  Gradient stats:")
                            logger.info(f"    Norm: {torch.norm(grad).item():.6f}")
                            logger.info(f"    Mean: {grad.mean().item():.6f}")
                            logger.info(f"    Max: {grad.max().item():.6f}")
                            logger.info(f"    Min: {grad.min().item():.6f}")
                        return grad
                    return grad
                
                tensor.register_hook(hook)
        
        # Track initial tensors
        save_tensor("input", x, "Initial input tensor")
        save_tensor("pattern_bundle.metric", pattern_bundle_metric, "Pattern bundle metric parameter")
        
        # Forward pass
        output = layer(self.reshape_for_layer(x, params))
        save_tensor("output", output, "Final output")
        
        # Use the same loss computation as the integration test
        loss = output.abs().pow(2).sum()
        logger.info(f"\nLoss value: {loss.item():.6f}")
        loss.backward()
        
        # Log gradient flow analysis
        logger.info("\n=== Gradient Flow Analysis ===")
        logger.info("=" * 50)
        
        # Check each tracked tensor
        for name, tensor in intermediate_tensors.items():
            logger.info(f"\nAnalyzing tensor: {name}")
            logger.info(f"  Shape: {tensor.shape}")
            logger.info(f"  Requires grad: {tensor.requires_grad}")
            if hasattr(tensor, 'grad') and tensor.grad is not None:
                grad = tensor.grad
                if grad.is_complex():
                    grad_abs = grad.abs()
                    logger.info(f"  Complex Gradient stats:")
                    logger.info(f"    Magnitude norm: {torch.norm(grad_abs).item():.6f}")
                    logger.info(f"    Real mean: {grad.real.mean().item():.6f}")
                    logger.info(f"    Imag mean: {grad.imag.mean().item():.6f}")
                    logger.info(f"    Max magnitude: {grad_abs.max().item():.6f}")
                    logger.info(f"    Min magnitude: {grad_abs.min().item():.6f}")
                else:
                    logger.info(f"  Gradient stats:")
                    logger.info(f"    Norm: {torch.norm(grad).item():.6f}")
                    logger.info(f"    Mean: {grad.mean().item():.6f}")
                    logger.info(f"    Max: {grad.max().item():.6f}")
                    logger.info(f"    Min: {grad.min().item():.6f}")
            else:
                logger.info("  No gradients")
        
        # Log computation steps
        logger.info("\n=== Computation Steps ===")
        for i, step in enumerate(computation_steps):
            logger.info(f"{i+1}. {step}")
        
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
        
        # Create input tensor with proper dimensions
        x = self.create_input_tensor(params)
        
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
                logger.info(f"Metric factors gradient shape in hook: {grad.shape}")
                logger.info(f"Metric factors gradient norm in hook: {grad.norm().item()}")
            return grad
        metric_factors.register_hook(hook)
        
        # Forward pass
        output = layer(self.reshape_for_layer(x, params))
        
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
        
        # Create input tensor with proper dimensions
        x = self.create_input_tensor(params)
        
        # Get the original connection parameter
        connection = layer.quantum_bridge.pattern_bundle.connection
        connection.requires_grad_(True)
        connection.retain_grad()
        
        # Track gradients for the connection parameter
        connection_grads = []
        def connection_hook(grad):
            if grad is not None:
                connection_grads.append(grad.detach().clone())
                logger.info(f"\nConnection gradient stats:")
                logger.info(f"Shape: {grad.shape}")
                logger.info(f"Norm: {grad.norm().item():.6f}")
                logger.info(f"Mean: {grad.mean().item():.6f}")
            return grad
        connection.register_hook(connection_hook)
        
        # Forward pass
        output = layer(self.reshape_for_layer(x, params))
        
        # Compute loss that ensures connection is used
        loss = output.abs().pow(2).sum()  # Use squared loss for stronger gradients
        logger.info(f"\nLoss value: {loss.item():.6f}")
        loss.backward()
        
        # Check connection gradients
        assert len(connection_grads) > 0, "Connection should have received gradients"
        grad = connection_grads[0]
        grad_norm = grad.abs().mean().item()
        logger.info(f"\nConnection gradient norm: {grad_norm:.6f}")
        assert grad_norm > 0, f"Connection gradients are zero (norm: {grad_norm:.6f})"
        
        # Verify gradient properties
        assert torch.isfinite(grad).all(), "Connection gradients contain inf/nan values"
        assert grad.shape == connection.shape, f"Gradient shape {grad.shape} doesn't match parameter shape {connection.shape}"

    @pytest.mark.timeout(30)  # 30 second timeout
    def test_energy_conservation(self, setup_attention):
        """Test energy conservation during gradient flow."""
        layer, params = setup_attention
        
        # Create input tensor with proper dimensions
        x = self.create_input_tensor(params)
        
        # Forward pass
        output = layer(self.reshape_for_layer(x, params))
        
        # Compute initial energy
        def compute_energy(tensor):
            """Compute energy of a complex tensor."""
            return torch.sum(tensor.real ** 2 + tensor.imag ** 2)
        
        initial_energy = compute_energy(x)
        final_energy = compute_energy(output)
        
        # Log energy values
        logger.info(f"\nEnergy conservation test:")
        logger.info(f"Initial energy: {initial_energy.item():.6f}")
        logger.info(f"Final energy: {final_energy.item():.6f}")
        logger.info(f"Energy difference: {abs(final_energy - initial_energy).item():.6f}")
        logger.info(f"Relative energy difference: {abs(final_energy - initial_energy).item() / initial_energy.item():.6f}")
        
        # Check energy conservation with tolerance
        assert torch.allclose(
            initial_energy, final_energy,
            rtol=1e-2,  # 1% relative tolerance
            atol=1e-2   # Small absolute tolerance
        ), f"Energy not conserved: initial={initial_energy.item():.6f}, final={final_energy.item():.6f}"
        
        # Backward pass should also conserve energy
        loss = output.abs().mean()
        loss.backward()
        
        # Check gradient energy conservation
        if x.grad is not None:
            grad_energy = compute_energy(x.grad)
            logger.info(f"\nGradient energy analysis:")
            logger.info(f"Gradient energy: {grad_energy.item():.6f}")
            logger.info(f"Gradient norm: {x.grad.norm().item():.6f}")
            assert torch.isfinite(grad_energy), "Gradient energy should be finite"
            assert grad_energy > 0, "Gradient energy should be positive"

    @pytest.mark.timeout(30)  # 30 second timeout
    def test_dtype_consistency(self, setup_attention):
        """Test dtype consistency throughout the network."""
        layer, params = setup_attention
        
        # Create both real and complex input tensors
        real_input = torch.randn(
            params["batch_size"], 
            params["num_heads"], 
            params["seq_length"], 
            params["hidden_dim"] // params["num_heads"]
        )
        x = self.create_input_tensor(params)  # This will be complex input
        
        # Check real input
        real_input.requires_grad_(True)
        try:
            output_real = layer(self.reshape_for_layer(real_input, params))
            assert output_real.dtype == torch.complex64, "Output should be complex64 regardless of input type"
        except RuntimeError as e:
            if "must have the same dtype" in str(e):
                logger.info("Layer requires complex input")
            else:
                raise e
        
        # Check complex input
        output_complex = layer(self.reshape_for_layer(x, params))
        assert output_complex.dtype == torch.complex64, "Output should be complex64"
        
        # Check parameter dtypes
        for name, param in layer.named_parameters():
            logger.info(f"\nParameter {name}:")
            logger.info(f"  dtype: {param.dtype}")
            assert param.dtype in [torch.complex64, torch.float32], \
                f"Parameter {name} has unexpected dtype {param.dtype}"
            
        # Test gradient dtypes
        loss = output_complex.abs().mean()
        loss.backward()
        
        for name, param in layer.named_parameters():
            if param.grad is not None:
                logger.info(f"\nGradient for {name}:")
                logger.info(f"  dtype: {param.grad.dtype}")
                assert param.grad.dtype == param.dtype, \
                    f"Gradient dtype mismatch for {name}: param {param.dtype} vs grad {param.grad.dtype}"

    @pytest.mark.timeout(30)  # 30 second timeout
    def test_geometric_phases(self, setup_attention):
        """Test quantum geometric phases and their gradients."""
        layer, params = setup_attention
        
        # Create input tensor with proper dimensions
        x = self.create_input_tensor(params)
    
        # Get quantum bridge and its components
        quantum_bridge = layer.quantum_bridge
        pattern_bundle = quantum_bridge.pattern_bundle
        riemannian_framework = pattern_bundle.riemannian_framework
        geometric_flow = pattern_bundle.geometric_flow
    
        # Debug log manifold dimensions
        logger.info("\nManifold dimension debug:")
        logger.info(f"Layer manifold_dim: {params['manifold_dim']}")
        logger.info(f"Pattern bundle manifold_dim: {pattern_bundle.manifold_dim if hasattr(pattern_bundle, 'manifold_dim') else 'Not set'}")
        logger.info(f"Riemannian framework manifold_dim: {riemannian_framework.manifold_dim if hasattr(riemannian_framework, 'manifold_dim') else 'Not set'}")
    
        # Ensure manifold dimensions match
        if hasattr(pattern_bundle, 'manifold_dim'):
            pattern_bundle.manifold_dim = params['manifold_dim']
        if hasattr(riemannian_framework, 'manifold_dim'):
            riemannian_framework.manifold_dim = params['manifold_dim']
            
        # Reinitialize geometric flow with correct dimensions
        geometric_flow = RiemannianFlow(
            manifold_dim=params['manifold_dim'],
            hidden_dim=params['manifold_dim'],  # Use manifold_dim instead of hidden_dim
            num_layers=3,
            dt=0.1,
            stability_threshold=1e-6,
            dtype=layer.dtype,
            device=x.device
        )
        pattern_bundle.geometric_flow = geometric_flow

        # Ensure all parameters in quantum bridge require gradients
        for name, module in quantum_bridge.named_children():
            if isinstance(module, torch.nn.Module):
                for param in module.parameters():
                    param.requires_grad_(True)
                    param.retain_grad()
    
                    # Add gradient hook to each parameter
                    def param_hook(grad):
                        if grad is not None:
                            # Scale gradient to prevent explosion
                            grad = grad / (grad.norm() + 1e-8)
                            return grad
                        return grad
                    param.register_hook(param_hook)
    
        # Ensure all parameters in riemannian_framework require gradients
        for param in riemannian_framework.parameters():
            param.requires_grad_(True)
            param.retain_grad()
    
        # Add gradient hook to riemannian connection coefficients
        def riemannian_hook(grad):
            if grad is not None:
                # Scale gradient to prevent explosion
                grad = grad / (grad.norm() + 1e-8)
                return grad
            return grad
        riemannian_framework.connection_coeffs.register_hook(riemannian_hook)
    
        # Forward pass
        output = layer(self.reshape_for_layer(x, params))
    
        # Test output properties
        assert output.dtype == layer.dtype, "Should maintain complex dtype"
        assert not torch.isnan(output).any(), "Output should not contain NaN values"
        assert not torch.isinf(output).any(), "Output should not contain Inf values"
    
        # Project points to manifold dimension
        points = self.reshape_for_layer(x, params).reshape(-1, params["hidden_dim"])
    
        # Debug log shapes before projection
        logger.info("\nShape debug before projection:")
        logger.info(f"points shape: {points.shape}")
        logger.info(f"hidden_dim: {params['hidden_dim']}")
        logger.info(f"manifold_dim: {params['manifold_dim']}")
    
        # Project points using manifold projection
        points_proj = layer.manifold_proj(points)
    
        # Debug log shapes after projection
        logger.info("\nShape debug after projection:")
        logger.info(f"points_proj shape: {points_proj.shape}")
    
        # Keep the original batch size from points_proj
        batch_size = points_proj.shape[0]
        manifold_dim = params['manifold_dim']  # Use the manifold_dim from params
    
        # Ensure points_proj has the correct shape [batch_size, manifold_dim]
        assert points_proj.shape[1] == manifold_dim, f"Expected manifold_dim {manifold_dim}, got {points_proj.shape[1]}"
        logger.info(f"points_proj final shape: {points_proj.shape}")
    
        # Compute metric tensor using geometric_flow's compute_metric
        flow_metric = geometric_flow.compute_metric(points_proj)
        
        # Test metric tensor properties
        assert flow_metric.shape == (batch_size, manifold_dim, manifold_dim), f"Expected shape {(batch_size, manifold_dim, manifold_dim)}, got {flow_metric.shape}"
        assert flow_metric.dtype == layer.dtype, "Should maintain complex dtype"
        assert not torch.isnan(flow_metric).any(), "Metric tensor should not contain NaN values"
        assert not torch.isinf(flow_metric).any(), "Metric tensor should not contain Inf values"
        
        # Test Hermitian property
        hermitian_diff = torch.norm(flow_metric - flow_metric.transpose(-2, -1).conj())
        assert hermitian_diff < 1e-6, f"Metric tensor should be Hermitian, got difference norm {hermitian_diff}"
        
        # Test positive definiteness
        eigenvals = torch.linalg.eigvalsh(flow_metric)
        assert (eigenvals > 0).all(), "Metric tensor should be positive definite"
        
        # Compute loss using the metric tensor
        loss = flow_metric.abs().sum()
        
        # Backward pass
        loss.backward()
        
        # Test gradient properties
        for name, param in geometric_flow.named_parameters():
            assert param.grad is not None, f"Parameter {name} should have gradient"
            assert not torch.isnan(param.grad).any(), f"Parameter {name} gradient should not contain NaN values"
            assert not torch.isinf(param.grad).any(), f"Parameter {name} gradient should not contain Inf values"

    @pytest.mark.timeout(30)  # 30 second timeout
    def test_manifold_curvature(self, setup_attention):
        """Test attention manifold curvature properties."""
        layer, params = setup_attention
        x = complex_randn(params["batch_size"], params["seq_length"], params["hidden_dim"])
        x.requires_grad_(True)
        
        # Prepare attention state
        state = layer.prepare_attention_state(x)
        
        # Compute metric tensor
        metric = layer.compute_metric_tensor(state)
        
        # Test metric tensor properties
        assert metric.shape[-2:] == (params["manifold_dim"], params["manifold_dim"]), "Metric tensor should have manifold dimensions"
        assert torch.allclose(
            metric, metric.transpose(-1, -2).conj()
        ), "Metric tensor should be Hermitian"
        assert not torch.isnan(metric).any(), "Metric tensor should not contain NaN values"
        assert not torch.isinf(metric).any(), "Metric tensor should not contain Inf values"
        
        # Test positive definiteness (using real part for eigenvalues)
        eigenvalues = torch.linalg.eigvalsh(metric.real)
        assert torch.all(eigenvalues > -1e-6), "Metric tensor should be positive semi-definite"

    @pytest.mark.timeout(30)  # 30 second timeout
    def test_metric_combination(self, setup_attention):
        """Test metric combination and gradient flow."""
        layer, params = setup_attention
        x = complex_randn(params["batch_size"], params["seq_length"], params["hidden_dim"])
        x.requires_grad_(True)
        
        # Get initial metrics
        base_metric = layer.base_metric
        pattern_metric = layer.pattern_metric
        metric = layer.metric
        
        # Forward pass
        output = layer(x)
        
        # Check metric properties
        assert base_metric.requires_grad, "Base metric should require gradients"
        assert pattern_metric.requires_grad, "Pattern metric should require gradients"
        assert metric.requires_grad, "Metric should require gradients"
        
        # Compute loss and backward
        loss = output.abs().mean()
        loss.backward()
        
        # Check metric gradients
        assert base_metric.grad is not None, "Base metric should receive gradients"
        assert pattern_metric.grad is not None, "Pattern metric should receive gradients"
        assert metric.grad is not None, "Metric should receive gradients"
        assert base_metric.grad.abs().mean() > 0, "Base metric gradients should be non-zero"
        assert pattern_metric.grad.abs().mean() > 0, "Pattern metric gradients should be non-zero"
        assert metric.grad.abs().mean() > 0, "Metric gradients should be non-zero"

    @pytest.mark.timeout(30)  # 30 second timeout
    def test_combined_metric_tensor_flow(self, setup_attention):
        """Test gradient flow through combined metric tensor."""
        layer, params = setup_attention
        x = complex_randn(params["batch_size"], params["seq_length"], params["hidden_dim"])
        x.requires_grad_(True)
        
        # Get combined metric
        combined_metric = layer.combined_metric
        combined_metric.requires_grad_(True)
        combined_metric.retain_grad()
        
        # Forward pass
        output = layer(x)
        
        # Compute loss and backward
        loss = output.abs().mean()
        loss.backward()
        
        # Check combined metric gradients
        assert combined_metric.grad is not None, "Combined metric should receive gradients"
        assert not torch.isnan(combined_metric.grad).any(), "Combined metric gradients contain NaN"
        assert not torch.isinf(combined_metric.grad).any(), "Combined metric gradients contain Inf"
        assert combined_metric.grad.abs().mean() > 0, "Combined metric gradients should be non-zero"

    @pytest.mark.timeout(30)  # 30 second timeout
    def test_energy_conservation_during_normalization(self, setup_attention):
        """Test energy conservation during normalization."""
        layer, params = setup_attention
        x = complex_randn(params["batch_size"], params["seq_length"], params["hidden_dim"])
        x.requires_grad_(True)
        
        # Get initial energy
        initial_energy = torch.sum(x.abs() ** 2)
        
        # Forward pass
        output = layer(x)
        
        # Get final energy
        final_energy = torch.sum(output.abs() ** 2)
        
        # Check energy conservation
        assert torch.allclose(initial_energy, final_energy, rtol=1e-2), \
            f"Energy not conserved: initial={initial_energy.item():.6f}, final={final_energy.item():.6f}"
        
        # Check gradient flow with energy conservation
        loss = output.abs().mean()
        loss.backward()
        
        # Verify gradients respect energy conservation
        for name, param in layer.named_parameters():
            if param.grad is not None:
                # Gradients should be finite and non-zero
                assert torch.isfinite(param.grad).all(), f"Parameter {name} has inf/nan gradients"
                assert param.grad.abs().mean() > 0, f"Parameter {name} has zero gradients"

    @pytest.mark.timeout(30)  # 30 second timeout
    def test_quantum_bridge_energy_conservation(self, setup_attention):
        """Test energy conservation in quantum bridge."""
        layer, params = setup_attention
        x = complex_randn(params["batch_size"], params["seq_length"], params["hidden_dim"])
        x.requires_grad_(True)
        
        # Get quantum bridge
        quantum_bridge = layer.quantum_bridge
        
        # Track initial energy
        initial_energy = torch.sum(x.abs() ** 2)
        
        # Forward pass through quantum bridge components
        x_flat = x.reshape(-1, params["hidden_dim"])
        x_flat.requires_grad_(True)
        
        # Project to manifold space
        manifold_proj = layer.manifold_proj(x_flat)
        manifold_energy = torch.sum(manifold_proj.abs() ** 2)
        
        # Check energy conservation in projection
        assert torch.allclose(initial_energy, manifold_energy, rtol=1e-2), \
            "Energy not conserved in manifold projection"
        
        # Forward pass through quantum bridge
        output = layer(x)
        final_energy = torch.sum(output.abs() ** 2)
        
        # Check overall energy conservation
        assert torch.allclose(initial_energy, final_energy, rtol=1e-2), \
            "Energy not conserved through quantum bridge"
        
        # Check gradient flow with energy conservation
        loss = output.abs().mean()
        loss.backward()
        
        # Verify gradients in quantum bridge components
        bridge_params = [
            ('pattern_bundle.metric', quantum_bridge.pattern_bundle.metric),
            ('pattern_bundle.connection', quantum_bridge.pattern_bundle.connection),
            ('pattern_bundle.riemannian_framework.metric_factors', 
             quantum_bridge.pattern_bundle.riemannian_framework.metric_factors)
        ]
        
        for name, param in bridge_params:
            assert param.grad is not None, f"Parameter {name} should have gradients"
            assert torch.isfinite(param.grad).all(), f"Parameter {name} has inf/nan gradients"
            assert param.grad.abs().mean() > 0, f"Parameter {name} has zero gradients"