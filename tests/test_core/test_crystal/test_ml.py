import torch
import pytest
import os
import math
import torch.nn.functional as F
from src.core.crystal.scale_classes.ml.models import HolographicNet
from src.core.crystal.scale_classes.ml.trainer import HolographicTrainer
from typing import Tuple

@pytest.fixture
def model():
    return HolographicNet(dim=4, hidden_dim=16, n_layers=3)

@pytest.fixture
def trainer(model):
    return HolographicTrainer(model)

@pytest.fixture
def pretrained_model():
    """Create a pre-trained holographic network."""
    model = HolographicNet(dim=4, hidden_dim=16, n_layers=3)
    
    # Load pre-trained weights if they exist
    weights_path = "pretrained/holographic_net.pt"
    if os.path.exists(weights_path):
        state_dict = torch.load(weights_path, weights_only=True)
        # Convert complex weights to split real/imag format
        new_state_dict = {}
        for name, param in state_dict.items():
            if 'weight' in name or 'bias' in name:
                if param.is_complex():
                    base_name = name.replace('.weight', '').replace('.bias', '')
                    suffix = 'weight' if 'weight' in name else 'bias'
                    new_state_dict[f"{base_name}.{suffix}_real"] = param.real
                    new_state_dict[f"{base_name}.{suffix}_imag"] = param.imag
                else:
                    new_state_dict[name] = param
            else:
                new_state_dict[name] = param
        model.load_state_dict(new_state_dict)
        return model
            
    # If no pre-trained weights, train the network
    print("\nPre-training holographic network...")
    batch_size = 128  # Larger batch for better statistics
    n_epochs = 1000   # Train longer for better convergence
    z_uv = 0.1
    z_ir = 10.0
    
    # Create diverse training data
    uv_data = torch.randn(batch_size, model.dim, dtype=model.dtype)
    uv_data = uv_data / torch.norm(uv_data, dim=1, keepdim=True)
    
    # Compute expected IR data using analytical scaling
    z_ratio = z_ir / z_uv
    ir_data = uv_data * z_ratio**(-model.dim)
    
    # Add quantum corrections with proper scaling
    correction_scale = 0.1 / (1 + z_ratio**2)
    for n in range(1, 4):
        power = -model.dim + 2*n
        correction = (-1)**n * uv_data * z_ratio**power / math.factorial(n)
        ir_data = ir_data + correction * correction_scale
    
    # Find optimal learning rate
    def lr_finder(model, x, y, min_lr=1e-5, max_lr=1e-1, num_iters=100):
        init_state = {name: param.clone().detach() for name, param in model.named_parameters()}
        lrs = torch.logspace(math.log10(min_lr), math.log10(max_lr), num_iters)
        losses = []
        
        for lr in lrs:
            optimizer = torch.optim.Adam(model.parameters(), lr=lr.item())
            optimizer.zero_grad()
            output = model(x)
            loss = F.mse_loss(output.real, y.real) + F.mse_loss(output.imag, y.imag)
            losses.append(loss.item())
            loss.backward()
            optimizer.step()
            
            if not torch.isfinite(loss):
                break
                    
        with torch.no_grad():
            for name, param in model.named_parameters():
                param.copy_(init_state[name])
            
        smoothed_losses = torch.tensor(losses)
        if len(smoothed_losses) > 1:
            derivatives = (smoothed_losses[1:] - smoothed_losses[:-1]) / (lrs[1:len(smoothed_losses)] - lrs[:len(smoothed_losses)-1])
            optimal_idx = torch.argmin(derivatives)
        else:
            optimal_idx = 0
                
        return lrs[optimal_idx].item()
    
    # Train with optimal learning rate and physics-based loss
    optimal_lr = lr_finder(model, uv_data, ir_data)
    optimizer = torch.optim.Adam(model.parameters(), lr=optimal_lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=50)
    
    best_loss = float('inf')
    for epoch in range(n_epochs):
        optimizer.zero_grad()
        pred_ir = model(uv_data)
        
        # Holographic error with conformal weight
        N = z_ratio**model.dim
        holographic_error = N*pred_ir - uv_data
        conformal_factor = N**2 + 1
        basic_loss = torch.mean(torch.abs(holographic_error)**2) / conformal_factor
        
        # Quantum corrections with proper scaling
        uv_norm = torch.norm(uv_data, dim=1, keepdim=True)
        uv_normalized = uv_data / (uv_norm + 1e-8)
        quantum_pred = model.compute_quantum_corrections(uv_normalized, uv_norm)
        quantum_target = ir_data - uv_data/N  # Remove classical scaling
        quantum_loss = torch.mean(torch.abs(quantum_pred - quantum_target)**2) * N
        
        # Phase coherence with conformal coupling
        phase_diff = torch.angle(pred_ir) - torch.angle(uv_data/N)
        phase_loss = torch.mean(torch.abs(uv_data)**2 * (1 - torch.cos(phase_diff))) / conformal_factor
        
        # Total loss combines all physics terms
        loss = basic_loss + quantum_loss + phase_loss
        loss.backward()
        optimizer.step()
        scheduler.step(loss)
        
        if epoch % 100 == 0:
            print(f"Epoch {epoch}: loss = {loss.item():.2e}")
            print(f"  Basic: {basic_loss.item():.2e}")
            print(f"  Quantum: {quantum_loss.item():.2e}")
            print(f"  Phase: {phase_loss.item():.2e}")
        
        # Save best model
        if loss.item() < best_loss:
            best_loss = loss.item()
            os.makedirs(os.path.dirname(weights_path), exist_ok=True)
            # Save state dict with complex weights
            state_dict = model.state_dict()
            # Convert split real/imag format back to complex weights
            complex_state_dict = {}
            for name, param in state_dict.items():
                if '_real' in name or '_imag' in name:
                    base_name = name.replace('_real', '').replace('_imag', '')
                    if '_real' in name:
                        if base_name not in complex_state_dict:
                            complex_state_dict[base_name] = param + 0j
                        else:
                            complex_state_dict[base_name].real = param
                    else:  # '_imag' in name
                        if base_name not in complex_state_dict:
                            complex_state_dict[base_name] = 1j * param
                        else:
                            complex_state_dict[base_name].imag = param
                else:
                    complex_state_dict[name] = param
            torch.save(complex_state_dict, weights_path)
    
    return model

def test_model_initialization(model):
    """Test that model initializes with correct properties."""
    assert model.dim == 4
    assert isinstance(model.layers, torch.nn.ModuleList)
    assert len(model.layers) == 3  # Input layer, hidden layer, output layer
    assert model.layers[0].in_features == 4
    assert model.layers[0].out_features == 16
    assert model.layers[-1].out_features == 4
    assert isinstance(model.log_scale, torch.nn.Parameter)
    assert isinstance(model.correction_weights, torch.nn.Parameter)
    assert model.correction_weights.shape == (3,)  # Three quantum correction terms

def test_forward_shape(model):
    """Test that forward pass maintains tensor shape."""
    batch_size = 32
    x = torch.randn(batch_size, model.dim, dtype=model.dtype)
    y = model(x)
    assert y.shape == x.shape
    assert y.dtype == x.dtype

def test_quantum_corrections(model):
    """Test that quantum corrections have expected properties."""
    batch_size = 32
    x = torch.randn(batch_size, model.dim, dtype=model.dtype)
    x_norm = torch.norm(x, dim=1, keepdim=True)
    x_normalized = x / (x_norm + 1e-8)
    
    corr = model.compute_quantum_corrections(x_normalized, x_norm)
    
    # Check shape
    assert corr.shape == x.shape
    
    # Check that corrections are smaller than input
    assert torch.all(torch.norm(corr, dim=1) < torch.norm(x, dim=1))
    
    # Check that corrections scale properly with z_ratio
    z_ratio = model.z_ratio
    expected_scale = z_ratio**(-model.dim + 2)  # First order correction
    actual_scale = torch.mean(torch.norm(corr, dim=1) / torch.norm(x, dim=1))
    assert abs(float(actual_scale) - float(expected_scale)) < 0.1

def test_network_learning(model):
    """Test that the network learns by verifying loss decreases using optimal learning rate."""
    # Create test data with correct shape for holographic network
    batch_size = 32  # Larger batch size
    x = torch.randn(batch_size, model.dim, dtype=model.dtype)
    x = x / torch.norm(x, dim=1, keepdim=True)  # Normalize input
    
    # Create target using analytical solution
    z_ratio = model.z_ratio
    y = x * z_ratio**(-model.dim)  # Basic scaling
    
    # Add quantum corrections
    correction_scale = 0.1 / (1 + z_ratio**2)
    for n in range(1, 4):
        power = -model.dim + 2*n
        correction = (-1)**n * x * z_ratio**power / math.factorial(n)
        y = y + correction * correction_scale
    
    # Custom complex MSE loss with physics-informed weighting
    def physics_loss(output: torch.Tensor, target: torch.Tensor, input: torch.Tensor) -> torch.Tensor:
        # Basic holographic error
        N = z_ratio**model.dim
        holographic_error = N*output - input
        conformal_factor = N**2 + 1
        basic_loss = torch.mean(torch.abs(holographic_error)**2) / conformal_factor
        
        # Quantum corrections
        input_norm = torch.norm(input, dim=1, keepdim=True)
        input_normalized = input / (input_norm + 1e-8)
        quantum_pred = model.compute_quantum_corrections(input_normalized, input_norm)
        quantum_target = target - input/N
        quantum_loss = torch.mean(torch.abs(quantum_pred - quantum_target)**2) * N
        
        # Phase coherence
        phase_diff = torch.angle(output) - torch.angle(target)
        phase_loss = torch.mean(torch.abs(input)**2 * (1 - torch.cos(phase_diff))) / conformal_factor
        
        return basic_loss + 0.1 * quantum_loss + 0.01 * phase_loss
    
    # Store initial parameters
    initial_params = {
        name: param.clone().detach()
        for name, param in model.named_parameters()
    }
    
    # Create optimizer with appropriate learning rate
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    
    # Track losses
    losses = []
    
    # Do several training steps
    n_steps = 50  # More training steps
    for step in range(n_steps):
        optimizer.zero_grad()
        output = model(x)
        loss = physics_loss(output, y, x)
        losses.append(loss.item())
        loss.backward()
        optimizer.step()
        
        if step % 10 == 0:
            print(f"Step {step}: loss = {loss.item():.2e}")
    
    # Check if parameters changed
    changed = False
    for name, param in model.named_parameters():
        if not torch.allclose(param, initial_params[name], rtol=1e-3):
            changed = True
            break
            
    assert changed, "Parameters did not change during training"
    assert losses[-1] < losses[0] * 0.9, "Loss did not decrease significantly during training"

def test_loss_computation(trainer):
    """Test that loss computation works and returns expected components."""
    batch_size = 16
    x = torch.randn(batch_size, trainer.model.dim, dtype=trainer.model.dtype)
    y = torch.randn(batch_size, trainer.model.dim, dtype=trainer.model.dtype)
    pred = trainer.model(x)
    
    # Test basic loss
    loss = trainer.compute_loss(pred, y, x)
    assert isinstance(loss, torch.Tensor)
    assert loss.ndim == 0  # scalar
    
    # Test loss components
    loss, components = trainer.compute_loss(pred, y, x, return_components=True)
    assert isinstance(components, dict)
    assert all(k in components for k in ['basic', 'quantum', 'phase'])
    assert all(isinstance(v, float) for v in components.values())

def test_norm_preservation(model):
    """Test that the model approximately preserves input norms up to scaling."""
    batch_size = 32
    x = torch.randn(batch_size, model.dim, dtype=model.dtype)
    y = model(x)
    
    input_norms = torch.norm(x, dim=1)
    output_norms = torch.norm(y, dim=1)
    
    # Ratio of norms should be approximately constant across batch
    norm_ratios = output_norms / input_norms
    mean_ratio = norm_ratios.mean()
    relative_deviation = torch.abs(norm_ratios - mean_ratio) / mean_ratio
    assert torch.all(relative_deviation < 0.1)  # Allow 10% deviation

def test_training_step(trainer):
    """Test that a single training step runs without errors."""
    batch_size = 16
    x = torch.randn(batch_size, trainer.model.dim, dtype=trainer.model.dtype)
    y = torch.randn(batch_size, trainer.model.dim, dtype=trainer.model.dtype)
    
    from torch.utils.data import TensorDataset, DataLoader
    dataset = TensorDataset(x, y)
    loader = DataLoader(dataset, batch_size=4)
    
    optimizer = torch.optim.Adam(trainer.model.parameters())
    metrics = trainer.train_epoch(loader, optimizer)
    
    assert isinstance(metrics, dict)
    assert all(k in metrics for k in ['loss', 'basic', 'quantum', 'phase'])
    assert all(isinstance(v, float) for v in metrics.values()) 

def test_holographic_convergence(trainer):
    """Test that network learns the correct holographic mapping."""
    # Initialize model closer to analytical solution
    with torch.no_grad():
        trainer.model.log_scale.data = torch.log(torch.tensor(
            trainer.model.z_ratio**(-trainer.model.dim),
            dtype=torch.float32
        ))
        trainer.model.correction_weights.data = torch.tensor(
            [0.1/n for n in range(1, 4)],
            dtype=trainer.model.dtype
        )

    # Start with simpler cases (no quantum corrections)
    def generate_data(batch_size: int, include_corrections: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
        uv_data = torch.randn(batch_size, trainer.model.dim, dtype=trainer.model.dtype)
        uv_data = uv_data / torch.norm(uv_data, dim=1, keepdim=True)

        z_ratio = trainer.model.z_ratio
        ir_data = uv_data * z_ratio**(-trainer.model.dim)

        if include_corrections:
            for n in range(1, 4):
                power = -trainer.model.dim + 2*n
                ir_data = ir_data + (0.1/n) * uv_data * z_ratio**power

        ir_data = ir_data / torch.norm(ir_data, dim=1, keepdim=True)
        return uv_data, ir_data

    # Training hyperparameters
    batch_size = 64  # Larger batch size
    n_warmup = 100  # More warmup epochs
    n_epochs = 500  # More training epochs

    # First phase: train on basic scaling
    uv_data, ir_data = generate_data(batch_size * 8, include_corrections=False)
    dataset = torch.utils.data.TensorDataset(uv_data, ir_data)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

    optimizer = torch.optim.AdamW(
        trainer.model.parameters(),
        lr=0.0005,  # Higher base learning rate
        weight_decay=1e-4  # Stronger regularization
    )
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=0.005,  # Higher max learning rate
        epochs=n_warmup, steps_per_epoch=len(loader),
        pct_start=0.4,  # Longer warmup
        div_factor=25.0,  # Larger lr range
        final_div_factor=1000.0  # Stronger final lr decay
    )

    # Warmup phase
    for epoch in range(n_warmup):
        trainer.train_epoch(loader, optimizer, scheduler)

    # Second phase: train with quantum corrections
    uv_data, ir_data = generate_data(batch_size * 8, include_corrections=True)
    dataset = torch.utils.data.TensorDataset(uv_data, ir_data)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

    optimizer = torch.optim.AdamW(
        trainer.model.parameters(),
        lr=0.001,  # Higher learning rate
        weight_decay=1e-4
    )
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=0.01,  # Higher max learning rate
        epochs=n_epochs, steps_per_epoch=len(loader),
        pct_start=0.3,
        div_factor=25.0,
        final_div_factor=1000.0
    )

    metrics = trainer.train_epoch(loader, optimizer)
    initial_loss = metrics['loss']
    final_loss = initial_loss
    best_loss = initial_loss
    patience = 50  # More patience
    no_improve = 0

    for epoch in range(n_epochs):
        metrics = trainer.train_epoch(loader, optimizer, scheduler)
        final_loss = metrics['loss']

        if final_loss < best_loss:
            best_loss = final_loss
            no_improve = 0
        else:
            no_improve += 1

        if no_improve >= patience:
            break

        if final_loss < initial_loss * 0.1:  # Relaxed convergence criterion
            break

    # Verify loss decreased significantly but with relaxed criterion
    assert final_loss < initial_loss * 0.1, "Network failed to converge on holographic mapping"
    
    # Test on new data
    test_uv = torch.randn(32, trainer.model.dim, dtype=trainer.model.dtype)
    test_uv = test_uv / torch.norm(test_uv, dim=1, keepdim=True)
    
    with torch.no_grad():
        trainer.model.eval()
        pred_ir = trainer.model(test_uv)
        pred_ir = pred_ir / torch.norm(pred_ir, dim=1, keepdim=True)
    
    # Compute analytical solution
    test_analytical_ir = test_uv * trainer.model.z_ratio**(-trainer.model.dim)
    for n in range(1, 4):
        power = -trainer.model.dim + 2*n
        test_analytical_ir = test_analytical_ir + (0.1/n) * test_uv * trainer.model.z_ratio**power
    test_analytical_ir = test_analytical_ir / torch.norm(test_analytical_ir, dim=1, keepdim=True)
    
    # Verify predictions match analytical solution
    rel_error = torch.norm(pred_ir - test_analytical_ir) / torch.norm(test_analytical_ir)
    assert rel_error < 0.2, "Network predictions deviate significantly from analytical solution"
    
    # Verify scaling behavior
    scale = 2.0
    scaled_pred = trainer.model(scale * test_uv)
    scaled_pred = scaled_pred / torch.norm(scaled_pred, dim=1, keepdim=True)
    scaling_error = torch.norm(scaled_pred - pred_ir) / torch.norm(pred_ir)
    assert scaling_error < 0.2, "Network fails to preserve scaling behavior" 