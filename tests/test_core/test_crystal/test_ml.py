import torch
import pytest
from src.core.crystal.scale_classes.ml.models import HolographicNet
from src.core.crystal.scale_classes.ml.trainer import HolographicTrainer
from typing import Tuple
import math

@pytest.fixture
def model():
    return HolographicNet(dim=4, hidden_dim=16, n_layers=3)

@pytest.fixture
def trainer(model):
    return HolographicTrainer(model)

def test_model_initialization(model):
    """Test that model initializes with correct properties."""
    assert model.dim == 4
    assert isinstance(model.input_proj, torch.nn.Sequential)
    assert isinstance(model.blocks, torch.nn.ModuleList)
    assert model.dtype == torch.complex64
    assert isinstance(model.log_scale, torch.nn.Parameter)
    assert isinstance(model.correction_weights, torch.nn.Parameter)

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
    corr = model.compute_quantum_corrections(x)
    assert corr.shape == x.shape
    assert corr.dtype == x.dtype
    # Corrections should scale with input
    scale = 2.0
    scaled_corr = model.compute_quantum_corrections(scale * x)
    assert torch.allclose(scaled_corr / corr, torch.tensor(scale, dtype=corr.dtype), rtol=1e-5)

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
    assert all(k in components for k in ['mse', 'scale', 'quantum'])
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
    assert all(k in metrics for k in ['loss', 'mse', 'scale', 'quantum'])
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
    batch_size = 32
    n_warmup = 50
    n_epochs = 300
    
    # First phase: train on basic scaling
    uv_data, ir_data = generate_data(batch_size * 8, include_corrections=False)
    dataset = torch.utils.data.TensorDataset(uv_data, ir_data)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    optimizer = torch.optim.AdamW(trainer.model.parameters(), lr=0.0001, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=0.001, 
        epochs=n_warmup, steps_per_epoch=len(loader),
        pct_start=0.3  # Longer warmup
    )
    
    # Warmup phase
    for epoch in range(n_warmup):
        trainer.train_epoch(loader, optimizer, scheduler)
    
    # Second phase: train with quantum corrections
    uv_data, ir_data = generate_data(batch_size * 8, include_corrections=True)
    dataset = torch.utils.data.TensorDataset(uv_data, ir_data)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    optimizer = torch.optim.AdamW(trainer.model.parameters(), lr=0.001, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=0.005,
        epochs=n_epochs, steps_per_epoch=len(loader),
        pct_start=0.2
    )
    
    metrics = trainer.train_epoch(loader, optimizer)
    initial_loss = metrics['loss']
    final_loss = initial_loss
    best_loss = initial_loss
    patience = 40
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
            
        if final_loss < initial_loss * 0.05:  # Stricter convergence
            break
    
    # Verify loss decreased significantly
    assert final_loss < initial_loss * 0.05, "Network failed to converge on holographic mapping"
    
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

def test_learning_progress(trainer):
    """Test that the model actually learns and validation loss decreases."""
    # Generate a larger dataset for meaningful validation
    batch_size = 64
    n_samples = batch_size * 10
    val_split = 0.2
    
    # Generate dataset with both UV and IR data
    uv_data, ir_data = trainer.generate_training_data(n_samples)
    
    # Split into train/val
    n_val = int(n_samples * val_split)
    indices = torch.randperm(n_samples)
    
    train_idx, val_idx = indices[n_val:], indices[:n_val]
    train_data = torch.utils.data.TensorDataset(uv_data[train_idx], ir_data[train_idx])
    val_data = torch.utils.data.TensorDataset(uv_data[val_idx], ir_data[val_idx])
    
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_data, batch_size=batch_size)
    
    # Training setup
    optimizer = torch.optim.Adam(trainer.model.parameters(), lr=0.001)
    n_epochs = 10
    
    # Record initial losses
    initial_train_metrics = trainer.train_epoch(train_loader, optimizer)
    initial_val_metrics = trainer.validate(val_loader)
    initial_train_loss = initial_train_metrics['loss']
    initial_val_loss = initial_val_metrics['loss']
    
    # Train for several epochs
    train_losses = [initial_train_loss]
    val_losses = [initial_val_loss]
    
    for _ in range(n_epochs):
        train_metrics = trainer.train_epoch(train_loader, optimizer)
        val_metrics = trainer.validate(val_loader)
        train_losses.append(train_metrics['loss'])
        val_losses.append(val_metrics['loss'])
    
    final_train_loss = train_losses[-1]
    final_val_loss = val_losses[-1]
    
    # Verify that training loss decreased significantly (at least 50%)
    assert final_train_loss < initial_train_loss * 0.5, \
        f"Training loss did not decrease significantly: {initial_train_loss:.4f} -> {final_train_loss:.4f}"
    
    # Verify that validation loss also decreased significantly
    assert final_val_loss < initial_val_loss * 0.5, \
        f"Validation loss did not decrease significantly: {initial_val_loss:.4f} -> {final_val_loss:.4f}"
    
    # Check for catastrophic overfitting by ensuring val loss isn't much worse than train loss
    assert final_val_loss < final_train_loss * 2.0, \
        f"Validation loss ({final_val_loss:.4f}) is much higher than training loss ({final_train_loss:.4f})"
    
    # Verify loss decreased monotonically (allowing for small fluctuations)
    for i in range(1, len(train_losses)):
        assert train_losses[i] < train_losses[0] * 0.95, \
            f"Training loss increased significantly at epoch {i}"
        assert val_losses[i] < val_losses[0] * 0.95, \
            f"Validation loss increased significantly at epoch {i}" 

def test_lr_finder(trainer):
    """Test that learning rate finder identifies a reasonable learning rate."""
    # Generate test data
    batch_size = 32
    min_lr = 1e-5
    max_lr = 1e-1
    uv_data, ir_data = trainer.generate_training_data(batch_size)
    
    def find_optimal_lr(
        model: torch.nn.Module,
        x: torch.Tensor,
        y: torch.Tensor,
        min_lr: float = 1e-5,
        max_lr: float = 1e-1,
        num_iters: int = 100
    ) -> float:
        """Find optimal learning rate by testing increasing LRs and analyzing loss curve."""
        # Store initial parameters
        init_state = {
            name: param.clone().detach()
            for name, param in model.named_parameters()
        }
        
        # Log-spaced learning rates
        lrs = torch.logspace(math.log10(min_lr), math.log10(max_lr), num_iters)
        losses = []
        
        for lr in lrs:
            optimizer = torch.optim.Adam(model.parameters(), lr=lr.item())
            optimizer.zero_grad()
            output = model(x)
            loss = trainer.compute_loss(output, y, x)
            losses.append(loss.item())
            loss.backward()
            optimizer.step()
            
            if not torch.isfinite(loss):
                break
        
        # Restore initial parameters
        with torch.no_grad():
            for name, param in model.named_parameters():
                param.copy_(init_state[name])
        
        # Find optimal LR by analyzing loss curve
        smoothed_losses = torch.tensor(losses)
        if len(smoothed_losses) > 1:
            # Look at rate of change of loss
            derivatives = (smoothed_losses[1:] - smoothed_losses[:-1]) / (lrs[1:len(smoothed_losses)] - lrs[:len(smoothed_losses)-1])
            optimal_idx = torch.argmin(derivatives)
            return lrs[optimal_idx].item()
        
        return min_lr  # Default to minimum if no valid range found
    
    # Test the learning rate finder
    optimal_lr = find_optimal_lr(trainer.model, uv_data, ir_data)
    
    # Verify the found learning rate is reasonable
    assert math.isclose(optimal_lr, min_lr, rel_tol=1e-5) or (min_lr <= optimal_lr <= max_lr), \
        f"Found learning rate {optimal_lr} outside expected range [{min_lr}, {max_lr}]"
    
    # Test that the found learning rate actually works
    optimizer = torch.optim.Adam(trainer.model.parameters(), lr=optimal_lr)
    initial_loss = trainer.compute_loss(trainer.model(uv_data), ir_data, uv_data).item()
    
    # Train for a few steps with the found learning rate
    for _ in range(5):
        optimizer.zero_grad()
        pred = trainer.model(uv_data)
        loss = trainer.compute_loss(pred, ir_data, uv_data)
        loss.backward()
        optimizer.step()
    
    final_loss = trainer.compute_loss(trainer.model(uv_data), ir_data, uv_data).item()
    
    # Verify loss decreased with the found learning rate
    assert final_loss < initial_loss, \
        f"Loss did not decrease with found learning rate {optimal_lr}" 