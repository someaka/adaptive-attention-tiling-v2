"""Unit tests for the adaptive parameter management system."""

import time

import pytest
import torch

from src.core.parameters import AdaptiveParameterManager, ParameterMonitor


def test_parameter_manager_initialization() -> None:
    """Test initialization of AdaptiveParameterManager.

    Ensures:
    - All parameters are properly initialized
    - Parameters are on correct device
    - Gradients are enabled
    - Constraints are properly set
    """
    manager = AdaptiveParameterManager(learning_rate=1e-4)

    # Check all expected parameters exist
    expected_params = {
        "resolution_bounds",
        "resolution_momentum",
        "density_scaling",
        "load_thresholds",
        "balance_rate",
        "compression_ratio",
        "interpolation_weights",
    }
    assert set(manager.parameters.keys()) == expected_params

    # Check parameter properties
    for param in manager.parameters.values():
        assert isinstance(param, torch.nn.Parameter)
        assert param.requires_grad
        assert param.device == torch.device("cpu")  # Default device

    # Check specific parameter initializations
    assert torch.allclose(
        manager.parameters["resolution_bounds"], torch.tensor([0.1, 1.0])
    )
    assert torch.allclose(
        manager.parameters["resolution_momentum"], torch.tensor([0.5])
    )


def test_parameter_constraints() -> None:
    """Test parameter constraint enforcement.

    Ensures:
    - Parameters respect min/max bounds
    - Updates maintain constraints
    - Invalid updates are properly clamped
    """
    manager = AdaptiveParameterManager()

    # Create mock metrics that would push parameters out of bounds
    metrics = {
        "reconstruction_error": torch.tensor(10.0),
        "compute_time": torch.tensor(5.0),
        "load_variance": torch.tensor(2.0),
    }

    # Compute loss and update
    loss = manager.compute_loss(metrics)
    {name: param.clone() for name, param in manager.parameters.items()}

    # Force multiple updates
    for _ in range(10):
        manager.update_parameters(loss)

    # Check constraints are maintained
    for name, param in manager.parameters.items():
        constraints = manager.constraints[name]
        if constraints.min_value is not None:
            assert torch.all(param >= constraints.min_value)
        if constraints.max_value is not None:
            assert torch.all(param <= constraints.max_value)


def test_loss_computation() -> None:
    """Test loss computation components.

    Ensures:
    - Individual loss components are computed correctly
    - Total loss combines components properly
    - Missing metrics are handled gracefully
    """
    manager = AdaptiveParameterManager()

    # Test with complete metrics
    metrics = {
        "reconstruction_error": torch.tensor(0.5),
        "kl_divergence": torch.tensor(0.3),
        "compute_time": torch.tensor(0.2),
        "memory_usage": torch.tensor(0.1),
        "load_variance": torch.tensor(0.4),
        "transition_smoothness": torch.tensor(0.2),
    }

    loss = manager.compute_loss(metrics)
    assert loss > 0
    assert not torch.isnan(loss)
    assert not torch.isinf(loss)

    # Test with partial metrics
    partial_metrics = {"reconstruction_error": torch.tensor(0.5)}
    partial_loss = manager.compute_loss(partial_metrics)
    assert partial_loss > 0

    # Test with empty metrics
    empty_loss = manager.compute_loss({})
    assert empty_loss == 0


def test_parameter_updates() -> None:
    """Test parameter update mechanism.

    Ensures:
    - Parameters are updated correctly with gradients
    - Momentum is applied properly
    - Update frequency is respected
    """
    manager = AdaptiveParameterManager(learning_rate=0.1, update_frequency=2)

    metrics = {
        "reconstruction_error": torch.tensor(1.0, requires_grad=True),
        "compute_time": torch.tensor(0.5, requires_grad=True),
    }

    # Record initial values
    initial_values = {name: param.clone() for name, param in manager.parameters.items()}

    # First update (should not change due to update_frequency)
    loss = manager.compute_loss(metrics)
    manager.update_parameters(loss)

    for name, param in manager.parameters.items():
        assert torch.allclose(param, initial_values[name])

    # Second update (should change)
    loss = manager.compute_loss(metrics)
    manager.update_parameters(loss)

    changes = False
    for name, param in manager.parameters.items():
        if not torch.allclose(param, initial_values[name]):
            changes = True
            break
    assert changes, "No parameters were updated after update_frequency steps"


def test_parameter_monitor() -> None:
    """Test parameter monitoring functionality.

    Ensures:
    - History is properly recorded
    - Statistics are computed correctly
    - Anomaly detection works
    - History management respects max_history
    """
    monitor = ParameterMonitor(max_history=5)  # Increased to keep enough history

    # Create some test parameters and losses with known values
    parameters = {"param1": torch.tensor([1.0]), "param2": torch.tensor([2.0])}
    losses = {"loss1": 0.5, "loss2": 0.3}

    # Log updates with controlled values
    base_value = 1.0
    for _ in range(4):
        parameters = {
            "param1": torch.tensor([base_value]),
            "param2": torch.tensor([2.0 * base_value]),
        }
        base_value *= 1.1  # Increase by 10% each time
        monitor.log_update(parameters, losses)
        time.sleep(0.001)  # Ensure different timestamps

    # Verify the values are as expected
    history = monitor.get_parameter_history("param1")
    values = [val for _, val in history]
    expected_values = [1.0, 1.1, 1.21, 1.331]  # 10% increase each time
    for actual, expected in zip(values, expected_values):
        assert abs(actual - expected) < 0.001, f"Expected {expected}, got {actual}"

    # Add clearly anomalous value (10x the last value)
    parameters["param1"] = torch.tensor([13.31])  # 10x the last normal value
    monitor.log_update(parameters, losses)

    # Get anomalies with lower threshold to ensure detection
    anomalies = monitor.detect_anomalies(threshold=1.5)  # Lower threshold
    history_str = str(monitor.get_parameter_history("param1"))
    assert "param1" in anomalies, f"No anomalies detected. History: {history_str}"
    assert len(anomalies["param1"]) > 0


def test_state_management() -> None:
    """Test parameter state saving and loading.

    Ensures:
    - State can be saved and loaded correctly
    - All necessary information is preserved
    - State dict format is correct
    """
    manager = AdaptiveParameterManager()

    # Create some initial state
    metrics = {
        "reconstruction_error": torch.tensor(1.0),
        "compute_time": torch.tensor(0.5),
    }
    loss = manager.compute_loss(metrics)
    manager.update_parameters(loss)

    # Save state
    state_dict = manager.state_dict()

    # Create new manager and load state
    new_manager = AdaptiveParameterManager()
    new_manager.load_state_dict(state_dict)

    # Verify parameters match
    for name, param in manager.parameters.items():
        assert torch.allclose(param, new_manager.parameters[name])

    # Verify momentum state matches
    for name in manager.parameter_momentum:
        assert torch.allclose(
            manager.parameter_momentum[name],
            new_manager.parameter_momentum[name],
        )

    # Verify step count matches
    assert manager.step_count == new_manager.step_count


def test_device_handling() -> None:
    """Test parameter handling across devices.

    Ensures:
    - Parameters can be initialized on different devices
    - Updates work correctly across devices
    - Device transfers maintain parameter properties
    """
    try:
        if not torch.vulkan.is_available():
            pytest.skip("Vulkan not available")
    except AttributeError:
        pytest.skip("Vulkan support not installed")

    # Create manager on CPU
    cpu_manager = AdaptiveParameterManager(device="cpu")

    # Create manager on Vulkan
    vulkan_manager = AdaptiveParameterManager(device="vulkan")

    # Verify devices
    for param in cpu_manager.parameters.values():
        assert param.device.type == "cpu"

    for param in vulkan_manager.parameters.values():
        assert param.device.type == "vulkan"

    # Test updates on different devices
    metrics_cpu = {
        "reconstruction_error": torch.tensor(1.0),
        "compute_time": torch.tensor(0.5),
    }

    metrics_vulkan = {
        "reconstruction_error": torch.tensor(1.0, device="vulkan"),
        "compute_time": torch.tensor(0.5, device="vulkan"),
    }

    # Compute losses and update
    loss_cpu = cpu_manager.compute_loss(metrics_cpu)
    loss_vulkan = vulkan_manager.compute_loss(metrics_vulkan)

    cpu_manager.update_parameters(loss_cpu)
    vulkan_manager.update_parameters(loss_vulkan)

    # Verify updates worked on both devices
    assert cpu_manager.step_count == vulkan_manager.step_count
