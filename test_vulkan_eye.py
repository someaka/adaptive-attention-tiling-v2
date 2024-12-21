import torch
from src.utils.device import get_device, create_identity

def test_device_identity():
    # Get the best available device
    device, is_vulkan = get_device()
    print(f"Using device: {device} (Vulkan available: {is_vulkan})")

    # Create identity matrix
    size = 3
    try:
        # Create identity matrix directly on target device
        device_eye = create_identity(size, device)
        print(f"\nIdentity matrix created on {device}:")
        print(device_eye)
        # Move to CPU for verification
        cpu_identity = device_eye.cpu()
        print("\nVerification on CPU:")
        print(cpu_identity)
        # Verify it's actually an identity matrix
        print("\nIs it a proper identity matrix?", torch.allclose(cpu_identity, torch.eye(size)))
    except Exception as e:
        print(f"\nError during tensor operations: {e}")

if __name__ == "__main__":
    test_device_identity()