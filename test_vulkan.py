
import torch


def test_vulkan_support():
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")

    # Check if Vulkan is available through device enumeration
    try:
        vulkan_device = torch.device("vulkan")
        print("Vulkan device created successfully")

        # Create a simple tensor and move it to Vulkan
        x = torch.randn(2, 3)
        print("\nOriginal tensor:")
        print(x)

        # Move to Vulkan
        x_vulkan = x.to(vulkan_device)
        print("\nMoved to Vulkan")

        # Move back to CPU and verify
        x_back = x_vulkan.to("cpu")
        print("\nMoved back to CPU:")
        print(x_back)

        # Verify numerical consistency
        print("\nNumerical difference (should be very small):")
        print(torch.max(torch.abs(x - x_back)))

    except (RuntimeError, ValueError) as e:
        print(f"Vulkan not available: {e!s}")


if __name__ == "__main__":
    test_vulkan_support()
