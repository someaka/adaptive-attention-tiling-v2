import torch
import time

def test_vulkan_support():
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    
    # Test for Vulkan module and capabilities
    if hasattr(torch, '_C') and hasattr(torch._C, '_has_vulkan'):
        print(f"Vulkan support: {torch._C._has_vulkan()}")
    else:
        print("Vulkan support check not available in this PyTorch build")
    
    # Try to perform a simple operation with Vulkan
    try:
        # Create a test tensor
        x = torch.randn(1000, 1000)
        
        # Time CPU operation
        start = time.time()
        cpu_result = x @ x.t()
        cpu_time = time.time() - start
        print(f"\nCPU matrix multiplication time: {cpu_time:.4f} seconds")
        
        # Try Vulkan if available
        if hasattr(torch, 'vulkan') and hasattr(torch.vulkan, 'is_available') and torch.vulkan.is_available():
            x_vulkan = x.vulkan()
            start = time.time()
            vulkan_result = x_vulkan @ x_vulkan.t()
            vulkan_time = time.time() - start
            print(f"Vulkan matrix multiplication time: {vulkan_time:.4f} seconds")
            
            # Verify results match
            max_diff = (cpu_result - vulkan_result.cpu()).abs().max().item()
            print(f"Maximum difference between CPU and Vulkan results: {max_diff}")
        else:
            print("Vulkan device operations not available")
            
    except Exception as e:
        print(f"Error during Vulkan test: {e}")

if __name__ == "__main__":
    test_vulkan_support()
