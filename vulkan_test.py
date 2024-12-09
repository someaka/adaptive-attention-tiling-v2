import time

import torch


def run_test(a, b, device, op="matmul", iterations=3):
    if device == "vulkan":
        a_dev = a.to("vulkan")
        b_dev = b.to("vulkan") if b is not None else None
    else:
        a_dev = a
        b_dev = b

    # Warmup
    if op == "matmul":
        c = torch.matmul(a_dev, b_dev)
    elif op == "add":
        c = a_dev + b_dev
    elif op == "mul":
        c = a_dev * b_dev
    elif op == "relu":
        c = torch.relu(a_dev)

    if device == "vulkan":
        c = c.cpu()  # Force sync

    # Timed runs
    start = time.time()
    for _ in range(iterations):
        if op == "matmul":
            c = torch.matmul(a_dev, b_dev)
        elif op == "add":
            c = a_dev + b_dev
        elif op == "mul":
            c = a_dev * b_dev
        elif op == "relu":
            c = torch.relu(a_dev)

        if device == "vulkan":
            c = c.cpu()  # Force sync
    total_time = time.time() - start

    return c, total_time / iterations


def test_vulkan():
    print("PyTorch version:", torch.__version__)
    print("Vulkan available:", torch.is_vulkan_available())

    if not torch.is_vulkan_available():
        print("Vulkan is not available")
        return

    # Test different operations and sizes
    sizes = [512, 1024, 2048]
    operations = ["matmul", "add", "mul", "relu"]
    iterations = 3

    for op in operations:
        print(f"\n=== Testing {op} operation ===")
        for size in sizes:
            print(f"\nMatrix size: {size}x{size}")

            # Create test tensors
            a = torch.randn(size, size)
            b = torch.randn(size, size) if op != "relu" else None

            # CPU test
            c_cpu, cpu_time = run_test(a, b, "cpu", op, iterations)
            print(f"CPU average time: {cpu_time:.4f}s")

            # Vulkan test
            c_vulkan, vulkan_time = run_test(a, b, "vulkan", op, iterations)
            print(f"Vulkan average time: {vulkan_time:.4f}s")
            print(f"Speedup: {cpu_time/vulkan_time:.2f}x")

            # Verify results
            max_diff = torch.max(torch.abs(c_cpu - c_vulkan))
            print(f"Maximum difference: {max_diff}")


if __name__ == "__main__":
    test_vulkan()
