"""Analyze information density in sequences."""

from __future__ import annotations

import logging

import torch
import torch.nn.functional as F

logger = logging.getLogger(__name__)


class InformationDensityAnalyzer:
    """Analyze information density in sequences using various metrics."""

    def __init__(
        self: InformationDensityAnalyzer,
        window_size: int = 5,
        smoothing_factor: float = 0.5,
    ) -> None:
        """Initialize analyzer with window size and smoothing parameters."""
        if window_size < 1:
            msg = "Window size must be positive"
            raise ValueError(msg)
        if not 0 <= smoothing_factor <= 1:
            msg = "Smoothing factor must be between 0 and 1"
            raise ValueError(msg)

        self.window_size = window_size
        self.smoothing_factor = smoothing_factor

    def compute_gradient_density(
        self: InformationDensityAnalyzer,
        sequence: torch.Tensor,
    ) -> torch.Tensor:
        """Compute density based on gradient magnitudes.

        Args:
            sequence: Input tensor of shape (batch_size, sequence_length)

        Returns:
            Tensor of shape (batch_size, sequence_length) containing density values

        """
        # Handle single token case
        if sequence.shape[1] == 1:
            return torch.zeros_like(sequence, dtype=torch.float)

        # Convert to float for gradient computation
        x = sequence.float()

        # Compute forward differences
        forward_diff = x[:, 1:] - x[:, :-1]

        # Get non-zero differences (changes in sequence)
        changes = (forward_diff != 0).float()

        # Create density tensor with 1s at change points
        density = torch.zeros_like(x)
        density[:, :-1] = changes

        return density

    def compute_entropy_density(
        self: InformationDensityAnalyzer,
        sequence: torch.Tensor,
        vocab_size: int | None = None,
    ) -> torch.Tensor:
        """Compute density based on local entropy.

        Args:
            sequence: Input tensor of shape (batch_size, sequence_length)
            vocab_size: Size of vocabulary for normalization

        Returns:
            Tensor of shape (batch_size, sequence_length) containing density values

        """
        # Handle single token case
        if sequence.shape[1] == 1:
            return torch.zeros_like(sequence, dtype=torch.float)

        batch_size, seq_length = sequence.shape

        # Determine vocabulary size if not provided
        if vocab_size is None:
            vocab_size = int(torch.max(sequence)) + 1

        # Create sliding windows
        pad_size = self.window_size // 2
        padded = F.pad(sequence, (pad_size, pad_size), mode="replicate")

        # Initialize output
        entropy = torch.zeros(batch_size, seq_length, device=sequence.device)

        # Compute entropy for each position
        for i in range(seq_length):
            # Extract window
            window = padded[:, i : i + self.window_size]

            # Count unique tokens in window
            unique_tokens = torch.unique(window, dim=1)
            entropy[:, i] = unique_tokens.size(1) / self.window_size

        # Scale entropy to [0, 1]
        return entropy.clamp(0, 1)

    def smooth_density(
        self: InformationDensityAnalyzer,
        density: torch.Tensor,
        *,
        use_exponential: bool = True,
    ) -> torch.Tensor:
        """Apply smoothing to density values.

        Args:
            density: Input tensor of shape (batch_size, sequence_length)
            use_exponential: Whether to use exponential smoothing

        Returns:
            Smoothed tensor of same shape as input

        """
        # Handle single token case
        if density.shape[1] == 1:
            return density

        # Apply moving average using convolution
        kernel_size = min(self.window_size, density.size(1))
        if kernel_size % 2 == 0:
            kernel_size -= 1  # Ensure odd kernel size

        kernel = torch.ones(1, 1, kernel_size) / kernel_size
        if density.device.type == "vulkan":
            kernel = kernel.vulkan()

        # Add channel dimension for conv1d
        x = density.unsqueeze(1)
        # pylint: disable=not-callable
        smoothed = F.conv1d(x, kernel, padding=kernel_size // 2)

        # Remove channel dimension
        smoothed = smoothed.squeeze(1)

        if use_exponential:
            # Apply exponential smoothing
            alpha = self.smoothing_factor
            smoothed = alpha * density + (1 - alpha) * smoothed

        return smoothed

    def analyze_density(
        self: InformationDensityAnalyzer,
        sequence: torch.Tensor,
        method: str = "gradient",
        *,
        apply_smoothing: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Analyze sequence density using specified method.

        Args:
            sequence: Input tensor of shape (batch_size, sequence_length)
            method: Density estimation method ('gradient' or 'entropy')
            apply_smoothing: Whether to apply smoothing to density values

        Returns:
            Tuple of (raw_density, smoothed_density) tensors

        """
        if method not in ["gradient", "entropy"]:
            msg = "Method must be 'gradient' or 'entropy'"
            raise ValueError(msg)

        # Compute raw density
        if method == "gradient":
            density = self.compute_gradient_density(sequence)
        else:
            density = self.compute_entropy_density(sequence)

        # Apply smoothing if requested
        smoothed = self.smooth_density(density) if apply_smoothing else density

        return density, smoothed

    def analyze_multi_scale(
        self: InformationDensityAnalyzer,
        sequence: torch.Tensor,
        num_scales: int = 3,
        method: str = "gradient",
    ) -> torch.Tensor:
        """Analyze density at multiple scales.

        Args:
            sequence: Input tensor of shape (batch_size, sequence_length)
            num_scales: Number of scales to analyze
            method: Density estimation method ('gradient' or 'entropy')

        Returns:
            Tensor of shape (batch_size, num_scales, sequence_length)

        """
        batch_size, seq_length = sequence.shape
        densities = torch.zeros(batch_size, num_scales, seq_length)

        for scale in range(num_scales):
            # Compute density at current scale
            density, _ = self.analyze_density(
                sequence,
                method=method,
                apply_smoothing=True,
            )

            densities[:, scale] = density

        return densities
