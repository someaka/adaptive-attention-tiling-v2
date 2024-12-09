"""Generate synthetic data for testing and development."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from collections.abc import Sequence

# Constants for validation and generation
min_sequence_length = 3
min_values = 2
num_regions_default = 4
period_default = 2


class SyntheticDataGenerator:
    """Generate synthetic sequences with known density patterns."""

    def __init__(self: SyntheticDataGenerator) -> None:
        """Initialize generator."""
        self.vocab_size = 1000
        self.num_values = 1000

    def generate_constant_sequence(
        self: SyntheticDataGenerator,
        length: int,
        batch_size: int = 1,
    ) -> torch.Tensor:
        """Generate sequence with constant tokens."""
        token = torch.randint(0, self.vocab_size, (1,))
        return token.repeat(batch_size, length)

    def generate_random_sequence(
        self: SyntheticDataGenerator,
        length: int,
        batch_size: int = 1,
    ) -> torch.Tensor:
        """Generate sequence with random tokens."""
        return torch.randint(0, self.vocab_size, (batch_size, length))

    def generate_pattern_sequence(
        self: SyntheticDataGenerator,
        length: int,
        pattern_length: int,
        batch_size: int = 1,
    ) -> torch.Tensor:
        """Generate sequence with repeating pattern."""
        if length < pattern_length:
            msg = "Length must be >= pattern_length"
            raise ValueError(msg)

        # Generate random pattern
        pattern = torch.randint(0, self.vocab_size, (batch_size, pattern_length))

        # Repeat pattern to fill sequence
        repeats = length // pattern_length
        sequence = pattern.repeat(1, repeats)

        # Add remaining tokens if needed
        remainder = length % pattern_length
        if remainder > 0:
            sequence = torch.cat([sequence, pattern[:, :remainder]], dim=1)

        return sequence

    def generate_density_pattern(
        self: SyntheticDataGenerator,
        sequence_length: int,
    ) -> torch.Tensor:
        """Generate sequence with known density pattern."""
        if sequence_length < min_sequence_length:
            msg = f"Sequence length must be at least {min_sequence_length}"
            raise ValueError(msg)

        # Split sequence into regions
        region_size = sequence_length // num_regions_default

        # Generate values for each region
        sequence = torch.zeros(sequence_length, dtype=torch.long)
        for i in range(num_regions_default):
            start = i * region_size
            end = start + region_size if i < num_regions_default - 1 else sequence_length

            # Alternate between constant and varying regions
            if i % min_values == 0:
                sequence[start:end] = i
            else:
                sequence[start:end] = torch.randint(0, self.num_values, (end - start,))

        return sequence

    def generate_mixed_sequence(
        self: SyntheticDataGenerator,
        sequence_length: int,
    ) -> torch.Tensor:
        """Generate sequence with mixed patterns."""
        if sequence_length < min_sequence_length:
            msg = f"Sequence length must be at least {min_sequence_length}"
            raise ValueError(msg)
        if self.num_values < min_values:
            msg = f"Must have at least {min_values} values"
            raise ValueError(msg)

        # Create sequence with alternating patterns
        sequence = torch.zeros(sequence_length, dtype=torch.long)

        # Split into regions
        region_size = sequence_length // num_regions_default

        for i in range(num_regions_default):
            start = i * region_size
            end = start + region_size if i < num_regions_default - 1 else sequence_length

            # Alternate between patterns
            if i % min_values == 0:
                sequence[start:end] = i
            else:
                sequence[start:end] = torch.randint(0, self.num_values, (end - start,))

        return sequence

    def generate_test_batch(
        self: SyntheticDataGenerator,
        num_sequences: int = 10,
        min_length: int = 50,
        max_length: int = 200,
    ) -> Sequence[torch.Tensor]:
        """Generate batch of test sequences with varying lengths."""
        return [
            self.generate_density_pattern(
                torch.randint(min_length, max_length + 1, (1,)).item(),
            )
            for _ in range(num_sequences)
        ]

    def generate_test_sequence(
        self: SyntheticDataGenerator,
        sequence_length: int = 1000,
        batch_size: int = 4,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Generate test sequence with varying density."""
        # Create sequence with alternating density regions
        sequence = torch.zeros(batch_size, sequence_length, dtype=torch.long)
        density = torch.zeros(batch_size, sequence_length)

        # Split into regions
        region_size = sequence_length // 4
        num_regions = 4

        # Generate regions with different patterns
        regions = [
            (self.generate_constant_sequence, 0.2),  # Low density
            (self.generate_random_sequence, 0.8),  # High density
            (
                lambda length, batch: self.generate_pattern_sequence(length, 4, batch),
                0.5,
            ),  # Medium density
            (self.generate_random_sequence, 0.9),  # Very high density
        ]

        # Fill each region
        for i, (gen_func, density_val) in enumerate(regions):
            start = i * region_size
            end = start + region_size if i < num_regions - 1 else sequence_length
            length = end - start

            sequence[:, start:end] = gen_func(length, batch_size)
            density[:, start:end] = density_val

        return sequence, density

    def generate_edge_cases(
        self: SyntheticDataGenerator,
    ) -> Sequence[tuple[torch.Tensor, torch.Tensor]]:
        """Generate edge cases for testing."""
        cases = []

        # Single token
        seq = torch.tensor([[1]], dtype=torch.long)
        density = torch.tensor([[1.0]])
        cases.append((seq, density))

        # All same token
        seq = torch.ones(1, 10, dtype=torch.long)
        density = torch.zeros(1, 10)
        cases.append((seq, density))

        # Random sequence
        seq = torch.randint(0, self.vocab_size, (1, 20))
        density = torch.ones(1, 20)
        cases.append((seq, density))

        return cases


def generate_periodic_sequence(
    batch_size: int,
    sequence_length: int,
    period: int = period_default,
    num_values: int = min_values,
    device: torch.device | None = None,
) -> torch.Tensor:
    """Generate periodic sequences for testing.

    Args:
        batch_size: Number of sequences in batch
        sequence_length: Length of each sequence
        period: Length of repeating pattern
        num_values: Number of unique values to use
        device: Device to place tensor on

    Returns:
        Tensor of shape (batch_size, sequence_length)

    """
    if num_values < min_values:
        msg = f"Must have at least {min_values} values"
        raise ValueError(msg)

    # Generate base pattern
    pattern = torch.randint(0, num_values, (batch_size, period))

    # Repeat pattern to fill sequence
    repeats = sequence_length // period + 1
    sequence = pattern.repeat(1, repeats)[:, :sequence_length]

    if device is not None:
        sequence = sequence.to(device)

    return sequence


def generate_random_sequence(
    batch_size: int,
    sequence_length: int,
    num_values: int = 1000,
    device: torch.device | None = None,
) -> torch.Tensor:
    """Generate random sequences for testing.

    Args:
        batch_size: Number of sequences in batch
        sequence_length: Length of each sequence
        num_values: Number of unique values to use
        device: Device to place tensor on

    Returns:
        Tensor of shape (batch_size, sequence_length)

    """
    if num_values < min_values:
        msg = f"Must have at least {min_values} values"
        raise ValueError(msg)

    sequence = torch.randint(0, num_values, (batch_size, sequence_length))

    if device is not None:
        sequence = sequence.to(device)

    return sequence


def generate_constant_sequence(
    batch_size: int,
    sequence_length: int,
    value: int = 0,
    device: torch.device | None = None,
) -> torch.Tensor:
    """Generate constant sequences for testing.

    Args:
        batch_size: Number of sequences in batch
        sequence_length: Length of each sequence
        value: Value to repeat
        device: Device to place tensor on

    Returns:
        Tensor of shape (batch_size, sequence_length)

    """
    return torch.full((batch_size, sequence_length), value, device=device)


def generate_mixed_sequence(
    batch_size: int,
    sequence_length: int,
    num_values: int = 1000,
    device: torch.device | None = None,
) -> torch.Tensor:
    """Generate batch of mixed sequences.

    Args:
        batch_size: Number of sequences to generate
        sequence_length: Length of each sequence
        num_values: Number of unique values to use
        device: Device to place tensor on

    Returns:
        Tensor of shape (batch_size, sequence_length)

    """
    if sequence_length < min_sequence_length:
        msg = "Sequence length must be at least 3"
        raise ValueError(msg)
    if num_values < min_values:
        msg = f"Must have at least {min_values} values"
        raise ValueError(msg)

    # Split sequence into thirds
    third = sequence_length // 3

    # Create tensor
    sequence = torch.zeros(batch_size, sequence_length, dtype=torch.long)

    # First third: constant values
    sequence[:, :third] = torch.randint(0, num_values, (batch_size, 1)).expand(-1, third)

    # Middle third: random values
    sequence[:, third : 2 * third] = torch.randint(0, num_values, (batch_size, third))

    # Final third: alternating values
    sequence[:, 2 * third :] = torch.randint(0, num_values, (batch_size, 1)).expand(
        -1,
        sequence_length - 2 * third,
    )

    if device is not None:
        sequence = sequence.to(device)

    return sequence


def generate_batch_with_known_density(
    batch_size: int = 4,
    sequence_length: int = 1000,
    num_values: int = 1000,
    device: torch.device | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Generate a batch of test sequences with known density patterns.

    Args:
        batch_size: Number of sequences in batch
        sequence_length: Length of each sequence
        num_values: Number of unique values to use
        device: Device to place tensors on

    Returns:
        Tuple of (sequences, density) tensors

    """
    if sequence_length < min_sequence_length:
        msg = "Sequence length must be at least 3"
        raise ValueError(msg)
    if num_values < min_values:
        msg = f"Must have at least {min_values} values"
        raise ValueError(msg)

    # Generate sequences with mixed patterns
    sequence = generate_mixed_sequence(
        batch_size=batch_size,
        sequence_length=sequence_length,
        num_values=num_values,
        device=device,
    )

    # Create density tensor marking transitions
    density = torch.zeros_like(sequence, dtype=torch.float)
    density[:, :-1] = (sequence[:, 1:] != sequence[:, :-1]).float()

    return sequence, density
