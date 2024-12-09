from typing import Protocol

from torch import Tensor


class BackendProtocol(Protocol):
    """Protocol for backend implementations."""

    def allocate(self, shape: tuple[int, ...]) -> Tensor: ...
    def compute(self, *tensors: Tensor) -> Tensor: ...
