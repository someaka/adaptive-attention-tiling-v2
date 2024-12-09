from typing import Protocol, TypeVar

from torch import Tensor

T = TypeVar("T", bound=Tensor)


class AttentionProtocol(Protocol[T]):
    """Protocol for attention mechanisms."""

    def forward(self, query: T, key: T, value: T) -> T: ...
    def prepare_tiles(self, shape: tuple[int, ...]) -> None: ...
