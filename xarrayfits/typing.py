from collections.abc import Iterable, Mapping
from typing import Union

HduType = Union[None, int, Iterable[int], Iterable[str], Mapping[int, str]]
ChunksType = Union[None, Mapping[str, int], Iterable[Mapping[str, int]]]
