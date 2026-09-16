from __future__ import annotations

from typing import Protocol

from hhofg.core.types import FrameRecord


class PosedRGBDSequence(Protocol):
    def __len__(self) -> int: ...

    def __getitem__(self, index: int) -> FrameRecord: ...
