"""
base.py — Abstract base class for all processing objects.

Every stage in the pipeline (trim, normalize, loop finder, etc.) is a
ProcessingObject subclass. The contract is simple:

    process(data: InstrumentData) -> InstrumentData

Processors may modify `data` in-place and return it, or return a new object.
By convention, always return `data` (even if modified in-place) so that the
pipeline runner can chain calls with:

    data = processor.process(data)
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from .data import InstrumentData


class ProcessingObject(ABC):
    """
    Abstract base class for all audio processing stages.

    Subclasses implement `process()` to transform InstrumentData.
    They should be stateless with respect to instrument data — all
    configuration should be encoded in the processor's __init__ or
    read from data.meta.
    """

    @abstractmethod
    def process(self, data: InstrumentData) -> InstrumentData:
        """
        Apply this processing stage to `data` and return the result.

        Args:
            data: InstrumentData flowing through the pipeline.

        Returns:
            InstrumentData after applying this stage's transformation.
        """
        ...

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"


class PassthroughProcessor(ProcessingObject):
    """No-op processor — useful for testing the pipeline wiring."""

    def process(self, data: InstrumentData) -> InstrumentData:
        return data
