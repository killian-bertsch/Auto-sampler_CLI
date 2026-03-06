"""
pipeline.py — Chains ProcessingObject instances and runs them in order.

Usage:
    from src.pipeline import Pipeline
    from src.processors.trim import TrimProcessor
    from src.processors.normalize import NormalizeProcessor

    pipeline = Pipeline([TrimProcessor(), NormalizeProcessor()])
    result = pipeline.run(instrument_data)
"""

from __future__ import annotations

from typing import List

from .base import ProcessingObject
from .data import InstrumentData


class Pipeline:
    """
    An ordered sequence of ProcessingObjects.

    Each processor receives the InstrumentData returned by the previous one,
    allowing processors to be mixed, reordered, or skipped freely.
    """

    def __init__(self, processors: List[ProcessingObject]) -> None:
        self.processors = processors

    def run(self, data: InstrumentData) -> InstrumentData:
        """
        Run all processors in order, passing data through each.

        Args:
            data: Initial InstrumentData (from InputModule).

        Returns:
            InstrumentData after all processors have been applied.
        """
        for proc in self.processors:
            print(f"  [{proc.__class__.__name__}] processing {data.meta.instrument_name} ...")
            data = proc.process(data)
        return data

    def __repr__(self) -> str:
        names = " -> ".join(p.__class__.__name__ for p in self.processors)
        return f"Pipeline([{names}])"
