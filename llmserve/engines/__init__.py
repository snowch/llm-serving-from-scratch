"""Engines, in the order the book builds them.

Each one implements the same ``Engine`` interface, so the benchmark harness from chapter 2
measures all of them without modification. That is what makes the scorecard comparable from the
first chapter to the last.
"""

from llmserve.engines.base import Engine
from llmserve.engines.naive import NaiveEngine

__all__ = ["Engine", "NaiveEngine"]
