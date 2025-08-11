"""
Reasoning module for the AGI system.

This module provides components for reasoning, including logical reasoning,
causal reasoning, and analogical reasoning.
"""

from .logical_reasoning import LogicalReasoning
from .causal_reasoning import CausalReasoning
from .analogical_reasoning import AnalogicalReasoning

__all__ = ['LogicalReasoning', 'CausalReasoning', 'AnalogicalReasoning']