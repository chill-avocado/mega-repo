"""
Memory module for the AGI system.

This module provides components for working memory and long-term memory.
"""

from .default_working_memory import DefaultWorkingMemory
from .default_long_term_memory import DefaultLongTermMemory

__all__ = ['DefaultWorkingMemory', 'DefaultLongTermMemory']