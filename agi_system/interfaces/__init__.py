"""
Interfaces for the AGI system.

This package provides the interfaces for the AGI system components.
"""

from .component import Component, ComponentRegistry
from .memory import Memory, WorkingMemory, LongTermMemory
from .cognition import Attention, MetaCognition, ExecutiveFunction

__all__ = [
    'Component',
    'ComponentRegistry',
    'Memory',
    'WorkingMemory',
    'LongTermMemory',
    'Attention',
    'MetaCognition',
    'ExecutiveFunction'
]