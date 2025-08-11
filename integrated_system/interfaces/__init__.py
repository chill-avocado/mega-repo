"""
Interfaces for the integrated system.

This package defines the interfaces for all components in the integrated system.
"""

from .base import Component, Connectable, Configurable, Executable, Observable, Observer
from .agent import Agent, Tool, Memory
from .ui import UserInterface, UIComponent, Layout
from .os import OSInteraction
from .browser import BrowserAutomation
from .code import CodeExecution
from .cognitive import CognitiveSystem
from .evolution import EvolutionOptimizer
from .integration import Integration
from .nlp import NLPProcessor
from .documentation import Documentation

__all__ = [
    'Component',
    'Connectable',
    'Configurable',
    'Executable',
    'Observable',
    'Observer',
    'Agent',
    'Tool',
    'Memory',
    'UserInterface',
    'UIComponent',
    'Layout',
    'OSInteraction',
    'BrowserAutomation',
    'CodeExecution',
    'CognitiveSystem',
    'EvolutionOptimizer',
    'Integration',
    'NLPProcessor',
    'Documentation',
]