"""
Language module for the AGI system.

This module provides components for language understanding, generation, and grounding.
"""

from .language_understanding import LanguageUnderstanding
from .language_generation import LanguageGeneration
from .language_grounding import LanguageGrounding

__all__ = ['LanguageUnderstanding', 'LanguageGeneration', 'LanguageGrounding']