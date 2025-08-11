"""
Learning module for the AGI system.

This module provides components for learning, including supervised learning,
unsupervised learning, and reinforcement learning.
"""

from .supervised_learning import SupervisedLearning
from .unsupervised_learning import UnsupervisedLearning
from .reinforcement_learning import ReinforcementLearning

__all__ = ['SupervisedLearning', 'UnsupervisedLearning', 'ReinforcementLearning']