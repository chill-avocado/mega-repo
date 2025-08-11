"""
Modules for the AGI system.

This package provides the specialized modules for the AGI system, including neural-symbolic
integration, hierarchical predictive processing, and self-improvement.
"""

from .neural_symbolic import NeuralSymbolicIntegration
from .predictive_processing import HierarchicalPredictiveProcessing
from .self_improvement import CodeImprovement, ModelImprovement, KnowledgeImprovement

__all__ = [
    'NeuralSymbolicIntegration',
    'HierarchicalPredictiveProcessing',
    'CodeImprovement',
    'ModelImprovement',
    'KnowledgeImprovement'
]