# Unified interface for evolution_optimization
from .evolutionoptimizer import EvolutionOptimizer

class UnifiedEvolutionOptimizer(EvolutionOptimizer):
    """Unified interface for evolution_optimization functionality."""
    
    def __init__(self, implementation=None, config=None):
        """
        Initialize the unified interface.
        
        Args:
            implementation: The specific implementation to use.
            config: Optional configuration.
        """
        super().__init__(config)
        self.implementation = implementation
    
    def evolve(self, *args, **kwargs):
        """
        Evolve operation.
        
        Args:
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.
            
        Returns:
            The result of the evolve operation.
        """
        if self.implementation and hasattr(self.implementation, 'evolve'):
            return getattr(self.implementation, 'evolve')(*args, **kwargs)
        return super().evolve(*args, **kwargs)
    
    def evaluate(self, *args, **kwargs):
        """
        Evaluate operation.
        
        Args:
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.
            
        Returns:
            The result of the evaluate operation.
        """
        if self.implementation and hasattr(self.implementation, 'evaluate'):
            return getattr(self.implementation, 'evaluate')(*args, **kwargs)
        return super().evaluate(*args, **kwargs)
    
    def select(self, *args, **kwargs):
        """
        Select operation.
        
        Args:
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.
            
        Returns:
            The result of the select operation.
        """
        if self.implementation and hasattr(self.implementation, 'select'):
            return getattr(self.implementation, 'select')(*args, **kwargs)
        return super().select(*args, **kwargs)
    
    def get_state(self, *args, **kwargs):
        """
        Get state operation.
        
        Args:
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.
            
        Returns:
            The result of the get_state operation.
        """
        if self.implementation and hasattr(self.implementation, 'get_state'):
            return getattr(self.implementation, 'get_state')(*args, **kwargs)
        return super().get_state(*args, **kwargs)
    
