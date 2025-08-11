# Unified interface for cognitive_systems
from .cognitivesystem import CognitiveSystem

class UnifiedCognitiveSystem(CognitiveSystem):
    """Unified interface for cognitive_systems functionality."""
    
    def __init__(self, implementation=None, config=None):
        """
        Initialize the unified interface.
        
        Args:
            implementation: The specific implementation to use.
            config: Optional configuration.
        """
        super().__init__(config)
        self.implementation = implementation
    
    def process(self, *args, **kwargs):
        """
        Process operation.
        
        Args:
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.
            
        Returns:
            The result of the process operation.
        """
        if self.implementation and hasattr(self.implementation, 'process'):
            return getattr(self.implementation, 'process')(*args, **kwargs)
        return super().process(*args, **kwargs)
    
    def learn(self, *args, **kwargs):
        """
        Learn operation.
        
        Args:
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.
            
        Returns:
            The result of the learn operation.
        """
        if self.implementation and hasattr(self.implementation, 'learn'):
            return getattr(self.implementation, 'learn')(*args, **kwargs)
        return super().learn(*args, **kwargs)
    
    def reason(self, *args, **kwargs):
        """
        Reason operation.
        
        Args:
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.
            
        Returns:
            The result of the reason operation.
        """
        if self.implementation and hasattr(self.implementation, 'reason'):
            return getattr(self.implementation, 'reason')(*args, **kwargs)
        return super().reason(*args, **kwargs)
    
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
    
