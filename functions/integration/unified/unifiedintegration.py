# Unified interface for integration
from .integration import Integration

class UnifiedIntegration(Integration):
    """Unified interface for integration functionality."""
    
    def __init__(self, implementation=None, config=None):
        """
        Initialize the unified interface.
        
        Args:
            implementation: The specific implementation to use.
            config: Optional configuration.
        """
        super().__init__(config)
        self.implementation = implementation
    
    def connect(self, *args, **kwargs):
        """
        Connect operation.
        
        Args:
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.
            
        Returns:
            The result of the connect operation.
        """
        if self.implementation and hasattr(self.implementation, 'connect'):
            return getattr(self.implementation, 'connect')(*args, **kwargs)
        return super().connect(*args, **kwargs)
    
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
    
    def transform(self, *args, **kwargs):
        """
        Transform operation.
        
        Args:
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.
            
        Returns:
            The result of the transform operation.
        """
        if self.implementation and hasattr(self.implementation, 'transform'):
            return getattr(self.implementation, 'transform')(*args, **kwargs)
        return super().transform(*args, **kwargs)
    
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
    
