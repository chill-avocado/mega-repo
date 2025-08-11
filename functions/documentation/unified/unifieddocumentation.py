# Unified interface for documentation
from .documentation import Documentation

class UnifiedDocumentation(Documentation):
    """Unified interface for documentation functionality."""
    
    def __init__(self, implementation=None, config=None):
        """
        Initialize the unified interface.
        
        Args:
            implementation: The specific implementation to use.
            config: Optional configuration.
        """
        super().__init__(config)
        self.implementation = implementation
    
    def get_documentation(self, *args, **kwargs):
        """
        Get documentation operation.
        
        Args:
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.
            
        Returns:
            The result of the get_documentation operation.
        """
        if self.implementation and hasattr(self.implementation, 'get_documentation'):
            return getattr(self.implementation, 'get_documentation')(*args, **kwargs)
        return super().get_documentation(*args, **kwargs)
    
    def search(self, *args, **kwargs):
        """
        Search operation.
        
        Args:
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.
            
        Returns:
            The result of the search operation.
        """
        if self.implementation and hasattr(self.implementation, 'search'):
            return getattr(self.implementation, 'search')(*args, **kwargs)
        return super().search(*args, **kwargs)
    
    def generate_documentation(self, *args, **kwargs):
        """
        Generate documentation operation.
        
        Args:
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.
            
        Returns:
            The result of the generate_documentation operation.
        """
        if self.implementation and hasattr(self.implementation, 'generate_documentation'):
            return getattr(self.implementation, 'generate_documentation')(*args, **kwargs)
        return super().generate_documentation(*args, **kwargs)
    
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
    
