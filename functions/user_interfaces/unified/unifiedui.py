# Unified interface for user_interfaces
from .userinterface import UserInterface

class UnifiedUI(UserInterface):
    """Unified interface for user_interfaces functionality."""
    
    def __init__(self, implementation=None, config=None):
        """
        Initialize the unified interface.
        
        Args:
            implementation: The specific implementation to use.
            config: Optional configuration.
        """
        super().__init__(config)
        self.implementation = implementation
    
    def render(self, *args, **kwargs):
        """
        Render operation.
        
        Args:
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.
            
        Returns:
            The result of the render operation.
        """
        if self.implementation and hasattr(self.implementation, 'render'):
            return getattr(self.implementation, 'render')(*args, **kwargs)
        return super().render(*args, **kwargs)
    
    def handle_input(self, *args, **kwargs):
        """
        Handle input operation.
        
        Args:
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.
            
        Returns:
            The result of the handle_input operation.
        """
        if self.implementation and hasattr(self.implementation, 'handle_input'):
            return getattr(self.implementation, 'handle_input')(*args, **kwargs)
        return super().handle_input(*args, **kwargs)
    
    def update(self, *args, **kwargs):
        """
        Update operation.
        
        Args:
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.
            
        Returns:
            The result of the update operation.
        """
        if self.implementation and hasattr(self.implementation, 'update'):
            return getattr(self.implementation, 'update')(*args, **kwargs)
        return super().update(*args, **kwargs)
    
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
    
