# Unified interface for os_interaction
from .osinteraction import OSInteraction

class UnifiedOSInteraction(OSInteraction):
    """Unified interface for os_interaction functionality."""
    
    def __init__(self, implementation=None, config=None):
        """
        Initialize the unified interface.
        
        Args:
            implementation: The specific implementation to use.
            config: Optional configuration.
        """
        super().__init__(config)
        self.implementation = implementation
    
    def capture_screen(self, *args, **kwargs):
        """
        Capture screen operation.
        
        Args:
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.
            
        Returns:
            The result of the capture_screen operation.
        """
        if self.implementation and hasattr(self.implementation, 'capture_screen'):
            return getattr(self.implementation, 'capture_screen')(*args, **kwargs)
        return super().capture_screen(*args, **kwargs)
    
    def control_keyboard(self, *args, **kwargs):
        """
        Control keyboard operation.
        
        Args:
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.
            
        Returns:
            The result of the control_keyboard operation.
        """
        if self.implementation and hasattr(self.implementation, 'control_keyboard'):
            return getattr(self.implementation, 'control_keyboard')(*args, **kwargs)
        return super().control_keyboard(*args, **kwargs)
    
    def control_mouse(self, *args, **kwargs):
        """
        Control mouse operation.
        
        Args:
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.
            
        Returns:
            The result of the control_mouse operation.
        """
        if self.implementation and hasattr(self.implementation, 'control_mouse'):
            return getattr(self.implementation, 'control_mouse')(*args, **kwargs)
        return super().control_mouse(*args, **kwargs)
    
    def execute_command(self, *args, **kwargs):
        """
        Execute command operation.
        
        Args:
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.
            
        Returns:
            The result of the execute_command operation.
        """
        if self.implementation and hasattr(self.implementation, 'execute_command'):
            return getattr(self.implementation, 'execute_command')(*args, **kwargs)
        return super().execute_command(*args, **kwargs)
    
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
    
