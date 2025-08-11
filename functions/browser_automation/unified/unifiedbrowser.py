# Unified interface for browser_automation
from .browserautomation import BrowserAutomation

class UnifiedBrowser(BrowserAutomation):
    """Unified interface for browser_automation functionality."""
    
    def __init__(self, implementation=None, config=None):
        """
        Initialize the unified interface.
        
        Args:
            implementation: The specific implementation to use.
            config: Optional configuration.
        """
        super().__init__(config)
        self.implementation = implementation
    
    def navigate(self, *args, **kwargs):
        """
        Navigate operation.
        
        Args:
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.
            
        Returns:
            The result of the navigate operation.
        """
        if self.implementation and hasattr(self.implementation, 'navigate'):
            return getattr(self.implementation, 'navigate')(*args, **kwargs)
        return super().navigate(*args, **kwargs)
    
    def click(self, *args, **kwargs):
        """
        Click operation.
        
        Args:
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.
            
        Returns:
            The result of the click operation.
        """
        if self.implementation and hasattr(self.implementation, 'click'):
            return getattr(self.implementation, 'click')(*args, **kwargs)
        return super().click(*args, **kwargs)
    
    def type(self, *args, **kwargs):
        """
        Type operation.
        
        Args:
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.
            
        Returns:
            The result of the type operation.
        """
        if self.implementation and hasattr(self.implementation, 'type'):
            return getattr(self.implementation, 'type')(*args, **kwargs)
        return super().type(*args, **kwargs)
    
    def extract_content(self, *args, **kwargs):
        """
        Extract content operation.
        
        Args:
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.
            
        Returns:
            The result of the extract_content operation.
        """
        if self.implementation and hasattr(self.implementation, 'extract_content'):
            return getattr(self.implementation, 'extract_content')(*args, **kwargs)
        return super().extract_content(*args, **kwargs)
    
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
    
