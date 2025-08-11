# Adapter for MacOS-Agent/DeferredLogger
from functions.os_interaction.MacOS-Agent.macos_agent_server import DeferredLogger
from ..osinteraction import OSInteraction

class Macos_AgentDeferredLoggerAdapter(OSInteraction):
    """Adapter for MacOS-Agent/DeferredLogger."""
    
    def __init__(self, config=None):
        """Initialize the adapter."""
        super().__init__(config)
        self.implementation = DeferredLogger(**config) if config else DeferredLogger()
    
    def capture_screen(self, *args, **kwargs):
        """
        Capture screen operation.
        
        Args:
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.
            
        Returns:
            The result of the capture_screen operation.
        """
        # Method not implemented in the original class
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
        # Method not implemented in the original class
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
        # Method not implemented in the original class
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
        # Method not implemented in the original class
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
        if hasattr(self.implementation, 'info'):
            return getattr(self.implementation, 'info')(*args, **kwargs)
        return super().get_state(*args, **kwargs)
    
