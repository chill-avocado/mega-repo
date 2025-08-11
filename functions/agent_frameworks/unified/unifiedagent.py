# Unified interface for agent_frameworks
from .agent import Agent

class UnifiedAgent(Agent):
    """Unified interface for agent_frameworks functionality."""
    
    def __init__(self, implementation=None, config=None):
        """
        Initialize the unified interface.
        
        Args:
            implementation: The specific implementation to use.
            config: Optional configuration.
        """
        super().__init__(config)
        self.implementation = implementation
    
    def plan(self, *args, **kwargs):
        """
        Plan operation.
        
        Args:
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.
            
        Returns:
            The result of the plan operation.
        """
        if self.implementation and hasattr(self.implementation, 'plan'):
            return getattr(self.implementation, 'plan')(*args, **kwargs)
        return super().plan(*args, **kwargs)
    
    def execute(self, *args, **kwargs):
        """
        Execute operation.
        
        Args:
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.
            
        Returns:
            The result of the execute operation.
        """
        if self.implementation and hasattr(self.implementation, 'execute'):
            return getattr(self.implementation, 'execute')(*args, **kwargs)
        return super().execute(*args, **kwargs)
    
    def run(self, *args, **kwargs):
        """
        Run operation.
        
        Args:
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.
            
        Returns:
            The result of the run operation.
        """
        if self.implementation and hasattr(self.implementation, 'run'):
            return getattr(self.implementation, 'run')(*args, **kwargs)
        return super().run(*args, **kwargs)
    
    def add_tool(self, *args, **kwargs):
        """
        Add tool operation.
        
        Args:
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.
            
        Returns:
            The result of the add_tool operation.
        """
        if self.implementation and hasattr(self.implementation, 'add_tool'):
            return getattr(self.implementation, 'add_tool')(*args, **kwargs)
        return super().add_tool(*args, **kwargs)
    
    def add_memory(self, *args, **kwargs):
        """
        Add memory operation.
        
        Args:
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.
            
        Returns:
            The result of the add_memory operation.
        """
        if self.implementation and hasattr(self.implementation, 'add_memory'):
            return getattr(self.implementation, 'add_memory')(*args, **kwargs)
        return super().add_memory(*args, **kwargs)
    
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
    
