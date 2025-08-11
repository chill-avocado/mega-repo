# Adapter for AgentForge/Agent
from functions.agent_frameworks.AgentForge.src.agentforge.agent import Agent
from ..agent import Agent

class AgentforgeAgentAdapter(Agent):
    """Adapter for AgentForge/Agent."""
    
    def __init__(self, config=None):
        """Initialize the adapter."""
        super().__init__(config)
        self.implementation = Agent(**config) if config else Agent()
    
    def plan(self, *args, **kwargs):
        """
        Plan operation.
        
        Args:
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.
            
        Returns:
            The result of the plan operation.
        """
        # Method not implemented in the original class
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
        if hasattr(self.implementation, 'run'):
            return getattr(self.implementation, 'run')(*args, **kwargs)
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
        if hasattr(self.implementation, 'run'):
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
        # Method not implemented in the original class
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
        # Method not implemented in the original class
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
        # Method not implemented in the original class
        return super().get_state(*args, **kwargs)
    
