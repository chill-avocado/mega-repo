# Base class for agent_frameworks
class Agent:
    """Base class for agent_frameworks functionality."""
    
    def __init__(self, config=None):
        """Initialize with optional configuration."""
        self.config = config or {}
        self.state = "initialized"
    
    def plan(self, *args, **kwargs):
        """
        Plan operation.
        
        Args:
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.
            
        Returns:
            The result of the plan operation.
        """
        raise NotImplementedError("Subclasses must implement plan()")
    
    def execute(self, *args, **kwargs):
        """
        Execute operation.
        
        Args:
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.
            
        Returns:
            The result of the execute operation.
        """
        raise NotImplementedError("Subclasses must implement execute()")
    
    def run(self, *args, **kwargs):
        """
        Run operation.
        
        Args:
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.
            
        Returns:
            The result of the run operation.
        """
        raise NotImplementedError("Subclasses must implement run()")
    
    def add_tool(self, *args, **kwargs):
        """
        Add tool operation.
        
        Args:
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.
            
        Returns:
            The result of the add_tool operation.
        """
        raise NotImplementedError("Subclasses must implement add_tool()")
    
    def add_memory(self, *args, **kwargs):
        """
        Add memory operation.
        
        Args:
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.
            
        Returns:
            The result of the add_memory operation.
        """
        raise NotImplementedError("Subclasses must implement add_memory()")
    
    def get_state(self, *args, **kwargs):
        """
        Get state operation.
        
        Args:
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.
            
        Returns:
            The result of the get_state operation.
        """
        raise NotImplementedError("Subclasses must implement get_state()")
    
