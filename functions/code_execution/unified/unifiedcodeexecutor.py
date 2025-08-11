# Unified interface for code_execution
from .codeexecutor import CodeExecutor

class UnifiedCodeExecutor(CodeExecutor):
    """Unified interface for code_execution functionality."""
    
    def __init__(self, implementation=None, config=None):
        """
        Initialize the unified interface.
        
        Args:
            implementation: The specific implementation to use.
            config: Optional configuration.
        """
        super().__init__(config)
        self.implementation = implementation
    
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
    
    def analyze(self, *args, **kwargs):
        """
        Analyze operation.
        
        Args:
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.
            
        Returns:
            The result of the analyze operation.
        """
        if self.implementation and hasattr(self.implementation, 'analyze'):
            return getattr(self.implementation, 'analyze')(*args, **kwargs)
        return super().analyze(*args, **kwargs)
    
    def generate(self, *args, **kwargs):
        """
        Generate operation.
        
        Args:
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.
            
        Returns:
            The result of the generate operation.
        """
        if self.implementation and hasattr(self.implementation, 'generate'):
            return getattr(self.implementation, 'generate')(*args, **kwargs)
        return super().generate(*args, **kwargs)
    
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
    
