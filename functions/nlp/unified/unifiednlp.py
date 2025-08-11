# Unified interface for nlp
from .nlpprocessor import NLPProcessor

class UnifiedNLP(NLPProcessor):
    """Unified interface for nlp functionality."""
    
    def __init__(self, implementation=None, config=None):
        """
        Initialize the unified interface.
        
        Args:
            implementation: The specific implementation to use.
            config: Optional configuration.
        """
        super().__init__(config)
        self.implementation = implementation
    
    def process_text(self, *args, **kwargs):
        """
        Process text operation.
        
        Args:
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.
            
        Returns:
            The result of the process_text operation.
        """
        if self.implementation and hasattr(self.implementation, 'process_text'):
            return getattr(self.implementation, 'process_text')(*args, **kwargs)
        return super().process_text(*args, **kwargs)
    
    def generate_text(self, *args, **kwargs):
        """
        Generate text operation.
        
        Args:
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.
            
        Returns:
            The result of the generate_text operation.
        """
        if self.implementation and hasattr(self.implementation, 'generate_text'):
            return getattr(self.implementation, 'generate_text')(*args, **kwargs)
        return super().generate_text(*args, **kwargs)
    
    def analyze_text(self, *args, **kwargs):
        """
        Analyze text operation.
        
        Args:
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.
            
        Returns:
            The result of the analyze_text operation.
        """
        if self.implementation and hasattr(self.implementation, 'analyze_text'):
            return getattr(self.implementation, 'analyze_text')(*args, **kwargs)
        return super().analyze_text(*args, **kwargs)
    
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
    
