# Adapter for openagi/MemoryRagAction
from functions.cognitive_systems.openagi.src.openagi.actions.obs_rag import MemoryRagAction
from ..cognitivesystem import CognitiveSystem

class OpenagiMemoryRagActionAdapter(CognitiveSystem):
    """Adapter for openagi/MemoryRagAction."""
    
    def __init__(self, config=None):
        """Initialize the adapter."""
        super().__init__(config)
        self.implementation = MemoryRagAction(**config) if config else MemoryRagAction()
    
    def process(self, *args, **kwargs):
        """
        Process operation.
        
        Args:
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.
            
        Returns:
            The result of the process operation.
        """
        if hasattr(self.implementation, 'execute'):
            return getattr(self.implementation, 'execute')(*args, **kwargs)
        return super().process(*args, **kwargs)
    
    def learn(self, *args, **kwargs):
        """
        Learn operation.
        
        Args:
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.
            
        Returns:
            The result of the learn operation.
        """
        # Method not implemented in the original class
        return super().learn(*args, **kwargs)
    
    def reason(self, *args, **kwargs):
        """
        Reason operation.
        
        Args:
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.
            
        Returns:
            The result of the reason operation.
        """
        # Method not implemented in the original class
        return super().reason(*args, **kwargs)
    
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
    
