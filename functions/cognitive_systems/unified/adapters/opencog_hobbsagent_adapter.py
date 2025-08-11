# Adapter for opencog/HobbsAgent
from functions.cognitive_systems.opencog.opencog.nlp.anaphora.agents.hobbs import HobbsAgent
from ..cognitivesystem import CognitiveSystem

class OpencogHobbsAgentAdapter(CognitiveSystem):
    """Adapter for opencog/HobbsAgent."""
    
    def __init__(self, config=None):
        """Initialize the adapter."""
        super().__init__(config)
        self.implementation = HobbsAgent(**config) if config else HobbsAgent()
    
    def process(self, *args, **kwargs):
        """
        Process operation.
        
        Args:
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.
            
        Returns:
            The result of the process operation.
        """
        if hasattr(self.implementation, 'run'):
            return getattr(self.implementation, 'run')(*args, **kwargs)
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
    
