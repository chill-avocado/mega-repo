# Factory for nlp
from .unifiednlp import UnifiedNLP
from .adapters.senpai_agent_adapter import SenpaiAgentAdapter
from .adapters.senpai_agentlogger_adapter import SenpaiAgentLoggerAdapter

class NlpFactory:
    """Factory for creating nlp implementations."""
    
    @staticmethod
    def create(implementation=None, config=None):
        """
        Create a nlp implementation.
        
        Args:
            implementation (str): The name of the implementation to create.
            config (dict): Optional configuration.
            
        Returns:
            UnifiedNLP: A unified interface for the implementation.
        """
        config = config or {}
        
        # Create the appropriate implementation
        if implementation == 'senpai/Agent':
            return UnifiedNLP(SenpaiAgentAdapter(config), config)
        elif implementation == 'senpai/AgentLogger':
            return UnifiedNLP(SenpaiAgentLoggerAdapter(config), config)
        else:
            # Default to the first implementation
            return UnifiedNLP(SenpaiAgentAdapter(config), config)
