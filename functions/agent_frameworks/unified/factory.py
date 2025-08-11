# Factory for agent_frameworks
from .unifiedagent import UnifiedAgent
from .adapters.free-auto-gpt_agent_adapter import Free_Auto_GptAgentAdapter
from .adapters.agentforge_agent_adapter import AgentforgeAgentAdapter
from .adapters.autogen_singlethreadedagentruntime_adapter import AutogenSingleThreadedAgentRuntimeAdapter

class AgentFrameworksFactory:
    """Factory for creating agent_frameworks implementations."""
    
    @staticmethod
    def create(implementation=None, config=None):
        """
        Create a agent_frameworks implementation.
        
        Args:
            implementation (str): The name of the implementation to create.
            config (dict): Optional configuration.
            
        Returns:
            UnifiedAgent: A unified interface for the implementation.
        """
        config = config or {}
        
        # Create the appropriate implementation
        if implementation == 'Free-Auto-GPT/Agent':
            return UnifiedAgent(Free_Auto_GptAgentAdapter(config), config)
        elif implementation == 'AgentForge/Agent':
            return UnifiedAgent(AgentforgeAgentAdapter(config), config)
        elif implementation == 'autogen/SingleThreadedAgentRuntime':
            return UnifiedAgent(AutogenSingleThreadedAgentRuntimeAdapter(config), config)
        else:
            # Default to the first implementation
            return UnifiedAgent(Free_Auto_GptAgentAdapter(config), config)
