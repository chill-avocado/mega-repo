# Factory for cognitive_systems
from .unifiedcognitivesystem import UnifiedCognitiveSystem
from .adapters.openagi_memoryragaction_adapter import OpenagiMemoryRagActionAdapter
from .adapters.openagi_basememory_adapter import OpenagiBaseMemoryAdapter
from .adapters.opencog_hobbsagent_adapter import OpencogHobbsAgentAdapter

class CognitiveSystemsFactory:
    """Factory for creating cognitive_systems implementations."""
    
    @staticmethod
    def create(implementation=None, config=None):
        """
        Create a cognitive_systems implementation.
        
        Args:
            implementation (str): The name of the implementation to create.
            config (dict): Optional configuration.
            
        Returns:
            UnifiedCognitiveSystem: A unified interface for the implementation.
        """
        config = config or {}
        
        # Create the appropriate implementation
        if implementation == 'openagi/MemoryRagAction':
            return UnifiedCognitiveSystem(OpenagiMemoryRagActionAdapter(config), config)
        elif implementation == 'openagi/BaseMemory':
            return UnifiedCognitiveSystem(OpenagiBaseMemoryAdapter(config), config)
        elif implementation == 'opencog/HobbsAgent':
            return UnifiedCognitiveSystem(OpencogHobbsAgentAdapter(config), config)
        else:
            # Default to the first implementation
            return UnifiedCognitiveSystem(OpenagiMemoryRagActionAdapter(config), config)
