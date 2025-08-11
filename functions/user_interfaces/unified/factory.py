# Factory for user_interfaces
from .unifiedui import UnifiedUI
from .adapters.open-webui_custombuildhook_adapter import Open_WebuiCustomBuildHookAdapter
from .adapters.open-webui_persistentconfig_adapter import Open_WebuiPersistentConfigAdapter
from .adapters.open-webui_redisdict_adapter import Open_WebuiRedisDictAdapter

class UserInterfacesFactory:
    """Factory for creating user_interfaces implementations."""
    
    @staticmethod
    def create(implementation=None, config=None):
        """
        Create a user_interfaces implementation.
        
        Args:
            implementation (str): The name of the implementation to create.
            config (dict): Optional configuration.
            
        Returns:
            UnifiedUI: A unified interface for the implementation.
        """
        config = config or {}
        
        # Create the appropriate implementation
        if implementation == 'open-webui/CustomBuildHook':
            return UnifiedUI(Open_WebuiCustomBuildHookAdapter(config), config)
        elif implementation == 'open-webui/PersistentConfig':
            return UnifiedUI(Open_WebuiPersistentConfigAdapter(config), config)
        elif implementation == 'open-webui/RedisDict':
            return UnifiedUI(Open_WebuiRedisDictAdapter(config), config)
        else:
            # Default to the first implementation
            return UnifiedUI(Open_WebuiCustomBuildHookAdapter(config), config)
