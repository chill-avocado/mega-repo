# Factory for integration
from .unifiedintegration import UnifiedIntegration
from .adapters.unification_vncclient_adapter import UnificationVNCClientAdapter
from .adapters.unification_deferredlogger_adapter import UnificationDeferredLoggerAdapter
from .adapters.unification_operatingsystem_adapter import UnificationOperatingSystemAdapter

class IntegrationFactory:
    """Factory for creating integration implementations."""
    
    @staticmethod
    def create(implementation=None, config=None):
        """
        Create a integration implementation.
        
        Args:
            implementation (str): The name of the implementation to create.
            config (dict): Optional configuration.
            
        Returns:
            UnifiedIntegration: A unified interface for the implementation.
        """
        config = config or {}
        
        # Create the appropriate implementation
        if implementation == 'Unification/VNCClient':
            return UnifiedIntegration(UnificationVNCClientAdapter(config), config)
        elif implementation == 'Unification/DeferredLogger':
            return UnifiedIntegration(UnificationDeferredLoggerAdapter(config), config)
        elif implementation == 'Unification/OperatingSystem':
            return UnifiedIntegration(UnificationOperatingSystemAdapter(config), config)
        else:
            # Default to the first implementation
            return UnifiedIntegration(UnificationVNCClientAdapter(config), config)
