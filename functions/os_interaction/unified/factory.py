# Factory for os_interaction
from .unifiedosinteraction import UnifiedOSInteraction
from .adapters.self-operating-computer_operatingsystem_adapter import Self_Operating_ComputerOperatingSystemAdapter
from .adapters.mcp-remote-macos-use_vncclient_adapter import Mcp_Remote_Macos_UseVNCClientAdapter
from .adapters.macos-agent_deferredlogger_adapter import Macos_AgentDeferredLoggerAdapter

class OsInteractionFactory:
    """Factory for creating os_interaction implementations."""
    
    @staticmethod
    def create(implementation=None, config=None):
        """
        Create a os_interaction implementation.
        
        Args:
            implementation (str): The name of the implementation to create.
            config (dict): Optional configuration.
            
        Returns:
            UnifiedOSInteraction: A unified interface for the implementation.
        """
        config = config or {}
        
        # Create the appropriate implementation
        if implementation == 'self-operating-computer/OperatingSystem':
            return UnifiedOSInteraction(Self_Operating_ComputerOperatingSystemAdapter(config), config)
        elif implementation == 'mcp-remote-macos-use/VNCClient':
            return UnifiedOSInteraction(Mcp_Remote_Macos_UseVNCClientAdapter(config), config)
        elif implementation == 'MacOS-Agent/DeferredLogger':
            return UnifiedOSInteraction(Macos_AgentDeferredLoggerAdapter(config), config)
        else:
            # Default to the first implementation
            return UnifiedOSInteraction(Self_Operating_ComputerOperatingSystemAdapter(config), config)
