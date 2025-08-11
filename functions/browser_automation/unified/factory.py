# Factory for browser_automation
from .unifiedbrowser import UnifiedBrowser
from .adapters.browser-use_browseruseapp_adapter import Browser_UseBrowserUseAppAdapter
from .adapters.browser-use_browseruseserver_adapter import Browser_UseBrowserUseServerAdapter
from .adapters.browser-use_filesystem_adapter import Browser_UseFileSystemAdapter

class BrowserAutomationFactory:
    """Factory for creating browser_automation implementations."""
    
    @staticmethod
    def create(implementation=None, config=None):
        """
        Create a browser_automation implementation.
        
        Args:
            implementation (str): The name of the implementation to create.
            config (dict): Optional configuration.
            
        Returns:
            UnifiedBrowser: A unified interface for the implementation.
        """
        config = config or {}
        
        # Create the appropriate implementation
        if implementation == 'browser-use/BrowserUseApp':
            return UnifiedBrowser(Browser_UseBrowserUseAppAdapter(config), config)
        elif implementation == 'browser-use/BrowserUseServer':
            return UnifiedBrowser(Browser_UseBrowserUseServerAdapter(config), config)
        elif implementation == 'browser-use/FileSystem':
            return UnifiedBrowser(Browser_UseFileSystemAdapter(config), config)
        else:
            # Default to the first implementation
            return UnifiedBrowser(Browser_UseBrowserUseAppAdapter(config), config)
