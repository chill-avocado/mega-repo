# Factory for documentation
from .unifieddocumentation import UnifiedDocumentation
from .adapters.placeholder_documentation_adapter import PlaceholderDocumentationAdapter

class DocumentationFactory:
    """Factory for creating documentation implementations."""
    
    @staticmethod
    def create(implementation=None, config=None):
        """
        Create a documentation implementation.
        
        Args:
            implementation (str): The name of the implementation to create.
            config (dict): Optional configuration.
            
        Returns:
            UnifiedDocumentation: A unified interface for the implementation.
        """
        config = config or {}
        
        # Create the appropriate implementation
        if implementation == 'placeholder/Documentation':
            return UnifiedDocumentation(PlaceholderDocumentationAdapter(config), config)
        else:
            # Default to the first implementation
            return UnifiedDocumentation(PlaceholderDocumentationAdapter(config), config)
