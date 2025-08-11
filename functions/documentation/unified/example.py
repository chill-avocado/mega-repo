# Example usage of the unified interface for documentation
from .factory import DocumentationFactory

def main():
    """Example usage of the unified interface."""
    # Create a unified interface with the default implementation
    unified = DocumentationFactory.create()
    
    # Use the unified interface
    try:
        # Example usage of get_documentation
        result = unified.get_documentation()
        print(f"get_documentation result: {result}")
    except NotImplementedError:
        print(f"get_documentation not implemented in the default implementation")
    
    try:
        # Example usage of search
        result = unified.search()
        print(f"search result: {result}")
    except NotImplementedError:
        print(f"search not implemented in the default implementation")
    
    try:
        # Example usage of generate_documentation
        result = unified.generate_documentation()
        print(f"generate_documentation result: {result}")
    except NotImplementedError:
        print(f"generate_documentation not implemented in the default implementation")
    
    try:
        # Example usage of get_state
        result = unified.get_state()
        print(f"get_state result: {result}")
    except NotImplementedError:
        print(f"get_state not implemented in the default implementation")
    
    # Create a unified interface with the placeholder/Documentation implementation
    unified = DocumentationFactory.create('placeholder/Documentation')
    
    # Use the unified interface
    try:
        # Example usage
        result = unified.get_documentation()
        print(f"placeholder/Documentation get_documentation result: {result}")
    except NotImplementedError:
        print(f"get_documentation not implemented in placeholder/Documentation")
    

if __name__ == "__main__":
    main()
