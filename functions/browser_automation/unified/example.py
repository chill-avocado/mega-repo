# Example usage of the unified interface for browser_automation
from .factory import BrowserAutomationFactory

def main():
    """Example usage of the unified interface."""
    # Create a unified interface with the default implementation
    unified = BrowserAutomationFactory.create()
    
    # Use the unified interface
    try:
        # Example usage of navigate
        result = unified.navigate()
        print(f"navigate result: {result}")
    except NotImplementedError:
        print(f"navigate not implemented in the default implementation")
    
    try:
        # Example usage of click
        result = unified.click()
        print(f"click result: {result}")
    except NotImplementedError:
        print(f"click not implemented in the default implementation")
    
    try:
        # Example usage of type
        result = unified.type()
        print(f"type result: {result}")
    except NotImplementedError:
        print(f"type not implemented in the default implementation")
    
    try:
        # Example usage of extract_content
        result = unified.extract_content()
        print(f"extract_content result: {result}")
    except NotImplementedError:
        print(f"extract_content not implemented in the default implementation")
    
    try:
        # Example usage of get_state
        result = unified.get_state()
        print(f"get_state result: {result}")
    except NotImplementedError:
        print(f"get_state not implemented in the default implementation")
    
    # Create a unified interface with the browser-use/BrowserUseApp implementation
    unified = BrowserAutomationFactory.create('browser-use/BrowserUseApp')
    
    # Use the unified interface
    try:
        # Example usage
        result = unified.navigate()
        print(f"browser-use/BrowserUseApp navigate result: {result}")
    except NotImplementedError:
        print(f"navigate not implemented in browser-use/BrowserUseApp")
    
    # Create a unified interface with the browser-use/BrowserUseServer implementation
    unified = BrowserAutomationFactory.create('browser-use/BrowserUseServer')
    
    # Use the unified interface
    try:
        # Example usage
        result = unified.navigate()
        print(f"browser-use/BrowserUseServer navigate result: {result}")
    except NotImplementedError:
        print(f"navigate not implemented in browser-use/BrowserUseServer")
    
    # Create a unified interface with the browser-use/FileSystem implementation
    unified = BrowserAutomationFactory.create('browser-use/FileSystem')
    
    # Use the unified interface
    try:
        # Example usage
        result = unified.navigate()
        print(f"browser-use/FileSystem navigate result: {result}")
    except NotImplementedError:
        print(f"navigate not implemented in browser-use/FileSystem")
    

if __name__ == "__main__":
    main()
