# Example usage of the unified interface for os_interaction
from .factory import OsInteractionFactory

def main():
    """Example usage of the unified interface."""
    # Create a unified interface with the default implementation
    unified = OsInteractionFactory.create()
    
    # Use the unified interface
    try:
        # Example usage of capture_screen
        result = unified.capture_screen()
        print(f"capture_screen result: {result}")
    except NotImplementedError:
        print(f"capture_screen not implemented in the default implementation")
    
    try:
        # Example usage of control_keyboard
        result = unified.control_keyboard()
        print(f"control_keyboard result: {result}")
    except NotImplementedError:
        print(f"control_keyboard not implemented in the default implementation")
    
    try:
        # Example usage of control_mouse
        result = unified.control_mouse()
        print(f"control_mouse result: {result}")
    except NotImplementedError:
        print(f"control_mouse not implemented in the default implementation")
    
    try:
        # Example usage of execute_command
        result = unified.execute_command()
        print(f"execute_command result: {result}")
    except NotImplementedError:
        print(f"execute_command not implemented in the default implementation")
    
    try:
        # Example usage of get_state
        result = unified.get_state()
        print(f"get_state result: {result}")
    except NotImplementedError:
        print(f"get_state not implemented in the default implementation")
    
    # Create a unified interface with the self-operating-computer/OperatingSystem implementation
    unified = OsInteractionFactory.create('self-operating-computer/OperatingSystem')
    
    # Use the unified interface
    try:
        # Example usage
        result = unified.capture_screen()
        print(f"self-operating-computer/OperatingSystem capture_screen result: {result}")
    except NotImplementedError:
        print(f"capture_screen not implemented in self-operating-computer/OperatingSystem")
    
    # Create a unified interface with the mcp-remote-macos-use/VNCClient implementation
    unified = OsInteractionFactory.create('mcp-remote-macos-use/VNCClient')
    
    # Use the unified interface
    try:
        # Example usage
        result = unified.capture_screen()
        print(f"mcp-remote-macos-use/VNCClient capture_screen result: {result}")
    except NotImplementedError:
        print(f"capture_screen not implemented in mcp-remote-macos-use/VNCClient")
    
    # Create a unified interface with the MacOS-Agent/DeferredLogger implementation
    unified = OsInteractionFactory.create('MacOS-Agent/DeferredLogger')
    
    # Use the unified interface
    try:
        # Example usage
        result = unified.capture_screen()
        print(f"MacOS-Agent/DeferredLogger capture_screen result: {result}")
    except NotImplementedError:
        print(f"capture_screen not implemented in MacOS-Agent/DeferredLogger")
    

if __name__ == "__main__":
    main()
