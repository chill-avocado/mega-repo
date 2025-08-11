# Example usage of the unified interface for nlp
from .factory import NlpFactory

def main():
    """Example usage of the unified interface."""
    # Create a unified interface with the default implementation
    unified = NlpFactory.create()
    
    # Use the unified interface
    try:
        # Example usage of process_text
        result = unified.process_text()
        print(f"process_text result: {result}")
    except NotImplementedError:
        print(f"process_text not implemented in the default implementation")
    
    try:
        # Example usage of generate_text
        result = unified.generate_text()
        print(f"generate_text result: {result}")
    except NotImplementedError:
        print(f"generate_text not implemented in the default implementation")
    
    try:
        # Example usage of analyze_text
        result = unified.analyze_text()
        print(f"analyze_text result: {result}")
    except NotImplementedError:
        print(f"analyze_text not implemented in the default implementation")
    
    try:
        # Example usage of get_state
        result = unified.get_state()
        print(f"get_state result: {result}")
    except NotImplementedError:
        print(f"get_state not implemented in the default implementation")
    
    # Create a unified interface with the senpai/Agent implementation
    unified = NlpFactory.create('senpai/Agent')
    
    # Use the unified interface
    try:
        # Example usage
        result = unified.process_text()
        print(f"senpai/Agent process_text result: {result}")
    except NotImplementedError:
        print(f"process_text not implemented in senpai/Agent")
    
    # Create a unified interface with the senpai/AgentLogger implementation
    unified = NlpFactory.create('senpai/AgentLogger')
    
    # Use the unified interface
    try:
        # Example usage
        result = unified.process_text()
        print(f"senpai/AgentLogger process_text result: {result}")
    except NotImplementedError:
        print(f"process_text not implemented in senpai/AgentLogger")
    

if __name__ == "__main__":
    main()
