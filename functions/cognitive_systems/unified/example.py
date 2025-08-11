# Example usage of the unified interface for cognitive_systems
from .factory import CognitiveSystemsFactory

def main():
    """Example usage of the unified interface."""
    # Create a unified interface with the default implementation
    unified = CognitiveSystemsFactory.create()
    
    # Use the unified interface
    try:
        # Example usage of process
        result = unified.process()
        print(f"process result: {result}")
    except NotImplementedError:
        print(f"process not implemented in the default implementation")
    
    try:
        # Example usage of learn
        result = unified.learn()
        print(f"learn result: {result}")
    except NotImplementedError:
        print(f"learn not implemented in the default implementation")
    
    try:
        # Example usage of reason
        result = unified.reason()
        print(f"reason result: {result}")
    except NotImplementedError:
        print(f"reason not implemented in the default implementation")
    
    try:
        # Example usage of get_state
        result = unified.get_state()
        print(f"get_state result: {result}")
    except NotImplementedError:
        print(f"get_state not implemented in the default implementation")
    
    # Create a unified interface with the openagi/MemoryRagAction implementation
    unified = CognitiveSystemsFactory.create('openagi/MemoryRagAction')
    
    # Use the unified interface
    try:
        # Example usage
        result = unified.process()
        print(f"openagi/MemoryRagAction process result: {result}")
    except NotImplementedError:
        print(f"process not implemented in openagi/MemoryRagAction")
    
    # Create a unified interface with the openagi/BaseMemory implementation
    unified = CognitiveSystemsFactory.create('openagi/BaseMemory')
    
    # Use the unified interface
    try:
        # Example usage
        result = unified.process()
        print(f"openagi/BaseMemory process result: {result}")
    except NotImplementedError:
        print(f"process not implemented in openagi/BaseMemory")
    
    # Create a unified interface with the opencog/HobbsAgent implementation
    unified = CognitiveSystemsFactory.create('opencog/HobbsAgent')
    
    # Use the unified interface
    try:
        # Example usage
        result = unified.process()
        print(f"opencog/HobbsAgent process result: {result}")
    except NotImplementedError:
        print(f"process not implemented in opencog/HobbsAgent")
    

if __name__ == "__main__":
    main()
