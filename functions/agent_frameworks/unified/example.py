# Example usage of the unified interface for agent_frameworks
from .factory import AgentFrameworksFactory

def main():
    """Example usage of the unified interface."""
    # Create a unified interface with the default implementation
    unified = AgentFrameworksFactory.create()
    
    # Use the unified interface
    try:
        # Example usage of plan
        result = unified.plan()
        print(f"plan result: {result}")
    except NotImplementedError:
        print(f"plan not implemented in the default implementation")
    
    try:
        # Example usage of execute
        result = unified.execute()
        print(f"execute result: {result}")
    except NotImplementedError:
        print(f"execute not implemented in the default implementation")
    
    try:
        # Example usage of run
        result = unified.run()
        print(f"run result: {result}")
    except NotImplementedError:
        print(f"run not implemented in the default implementation")
    
    try:
        # Example usage of add_tool
        result = unified.add_tool()
        print(f"add_tool result: {result}")
    except NotImplementedError:
        print(f"add_tool not implemented in the default implementation")
    
    try:
        # Example usage of add_memory
        result = unified.add_memory()
        print(f"add_memory result: {result}")
    except NotImplementedError:
        print(f"add_memory not implemented in the default implementation")
    
    try:
        # Example usage of get_state
        result = unified.get_state()
        print(f"get_state result: {result}")
    except NotImplementedError:
        print(f"get_state not implemented in the default implementation")
    
    # Create a unified interface with the Free-Auto-GPT/Agent implementation
    unified = AgentFrameworksFactory.create('Free-Auto-GPT/Agent')
    
    # Use the unified interface
    try:
        # Example usage
        result = unified.plan()
        print(f"Free-Auto-GPT/Agent plan result: {result}")
    except NotImplementedError:
        print(f"plan not implemented in Free-Auto-GPT/Agent")
    
    # Create a unified interface with the AgentForge/Agent implementation
    unified = AgentFrameworksFactory.create('AgentForge/Agent')
    
    # Use the unified interface
    try:
        # Example usage
        result = unified.plan()
        print(f"AgentForge/Agent plan result: {result}")
    except NotImplementedError:
        print(f"plan not implemented in AgentForge/Agent")
    
    # Create a unified interface with the autogen/SingleThreadedAgentRuntime implementation
    unified = AgentFrameworksFactory.create('autogen/SingleThreadedAgentRuntime')
    
    # Use the unified interface
    try:
        # Example usage
        result = unified.plan()
        print(f"autogen/SingleThreadedAgentRuntime plan result: {result}")
    except NotImplementedError:
        print(f"plan not implemented in autogen/SingleThreadedAgentRuntime")
    

if __name__ == "__main__":
    main()
