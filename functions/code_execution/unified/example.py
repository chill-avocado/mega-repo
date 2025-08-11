# Example usage of the unified interface for code_execution
from .factory import CodeExecutionFactory

def main():
    """Example usage of the unified interface."""
    # Create a unified interface with the default implementation
    unified = CodeExecutionFactory.create()
    
    # Use the unified interface
    try:
        # Example usage of execute
        result = unified.execute()
        print(f"execute result: {result}")
    except NotImplementedError:
        print(f"execute not implemented in the default implementation")
    
    try:
        # Example usage of analyze
        result = unified.analyze()
        print(f"analyze result: {result}")
    except NotImplementedError:
        print(f"analyze not implemented in the default implementation")
    
    try:
        # Example usage of generate
        result = unified.generate()
        print(f"generate result: {result}")
    except NotImplementedError:
        print(f"generate not implemented in the default implementation")
    
    try:
        # Example usage of get_state
        result = unified.get_state()
        print(f"get_state result: {result}")
    except NotImplementedError:
        print(f"get_state not implemented in the default implementation")
    
    # Create a unified interface with the automata/ToolEvaluationHarness implementation
    unified = CodeExecutionFactory.create('automata/ToolEvaluationHarness')
    
    # Use the unified interface
    try:
        # Example usage
        result = unified.execute()
        print(f"automata/ToolEvaluationHarness execute result: {result}")
    except NotImplementedError:
        print(f"execute not implemented in automata/ToolEvaluationHarness")
    
    # Create a unified interface with the automata/AgentEvaluationHarness implementation
    unified = CodeExecutionFactory.create('automata/AgentEvaluationHarness')
    
    # Use the unified interface
    try:
        # Example usage
        result = unified.execute()
        print(f"automata/AgentEvaluationHarness execute result: {result}")
    except NotImplementedError:
        print(f"execute not implemented in automata/AgentEvaluationHarness")
    
    # Create a unified interface with the automata/PyCodeWriterToolkitBuilder implementation
    unified = CodeExecutionFactory.create('automata/PyCodeWriterToolkitBuilder')
    
    # Use the unified interface
    try:
        # Example usage
        result = unified.execute()
        print(f"automata/PyCodeWriterToolkitBuilder execute result: {result}")
    except NotImplementedError:
        print(f"execute not implemented in automata/PyCodeWriterToolkitBuilder")
    

if __name__ == "__main__":
    main()
