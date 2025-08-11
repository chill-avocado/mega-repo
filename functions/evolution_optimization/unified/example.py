# Example usage of the unified interface for evolution_optimization
from .factory import EvolutionOptimizationFactory

def main():
    """Example usage of the unified interface."""
    # Create a unified interface with the default implementation
    unified = EvolutionOptimizationFactory.create()
    
    # Use the unified interface
    try:
        # Example usage of evolve
        result = unified.evolve()
        print(f"evolve result: {result}")
    except NotImplementedError:
        print(f"evolve not implemented in the default implementation")
    
    try:
        # Example usage of evaluate
        result = unified.evaluate()
        print(f"evaluate result: {result}")
    except NotImplementedError:
        print(f"evaluate not implemented in the default implementation")
    
    try:
        # Example usage of select
        result = unified.select()
        print(f"select result: {result}")
    except NotImplementedError:
        print(f"select not implemented in the default implementation")
    
    try:
        # Example usage of get_state
        result = unified.get_state()
        print(f"get_state result: {result}")
    except NotImplementedError:
        print(f"get_state not implemented in the default implementation")
    
    # Create a unified interface with the openevolve/BulletproofMetalEvaluator implementation
    unified = EvolutionOptimizationFactory.create('openevolve/BulletproofMetalEvaluator')
    
    # Use the unified interface
    try:
        # Example usage
        result = unified.evolve()
        print(f"openevolve/BulletproofMetalEvaluator evolve result: {result}")
    except NotImplementedError:
        print(f"evolve not implemented in openevolve/BulletproofMetalEvaluator")
    
    # Create a unified interface with the openevolve/MLIRAttentionEvaluator implementation
    unified = EvolutionOptimizationFactory.create('openevolve/MLIRAttentionEvaluator')
    
    # Use the unified interface
    try:
        # Example usage
        result = unified.evolve()
        print(f"openevolve/MLIRAttentionEvaluator evolve result: {result}")
    except NotImplementedError:
        print(f"evolve not implemented in openevolve/MLIRAttentionEvaluator")
    
    # Create a unified interface with the openevolve/ProgramDatabase implementation
    unified = EvolutionOptimizationFactory.create('openevolve/ProgramDatabase')
    
    # Use the unified interface
    try:
        # Example usage
        result = unified.evolve()
        print(f"openevolve/ProgramDatabase evolve result: {result}")
    except NotImplementedError:
        print(f"evolve not implemented in openevolve/ProgramDatabase")
    

if __name__ == "__main__":
    main()
