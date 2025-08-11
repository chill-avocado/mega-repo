# Factory for evolution_optimization
from .unifiedevolutionoptimizer import UnifiedEvolutionOptimizer
from .adapters.openevolve_bulletproofmetalevaluator_adapter import OpenevolveBulletproofMetalEvaluatorAdapter
from .adapters.openevolve_mlirattentionevaluator_adapter import OpenevolveMLIRAttentionEvaluatorAdapter
from .adapters.openevolve_programdatabase_adapter import OpenevolveProgramDatabaseAdapter

class EvolutionOptimizationFactory:
    """Factory for creating evolution_optimization implementations."""
    
    @staticmethod
    def create(implementation=None, config=None):
        """
        Create a evolution_optimization implementation.
        
        Args:
            implementation (str): The name of the implementation to create.
            config (dict): Optional configuration.
            
        Returns:
            UnifiedEvolutionOptimizer: A unified interface for the implementation.
        """
        config = config or {}
        
        # Create the appropriate implementation
        if implementation == 'openevolve/BulletproofMetalEvaluator':
            return UnifiedEvolutionOptimizer(OpenevolveBulletproofMetalEvaluatorAdapter(config), config)
        elif implementation == 'openevolve/MLIRAttentionEvaluator':
            return UnifiedEvolutionOptimizer(OpenevolveMLIRAttentionEvaluatorAdapter(config), config)
        elif implementation == 'openevolve/ProgramDatabase':
            return UnifiedEvolutionOptimizer(OpenevolveProgramDatabaseAdapter(config), config)
        else:
            # Default to the first implementation
            return UnifiedEvolutionOptimizer(OpenevolveBulletproofMetalEvaluatorAdapter(config), config)
