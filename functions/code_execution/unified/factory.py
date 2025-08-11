# Factory for code_execution
from .unifiedcodeexecutor import UnifiedCodeExecutor
from .adapters.automata_toolevaluationharness_adapter import AutomataToolEvaluationHarnessAdapter
from .adapters.automata_agentevaluationharness_adapter import AutomataAgentEvaluationHarnessAdapter
from .adapters.automata_pycodewritertoolkitbuilder_adapter import AutomataPyCodeWriterToolkitBuilderAdapter

class CodeExecutionFactory:
    """Factory for creating code_execution implementations."""
    
    @staticmethod
    def create(implementation=None, config=None):
        """
        Create a code_execution implementation.
        
        Args:
            implementation (str): The name of the implementation to create.
            config (dict): Optional configuration.
            
        Returns:
            UnifiedCodeExecutor: A unified interface for the implementation.
        """
        config = config or {}
        
        # Create the appropriate implementation
        if implementation == 'automata/ToolEvaluationHarness':
            return UnifiedCodeExecutor(AutomataToolEvaluationHarnessAdapter(config), config)
        elif implementation == 'automata/AgentEvaluationHarness':
            return UnifiedCodeExecutor(AutomataAgentEvaluationHarnessAdapter(config), config)
        elif implementation == 'automata/PyCodeWriterToolkitBuilder':
            return UnifiedCodeExecutor(AutomataPyCodeWriterToolkitBuilderAdapter(config), config)
        else:
            # Default to the first implementation
            return UnifiedCodeExecutor(AutomataToolEvaluationHarnessAdapter(config), config)
