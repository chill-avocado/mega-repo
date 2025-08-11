# Adapter for automata/ToolEvaluationHarness
from functions.code_execution.automata.automata.eval.tool.tool_eval_harness import ToolEvaluationHarness
from ..codeexecutor import CodeExecutor

class AutomataToolEvaluationHarnessAdapter(CodeExecutor):
    """Adapter for automata/ToolEvaluationHarness."""
    
    def __init__(self, config=None):
        """Initialize the adapter."""
        super().__init__(config)
        self.implementation = ToolEvaluationHarness(**config) if config else ToolEvaluationHarness()
    
    def execute(self, *args, **kwargs):
        """
        Execute operation.
        
        Args:
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.
            
        Returns:
            The result of the execute operation.
        """
        # Method not implemented in the original class
        return super().execute(*args, **kwargs)
    
    def analyze(self, *args, **kwargs):
        """
        Analyze operation.
        
        Args:
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.
            
        Returns:
            The result of the analyze operation.
        """
        if hasattr(self.implementation, 'evaluate'):
            return getattr(self.implementation, 'evaluate')(*args, **kwargs)
        return super().analyze(*args, **kwargs)
    
    def generate(self, *args, **kwargs):
        """
        Generate operation.
        
        Args:
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.
            
        Returns:
            The result of the generate operation.
        """
        # Method not implemented in the original class
        return super().generate(*args, **kwargs)
    
    def get_state(self, *args, **kwargs):
        """
        Get state operation.
        
        Args:
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.
            
        Returns:
            The result of the get_state operation.
        """
        # Method not implemented in the original class
        return super().get_state(*args, **kwargs)
    
