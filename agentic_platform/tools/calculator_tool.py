"""
Calculator tool implementation for the Agentic AI platform.

This module provides a calculator tool for the Agentic AI platform.
"""

import logging
import math
from typing import Any, Dict, List, Optional, Union

from .base_tool import BaseTool


class CalculatorTool(BaseTool):
    """
    Calculator tool implementation for the Agentic AI platform.
    
    This class provides a calculator tool that can perform basic arithmetic operations.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the calculator tool.
        
        Args:
            config: Configuration dictionary for the tool.
        """
        super().__init__(config)
    
    def get_description(self) -> str:
        """
        Get a description of this tool.
        
        Returns:
            Description of this tool.
        """
        return "A calculator tool that can perform basic arithmetic operations."
    
    def get_parameters(self) -> Dict[str, Dict[str, Any]]:
        """
        Get the parameters required by this tool.
        
        Returns:
            Dictionary mapping parameter names to parameter specifications.
        """
        return {
            "operation": {
                "type": "string",
                "description": "The operation to perform (add, subtract, multiply, divide, power, sqrt, sin, cos, tan).",
                "required": True
            },
            "a": {
                "type": "number",
                "description": "The first operand.",
                "required": True
            },
            "b": {
                "type": "number",
                "description": "The second operand (not required for sqrt, sin, cos, tan).",
                "required": False
            }
        }
    
    def execute(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute a calculator task.
        
        Args:
            task: Dictionary containing the task to execute.
        
        Returns:
            Dictionary containing the result of the task execution.
        """
        try:
            # Get the operation and operands
            operation = task.get('operation')
            a = task.get('a')
            b = task.get('b')
            
            if not operation:
                return {'success': False, 'error': 'No operation specified'}
            
            if a is None:
                return {'success': False, 'error': 'No first operand specified'}
            
            # Perform the operation
            if operation == 'add':
                if b is None:
                    return {'success': False, 'error': 'No second operand specified'}
                result = a + b
            elif operation == 'subtract':
                if b is None:
                    return {'success': False, 'error': 'No second operand specified'}
                result = a - b
            elif operation == 'multiply':
                if b is None:
                    return {'success': False, 'error': 'No second operand specified'}
                result = a * b
            elif operation == 'divide':
                if b is None:
                    return {'success': False, 'error': 'No second operand specified'}
                if b == 0:
                    return {'success': False, 'error': 'Division by zero'}
                result = a / b
            elif operation == 'power':
                if b is None:
                    return {'success': False, 'error': 'No second operand specified'}
                result = a ** b
            elif operation == 'sqrt':
                if a < 0:
                    return {'success': False, 'error': 'Cannot take square root of negative number'}
                result = math.sqrt(a)
            elif operation == 'sin':
                result = math.sin(a)
            elif operation == 'cos':
                result = math.cos(a)
            elif operation == 'tan':
                result = math.tan(a)
            else:
                return {'success': False, 'error': f'Unknown operation: {operation}'}
            
            return {'success': True, 'result': result}
        
        except Exception as e:
            self.logger.error(f"Failed to execute calculator task: {e}")
            return {'success': False, 'error': str(e)}
    
    def can_execute(self, task: Dict[str, Any]) -> bool:
        """
        Check if this tool can execute the given task.
        
        Args:
            task: Dictionary containing the task to check.
        
        Returns:
            True if this tool can execute the task, False otherwise.
        """
        # Check if the task has an operation
        if 'operation' not in task:
            return False
        
        # Check if the operation is supported
        operation = task['operation']
        supported_operations = ['add', 'subtract', 'multiply', 'divide', 'power', 'sqrt', 'sin', 'cos', 'tan']
        
        return operation in supported_operations