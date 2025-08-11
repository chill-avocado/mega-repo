"""
Tool adapter for integrating external tool components into the Agentic AI platform.

This module provides an adapter for integrating external tool components into the
Agentic AI platform.
"""

import logging
import inspect
from typing import Any, Dict, List, Optional, Union

from ...interfaces.agent import Tool
from ..base import BaseAdapter


class ToolAdapter(Tool):
    """
    Tool adapter for integrating external tool components into the Agentic AI platform.
    
    This class provides an adapter that wraps an external tool component and exposes
    it through the Tool interface.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the tool adapter.
        
        Args:
            config: Configuration dictionary for the adapter.
        """
        self.logger = logging.getLogger(__name__)
        self.config = config or {}
        self.initialized = False
        self.state = {}
        
        # Get the component path and class from the config
        self.component_path = self.config.get('component_path')
        self.component_class = self.config.get('component_class')
        
        # Create the adapter
        self.adapter = BaseAdapter(self.component_path, self.component_class, self.config.get('component_config'))
        
        # Load the component
        self.component = self.adapter.load_component()
    
    def initialize(self, config: Dict[str, Any]) -> bool:
        """
        Initialize the tool with the given configuration.
        
        Args:
            config: Configuration dictionary for the tool.
        
        Returns:
            True if initialization was successful, False otherwise.
        """
        try:
            self.config.update(config)
            
            # Initialize the component if it has an initialize method
            if hasattr(self.component, 'initialize') and callable(self.component.initialize):
                self.component.initialize(config)
            
            self.initialized = True
            return True
        except Exception as e:
            self.logger.error(f"Failed to initialize tool adapter: {e}")
            return False
    
    def get_info(self) -> Dict[str, Any]:
        """
        Get information about the tool.
        
        Returns:
            Dictionary containing information about the tool.
        """
        return {
            'type': self.__class__.__name__,
            'component_path': self.component_path,
            'component_class': self.component_class,
            'description': self.get_description(),
            'initialized': self.initialized
        }
    
    def get_config(self) -> Dict[str, Any]:
        """
        Get the current configuration of the tool.
        
        Returns:
            Dictionary containing the current configuration.
        """
        return self.config
    
    def set_config(self, config: Dict[str, Any]) -> bool:
        """
        Set the configuration of the tool.
        
        Args:
            config: Dictionary containing the configuration to set.
        
        Returns:
            True if the configuration was set successfully, False otherwise.
        """
        try:
            self.config.update(config)
            
            # Set the component config if it has a set_config method
            if hasattr(self.component, 'set_config') and callable(self.component.set_config):
                self.component.set_config(config)
            
            return True
        except Exception as e:
            self.logger.error(f"Failed to set tool adapter configuration: {e}")
            return False
    
    def get_default_config(self) -> Dict[str, Any]:
        """
        Get the default configuration of the tool.
        
        Returns:
            Dictionary containing the default configuration.
        """
        # Get the component default config if it has a get_default_config method
        if hasattr(self.component, 'get_default_config') and callable(self.component.get_default_config):
            return self.component.get_default_config()
        
        return {}
    
    def get_state(self) -> Dict[str, Any]:
        """
        Get the current state of the tool.
        
        Returns:
            Dictionary containing the current state of the tool.
        """
        # Get the component state if it has a get_state method
        if hasattr(self.component, 'get_state') and callable(self.component.get_state):
            return self.component.get_state()
        
        return {
            'config': self.config,
            'state': self.state
        }
    
    def set_state(self, state: Dict[str, Any]) -> bool:
        """
        Set the state of the tool.
        
        Args:
            state: Dictionary containing the state to set.
        
        Returns:
            True if the state was set successfully, False otherwise.
        """
        try:
            # Set the component state if it has a set_state method
            if hasattr(self.component, 'set_state') and callable(self.component.set_state):
                return self.component.set_state(state)
            
            if 'config' in state:
                self.config.update(state['config'])
            
            if 'state' in state:
                self.state.update(state['state'])
            
            return True
        except Exception as e:
            self.logger.error(f"Failed to set tool adapter state: {e}")
            return False
    
    def get_description(self) -> str:
        """
        Get a description of this tool.
        
        Returns:
            Description of this tool.
        """
        # Get the component description if it has a description attribute
        if hasattr(self.component, 'description') and isinstance(self.component.description, str):
            return self.component.description
        
        # Get the component docstring
        if self.component.__doc__:
            return self.component.__doc__.strip()
        
        return "External tool component"
    
    def get_parameters(self) -> Dict[str, Dict[str, Any]]:
        """
        Get the parameters required by this tool.
        
        Returns:
            Dictionary mapping parameter names to parameter specifications.
        """
        # Get the component parameters if it has a get_parameters method
        if hasattr(self.component, 'get_parameters') and callable(self.component.get_parameters):
            return self.component.get_parameters()
        
        # Try to infer parameters from the execute method signature
        if hasattr(self.component, 'execute') and callable(self.component.execute):
            params = {}
            sig = inspect.signature(self.component.execute)
            
            for name, param in sig.parameters.items():
                if name == 'self':
                    continue
                
                param_info = {
                    'type': 'any',
                    'description': f"Parameter {name} for the tool",
                    'required': param.default == inspect.Parameter.empty
                }
                
                # Try to get type information
                if param.annotation != inspect.Parameter.empty:
                    if hasattr(param.annotation, '__name__'):
                        param_info['type'] = param.annotation.__name__.lower()
                    else:
                        param_info['type'] = str(param.annotation).lower()
                
                params[name] = param_info
            
            return params
        
        return {}
    
    def execute(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute a task.
        
        Args:
            task: Dictionary containing the task to execute.
        
        Returns:
            Dictionary containing the result of the task execution.
        """
        try:
            # Execute the task using the component if it has an execute method
            if hasattr(self.component, 'execute') and callable(self.component.execute):
                # Set the component attributes from the task
                for key, value in task.items():
                    if hasattr(self.component, key):
                        setattr(self.component, key, value)
                
                # Execute the component
                result = self.component.execute()
                
                # Return the result
                if isinstance(result, dict):
                    return result
                else:
                    return {'success': True, 'result': result}
            
            return {'success': False, 'error': 'Tool does not have an execute method'}
        except Exception as e:
            self.logger.error(f"Failed to execute task: {e}")
            return {'success': False, 'error': str(e)}
    
    def can_execute(self, task: Dict[str, Any]) -> bool:
        """
        Check if this tool can execute the given task.
        
        Args:
            task: Dictionary containing the task to check.
        
        Returns:
            True if this tool can execute the task, False otherwise.
        """
        # Check if the component can execute the task if it has a can_execute method
        if hasattr(self.component, 'can_execute') and callable(self.component.can_execute):
            return self.component.can_execute(task)
        
        # Check if the component has an execute method
        if not hasattr(self.component, 'execute') or not callable(self.component.execute):
            return False
        
        # Check if the task has all required parameters
        sig = inspect.signature(self.component.execute)
        
        for name, param in sig.parameters.items():
            if name == 'self':
                continue
            
            if param.default == inspect.Parameter.empty and name not in task:
                return False
        
        return True